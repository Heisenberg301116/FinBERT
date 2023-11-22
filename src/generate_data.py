import pandas as pd
import regex as re
import csv
from itertools import islice
import pickle
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from pathlib import Path

# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
#from pyserini.search import SimpleSearcher
#from pyserini.search import pysearch

import sys
sys.path.append("src")
from utils import *
from main_bm25 import Search_Top_k_Docs

path = str(Path.cwd())

# Lucene indexer
fiqa_index = path + "/retriever/lucene-index-fiqa/"


def split_question(train_label, test_label, valid_label, queries):
    """
    Split questions into train, test, validation sets.

    Returns:
        train_questions: Dataframe with qids
        test_questions: Dataframe with qids
        valid_questions: Dataframe with qids
    ----------
    Arguments:
        train_label: Dictionary contraining qid and list of relevant docid
        test_label: Dictionary contraining qid and list of relevant docid
        valid_label: Dictionary contraining qid and list of relevant docid
        queries: Dataframe containing the question id and question text
    """
    # Get a list of question ids
    train_q = list(train_label.keys())
    test_q = list(test_label.keys())
    valid_q = list(valid_label.keys())

    # Split question dataframe into train, test, valid set

    # Here below, train_questions will be a new DataFrame object that contains only the rows from queries where the 'qid' column matches any of the values in the train_q list.
    train_questions = queries[queries['qid'].isin(train_q)]
    test_questions = queries[queries['qid'].isin(test_q)]
    valid_questions = queries[queries['qid'].isin(valid_q)]

    return train_questions, test_questions, valid_questions

def split_label(qid_docid):
    """
    Split question answer pairs into train, test, validation sets.

    Returns:
        train_label: Dictonary
            key - question id
            value - list of relevant docids
        test_label: Dictonary
            key - question id
            value - list of relevant docids
        valid_label: Dictonary
            key - question id
            value - list of relevant docids
    ----------
    Arguments:
        qid_docid: Dataframe containing the question id and relevant docids
    """
    # Group the answers for each question into a list
    qid_docid = qid_docid.groupby(['qid']).agg(lambda x: tuple(x)).applymap(list).reset_index()
    # Split data
    train, test_set = train_test_split(qid_docid, test_size=0.05)
    train_set, valid_set = train_test_split(train, test_size=0.1)
    # Expand the list of docids into individual rows to represent a single sample
    train_data = train_set.explode('docid')
    test_data = test_set.explode('docid')
    valid_data = valid_set.explode('docid')

    # Convert data into dictionary - key: qid, value: list of relevant docid
    train_label = label_to_dict(train_data)
    test_label = label_to_dict(test_data)
    valid_label = label_to_dict(valid_data)

    return train_label, test_label, valid_label


def create_dataset(question_df, labels, cands_size, doc_index_id):
    """This function creates a dataset where each row represents 3 things:
    1) First column stores qid
    2) Second column stores list of relevant (actual) answers to this qid
    3) Third column stores list of 'k' candidate answers to this qid. Candidate answers were found by taking top 'k' answers whose embeddings are matching the most with qid. The similarity test was done and k answers were fetched using SimpleSearcher() method of pyserini."""
    
    """Retrieves the top-k candidate answers for a question and
    creates a list of lists of the dataset containing the question id,
    list of relevant answer ids, and the list of answer candidates

    Returns:
        dataset: list of list in the form [qid, [pos ans], [ans candidates]]
    ----------
    Arguments:
        question_df: Dataframe containing the qid and question text
        labels: Dictonary containing the qid to text map
        cands_size: int - number of candidates to retrieve
    """
    dataset = []
    # Calls retriever
    #searcher = SimpleSearcher(fiqa_index)
    # For each question
    query_id = 1
    for i, row in question_df.iterrows():
        qid = row['qid']
        tmp = []
        # Append qid
        tmp.append(qid)
        # Append list of relevant docs
        tmp.append(labels[qid])
        # Retrieves answer candidates
        cands = []
        query = row['question']
        query = re.sub('[£€§]', '', query)

        '''This command appears to use regular expressions (regex) to replace any occurrence of the characters '£', '€', or '§' in the query string with an empty string.
        
        In Python, the re.sub() function is used for regex substitution, where the first argument is the regex pattern to match and the second argument is the replacement string. In this case, the regex pattern [£€§] matches any of the three characters within the square brackets, and the replacement string is an empty string '', effectively removing any occurrences of these characters from the query string.
        
        For example, if query initially contained the string "I have £10 in my pocket", the command would replace the '£' character with an empty string and return the updated string "I have 10 in my pocket".
        
        Overall, this command may be used to pre-process text data by removing unwanted characters before performing further analysis or modeling.'''

        #hits = searcher.search(query, k=cands_size)         # k=50 by default. hits will retrieve and store top 50 relevant answers according to BM-25 algo
        hits = Search_Top_k_Docs(doc_index_id, query_id, query, cands_size)
        if(hits == -1):         # => error
            sys.exit()

        query_id += 1
        """The query parameter is the input query for the search, and k is the number of top-ranked documents to retrieve. The method returns a set of hits that represent the top-ranked documents retrieved by the search, where each hit contains information about the document such as its score, ID, and content."""

        for docid in hits:
            cands.append(int(docid))            # appending these top k fetched relevant answer ids (as they are candidate answers to our qid)
        # Append candidate answers
        tmp.append(cands)
        dataset.append(tmp)

    return dataset


def get_dataset(query_path, labels_path, cands_size, doc_index_id):
    """Splits the dataset into train, validation, and test set and creates
    the dataset form for training, validation, and testing.

    Returns:
        train_set: list of list in the form [qid, [pos ans], [ans candidates]]
        valid_set: list of list in the form [qid, [pos ans], [ans candidates]]
        test_set: list of list in the form [qid, [pos ans], [ans candidates]]
    ----------
    Arguments:
        query_path: str - path containing a list of qid and questions
        labels_path: str - path containing a list of qid and relevant docid
        cands_size: int - number of candidates to retrieve
    """
    # Question id and Question text
    queries = load_questions_to_df(query_path)                      # Returns a dataframe of question ids and question text.
    # Question id and Answer id pair
    qid_docid = load_qid_docid_to_df(labels_path)                   # Returns a dataframe of question id and relevant docid.

    # qid to docid label map
    labels = label_to_dict(qid_docid)                               # same as qid_docid dataframe but this function will convert the qid_docid dataframe into dictionary where key represents question id and value represents list of relevant docids.
    train_label, test_label, valid_label = split_label(qid_docid)   # Split 'labels' dictionary into train, test, validation sets.
    
    # Split Questions
    train_questions, test_questions, valid_questions = split_question(train_label, test_label, valid_label, queries)        # extract 'qid' and the corresponding query from train_label, test_label and vald_label respectively and stores these pairs into corresponding train_questions, test_questions, valid_questions dataframes.

    print("Generating training set...")
    train_set = create_dataset(train_questions, labels, cands_size, doc_index_id)
    print("Generating validation set...")
    valid_set = create_dataset(valid_questions, labels, cands_size, doc_index_id)
    print("Generating test set...")
    test_set = create_dataset(test_questions, labels, cands_size, doc_index_id)

    return train_set, valid_set, test_set


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--query_path", default=None, type=str, required=True,
    help="Path to the question id to text data in .tsv format. Each line should have at least two columns named (qid, question) separated by tab")
    
    parser.add_argument("--label_path", default=None, type=str, required=True,
    help="Path to the question id and answer id data in .tsv format. Each line should have at two columns named (qid, docid) separated by tab")
    
    parser.add_argument("--doc_index_id", default=None, type=str, required=True,
    help="The Lucene index id of the given docs.tsv file")

    # Optional parameters
    parser.add_argument("--cands_size", default=50, type=int, required=False,
    help="Number of candidates to retrieve per question.")
    
    parser.add_argument("--output_dir", default=Path.cwd()/'data/data_pickle/', type=str, required=False, 
    help="The output directory where the generated data will be stored.")

    args = parser.parse_args()

    if len(sys.argv) < 4:           # => insufficient arguments entered, display to the user the correct syntax for arguments
        print("Usage: python3 src/generate_data.py <query_path> <label_path>")
        sys.exit()

    train_set, valid_set, test_set = get_dataset(args.query_path, args.label_path, args.cands_size, args.doc_index_id)
    # Splits the dataset into train, validation, and test set and creates the dataset form for training, validation, and testing.

    """train_set, valid_set and test_set each is a dataframe containing the following 3 things:
    1) First column stores qid
    2) Second column stores list of relevant (actual) answers to this qid
    3_ Third column stores list of 'k' candidate answers to this qid. Candidate answers were found by taking top 'k' answers whose embeddings are matching the most with qid. The similarity test was done and k answers were fetched using SimpleSearcher() method of pyserini."""


    # Saving train_set, test_set and valid_set into "FinBERT-QA/data/data_pickle/" directory
    save_pickle(os.path.join(args.output_dir, "train_set.pickle"), train_set)
    save_pickle(os.path.join(args.output_dir, "valid_set.pickle"), valid_set)
    save_pickle(os.path.join(args.output_dir, "test_set.pickle"), test_set)
                             

    #print("Done. The pickle files are saved in {}".format(args.output_dir))

if __name__ == "__main__":
    main()