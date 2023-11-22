from statistics import mean 
from utils import *
from process_data import *

def main():
    # Question id and Question text
    queries = load_questions_to_df("data/raw/FiQA_train_question_final.tsv")

    # Document id and Answer text
    collection = load_answers_to_df("data/raw/FiQA_train_doc_final.tsv")

    # Question id and Answer id pair
    qid_docid = load_qid_docid_to_df("data/raw/FiQA_train_question_doc_final.tsv")

    # Cleaning data: empty_docs contains doc_ids whose answers are empty (null) and empty_id contains the index in collection dataframe of all such docids present in empty_docs in 
    empty_docs, empty_id = get_empty_docs(collection)
    
    # Remove empty answers from collection of answers
    collection_cleaned = collection.drop(empty_id)

    # Remove empty answers from qa pairs
    qid_docid_cleaned = qid_docid[~qid_docid['docid'].isin(empty_docs)]

    # Create a hashmap for where key represents qid and value represents list of all actual answers for that qid
    qid_relevant_ans_mapping = label_to_dict(qid_docid_cleaned)

    qid_to_text, docid_to_text = id_to_text(collection, queries)
    # converts collection and queries dataframe into dictionaries so that we can save these 2 in pickle format

    # Saving queries dataframe and collection dataframe as pickle files
    save_pickle("data/id_to_text/qid_to_text.pickle", qid_to_text)
    save_pickle("data/id_to_text/docid_to_text.pickle", docid_to_text)

    # Storing qid_relevant_ans_mapping as labels.pickle for evaluating our model after it has been fine tuned
    save_pickle("data/evaluation_labels/labels.pickle", qid_relevant_ans_mapping)

    # Write collection_cleaned df to Lucene_data folder for producing its indexing
    save_tsv("data/Lucene_data/docs.tsv", collection_cleaned)