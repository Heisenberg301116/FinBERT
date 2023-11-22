import pandas as pd
from statistics import mean
import math
import numpy as np
from itertools import islice

def get_rel(labels, cands):
    """Get relevant positions of the hits.

    Returns: List of 0's and 1's incidating a relevant answer
    -------------------
    Arguments:
        labels: List of relevant docids
        cands: List of candidate docids
    """
    rel = []
    for cand in cands:
        if cand in labels:
            rel.append(1)
        else:
            rel.append(0)

    return rel

def dcg(rels, k):
    """
    Discounted Cumulative Gain. Computes the cumulated DCG of the top-k
    relevant docs across all queries.

    Returns:
        cumulated_sum: float - cumulated DCG
    ----------
    Arguments:
        rels: list
            List of relevant scores of 0 or 1 e.g. [0, 1, 0, 1]
        k: int
            Top-k relevant docs
    """
    cumulated_sum = rels[0]
    for i in range(1, len(rels)):
        cumulated_sum += rels[i]/math.log(i+1,2)
    return cumulated_sum

def avg_ndcg(rel_score, k):
    """
    Average Normalized Discounted Cumulative Gain. Computes the DCG, iDCG, and
    nDCG for each query and returns the averyage nDCG across all queries.

    Returns:
        avg: float - average nDCG
    ----------
    Arguments:
        rel_score: dictionary
            key - question id
            value - list of relevancy scores with 1 (relevant) and 0 (irrelevant)
            e.g. {0: [0, 1, 0], 1: [1, 1, 0]}
        k: int
            Top-k relevant docs
    """
    ndcg_list = []
    for qid, rels in rel_score.items():
        # Compute DCG for each question
        dcg_val = dcg(rels, k)
        sorted_rel = sorted(rels, reverse=True)
        # Compute iDCG for each question
        idcg_val = dcg(sorted_rel, k)

        try:
            ndcg_val = dcg_val/idcg_val
            ndcg_list.append(ndcg_val)
        except ZeroDivisionError:
            ndcg_list.append(0)

    assert len(ndcg_list) == len(rel_score), "Relevant score doesn't match"

    # Get the average nDCG across all queries
    avg = mean(ndcg_list)

    return avg

def compute_RR(cand_docs, rel_docs, cumulated_reciprocal_rank, rank_pos, k):
    """
    This function will accept the list of 'k' candidate/predicted answers and the list of actual/relevant answers of a particular query and assign rank/score of this predicted 
    query based upon the below logic:
    
    Case 1: None of the candidate/predicted answer is present in the list of actual answer to the query: Score of 0 will be assigned to this query.
    Case 2: The first candidate answer from the top (0th index) that is also present in the list of actual answer is present at the ith index: Score of 1/(i+1) will be assigned 
    to this query and we won't check further for remaining bottom candidate answers. This score will be added to cumulative_reciprocal_rank. This index will also be appended to 
    rank_pos list. rank_pos will store the first occurence of relevant answer in the candidate answer and store the index of 1st occurence in itself. But if there is no candidate
    answer that is also present in relevant answer list, then don't append anything to rank_pos.

    Computes the reciprocal rank - probability of correctness of rank. Returns the cumulated reciprocal rank across all queries and the positions of the relevant docs in the 
    candidates.

    Returns:
        cumulated_reciprocal_rank: float - cumulated Reciprocal Rank across all queries
        rank_pos: list - index of the relevant docs in the candidates
    ----------
    Arguments:
        cand_docs: list
            List of ranked docids for a question
        rel_docs: list
            List of the relevancy of docids for a question
        cumulated_reciprocal_rank: int
            Initial value = 0
        rank_pos: list
            Initial list = []
        k: int
            Top-k relevant docs
    """

    for i in range(0, len(cand_docs)):
        # If the doc_id of the top k ranked candidate passages is in
        # the list of relevant passages
        if cand_docs[i] in rel_docs:
            # Compute the reciprocal rank (i is the ranking)
            rank_pos.append(i+1)
            cumulated_reciprocal_rank += 1/(i+1)
            break

    return cumulated_reciprocal_rank, rank_pos

def create_qid_pred_rank(test_set):
    """Creates dictionary of qid and list of candidates from test set.

    Returns:
        qid_pred_rank: dictionary
            key - qid
            value - list of candidates
    ----------
    Arguments:
        test_set: list
            [[qid, [positive docids], [list of candidates]]]
    """
    qid_pred_rank = {}

    for row in test_set:
        qid_pred_rank[row[0]] = row[2]

    return qid_pred_rank

def evaluate(qid_pred_rank, labels, k):
    """
    Evaluate. Computes the MRR@k, average nDCG@k, and average precision@k1

    Returns:
        MRR: float
        average_ndcg: float
        avg_precision: float
        r_pos: int
    ----------
    Arguments:
        qid_pred_rank: dictionary
            key - qid
            value - list of cand ans
        labels:  dinctionary
            key- qid
            value - list of relevant ans
    """
    cumulated_reciprocal_rank = 0
    num_rel_docs = 0
    # Dictionary of the top-k relevancy scores of docs in the candidate answers
    rel_scores = {}
    precision_list = {}
    rank_pos = []

    # For each query
    for qid in qid_pred_rank:
        # If the query has a relevant passage
        if qid in labels:
            # Get the list of relevant docs for a query
            rel_docs = labels[qid]
            # Get the list of ranked (predicted) docs for a query
            cand_docs = qid_pred_rank[qid]
            # Compute relevant scores of the candidates
            if qid not in rel_scores:
                rel_scores[qid] = []
                for i in range(0, len(cand_docs)):               # k != len(cand_docs), k = top k number of docs predicted by our fine tuned model as answer to the query from the 50 answers
                    if cand_docs[i] in rel_docs:            # if the predicted answer is also the actual relevant answer 
                        rel_scores[qid].append(1)           # give score of 1 to this candidate answer 
                    else:
                        rel_scores[qid].append(0)           # else give score of 0 to this answer
            # Compute the reciprocal rank and rank positions
            cumulated_reciprocal_rank, r_pos = compute_RR(cand_docs, rel_docs, cumulated_reciprocal_rank, rank_pos, k)

    # Compute the average MRR@k across all queries
    MRR = cumulated_reciprocal_rank/len(qid_pred_rank)
    
    # Compute the nDCG@k across all queries
    average_ndcg = avg_ndcg(rel_scores, k)

    # Compute precision@1
    precision_at_k = []
    for qid, score in rel_scores.items():
        num_rel = 0
        for i in range(0, 1):
            if score[i] == 1:
                num_rel += 1
        precision_at_k.append(num_rel/1)

    avg_precision = mean(precision_at_k)

    return MRR, average_ndcg, avg_precision, r_pos
