#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import ast
import sys
import os
import getopt

from clayrs import content_analyzer as ca
from clayrs import recsys as rs
from clayrs import evaluation as eva

from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import kl_div
from gensim.matutils import hellinger

import warnings

warnings.filterwarnings("ignore")

# Multi-processing
### Initial values
dataset, algorithm, alpha_value = None, None, None

try:
    opts, args = getopt.getopt(
        sys.argv[1:], "d:a:v:",  # options that require argument are followed by a colon
        ["dataset=", "algorithm=", "alpha_value="])

except getopt.GetoptError:
    print("Invalid command-line arguments.")
    print("Usage: ADF_script.py [-w] [-d <dataset>] [-a <algorithm>] [-v <alpha-value>]")
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-d", "--dataset"):
        dataset = str(arg)
        # intensity =  ast.literal_eval(arg)

    elif opt in ("-a", "--algorithm"):
        algorithm = str(arg)

    elif opt in ("-v", "--alpha-value"):
        alpha_value = int(arg) / 10  # Division by 10 to have a value between 0 and 1

print(f'\ADF parameters:')
print(f"\tDataset: {dataset}")
print(f"\tAlgorithm: {algorithm}")
print(f"\tAlpha value: {alpha_value}")


# Transform embeddings (numerical news representation to dataframe)
def embeddings_to_df(e):
    e_list = e.tolist()
    df_e = pd.DataFrame(e_list)
    return df_e


# Performances that can be computed using the ClayRS library
def get_performances(train, test, k):
    em = eva.EvalModel(
        [train],
        [test],
        metric_list=[
            eva.PrecisionAtK(k, sys_average='macro'),
            eva.RecallAtK(k),
            eva.FMeasureAtK(k),
            eva.NDCG(),
        ],
    )
    sys_result, users_result = em.fit()
    return sys_result, users_result


def triangular_matrix(m):
    m_tri = m.where(np.triu(np.ones(m.shape), k=1).astype(bool))
    return m_tri


def ILS(m):
    m_tri = triangular_matrix(m).stack().reset_index()
    m_tri.columns = ['i', 'j', 'similarity']
    ils = (m_tri['similarity'].sum()) / len(m_tri)
    return ils


def compute_ils(recos, similarity_matrix, list_users):
    dict_ils = {}
    for u in tqdm(list_users):
        recos_user = recos[recos['user_id'] == u]
        news_ids = recos_user['news_id'].tolist()
        news_ids.sort()
        sim_matrix_user = similarity_matrix[similarity_matrix.index.isin(news_ids)][news_ids]
        ils_user = ILS(sim_matrix_user)
        dict_key = {u: ils_user}
        dict_ils.update(dict_key)
    return dict_ils


def get_results_categories(initial_results, news):
    results_categories = initial_results.copy()
    results_categories = results_categories.merge(news[['news_id', 'category_name']], on='news_id').rename(
        columns={'category_name': 'category'})
    return results_categories


def s_recall(recos, list_users, categories_list, k):
    dict_srecall = {}
    nb_categories = len(categories_list)
    for u in tqdm(list_users):
        recos_user = recos[recos['user_id'] == u].reset_index(drop=True)
        recos_categories = recos_user['category'].unique().tolist()
        s_recall_user = len(set(recos_categories)) / nb_categories
        dict_key = {u: s_recall_user}
        dict_srecall.update(dict_key)
    return dict_srecall


def homogeneization(distrib, param=0.5):
    n = len(distrib)
    new_distrib = [((1 - param) * p) + (param / n) for p in distrib]
    return new_distrib


def calibration_hellinger(recos, users_list, users_interest_df, categories_list):
    # recos = get_results_categories(recos)
    dict_ch_smooth = {}
    for u in tqdm(users_list):
        best_ch = 10
        recos_user = recos[recos['user_id'] == u].reset_index(drop=True)
        interest_user = users_interest_df.loc[u].values.tolist()
        distrib_cat_recos = []
        for c in categories_list:
            prop_cat = len(recos_user[recos_user['category'] == c])
            distrib_cat_recos.append(prop_cat / len(recos_user))
        for l in np.arange(0, 1.1, 0.1):
            new_distrib = homogeneization(interest_user, param=round(l, 1))
            c_h = hellinger(distrib_cat_recos, new_distrib)
            if c_h < best_ch:
                best_ch = c_h
                optimal_lambda = l
        dict_key = {u: best_ch}
        dict_ch_smooth.update(dict_key)
    return dict_ch_smooth


def get_all_results(results, test_set, users_list, news, interest, similarity_matrix, categories_list, name_parameters,
                    k=20):
    print('Pre-processing...')
    results_categories = get_results_categories(results, news)
    results_ratings = ca.Ratings.from_dataframe(results)
    test_ratings = ca.Ratings.from_dataframe(test_set)
    print('OK!')
    # Accuracy
    print('Precision...')
    sys_results, users_results = get_performances(results_ratings, test_ratings, k=k)
    print('OK!')
    # Instantiate the dataframe with global results
    eval_results_global = sys_results.reset_index().copy()
    eval_results_global = eval_results_global[eval_results_global['user_id'] == 'sys - fold1']
    eval_results_global['user_id'] = [name_parameters]
    eval_results_global = eval_results_global.rename(columns={'user_id': 'value'})
    eval_results_global = eval_results_global.set_index('value')
    eval_results_global = eval_results_global.rename(
        columns={'Precision@20 - macro': 'Precision', 'Recall@20 - macro': 'Recall', 'F1@20 - macro': 'F1'})

    # Instantiate the dataframe with individual results
    eval_results_indiv = users_results.copy()
    # eval_results_indiv.index = eval_results_indiv.index.astype(int)
    eval_results_indiv = eval_results_indiv.rename(
        columns={'Precision@20 - macro': 'Precision', 'Recall@20 - macro': 'Recall', 'F1@20 - macro': 'F1'})

    print('ILD...')
    # ILS
    dict_ils = compute_ils(results, similarity_matrix, users_list)
    # transformation to have intra-list diversity, not intra-list similarity
    dict_ild = {}
    for key in dict_ils.keys():
        dict_ild[key] = 1 - dict_ils[key]
    eval_results_global['ILD'] = np.mean(list(dict_ild.values()))
    eval_results_indiv['ILD'] = eval_results_indiv.index.map(dict_ild)
    print('OK!')

    print('S-Recall...')
    # S-Recall
    dict_srecall = s_recall(results_categories, users_list, categories_list, k=k)
    eval_results_global['s_recall'] = np.mean(list(dict_srecall.values()))
    eval_results_indiv['s_recall'] = eval_results_indiv.index.map(dict_srecall)
    print('OK!')

    print('Calibration Hellinger')
    # C_KL
    dict_ch = calibration_hellinger(results_categories, users_list, interest, categories_list)
    eval_results_global['c_hell'] = np.mean(list(dict_ch.values()))
    eval_results_indiv['c_hell'] = eval_results_indiv.index.map(dict_ch)
    print('OK!')

    eval_results_global = eval_results_global.round(3)

    eval_results_indiv = eval_results_indiv.round(3)

    return eval_results_global, eval_results_indiv


def get_results_df(path, news):
    results = pd.read_csv(path, index_col=0)
    results = results[['user_id', 'news_id', 'score']]
    results = results[results['news_id'].isin(news['news_id'].unique().tolist())]
    return results


print('Data import...')

if dataset == 'MIND':
    news = pd.read_pickle('data/{}/news_info.pkl'.format(dataset))
    news = news[['NewsID_small', 'Category', 'Title', 'Embedding']].rename(
        columns={'NewsID_small': 'news_id', 'Category': 'category_name'})
    news = news[~news['news_id'].isna()]
    news = news[~news['Embedding'].isna()]
    # Define list of categories
    categories_list = ['lifestyle', 'health', 'news', 'sports', 'weather', 'entertainment', 'foodanddrink', 'autos',
                       'travel', 'video', 'tv', 'finance', 'movies', 'music', 'kids']
    # Only keep news from these categories
    news = news[news['category_name'].isin(categories_list)]
    # Create a dictionary that maps each unique category_name to a unique integer
    category_mapping = {category: i for i, category in enumerate(news['category_name'].unique())}
    # Apply the mapping to the 'category_name' column
    news['category'] = news['category_name'].map(category_mapping)

elif dataset == 'ADRESSA':
    news = pd.read_csv('data/{}/news_adressa_emb.csv'.format(dataset), index_col=0)
    news = news[['nid', 'category', 'title', 'embeddings']].rename(
        columns={'nid': 'news_id', 'category': 'category_name', 'title': 'Title', 'embeddings': 'Embedding'})
    news = news[~news['news_id'].isna()]
    news = news[~news['Embedding'].isna()]
    # Rename the "100sport" category in "sport" to have one unique category corresponding to sport
    news['category_name'] = news['category_name'].replace('100sport', 'sport')
    # Define list of categories
    categories_list = ['nyheter', 'sport', 'forbruker', 'kultur', 'meninger', 'bolig', 'tema', 'tjenester', 'bil',
                       'migration catalog']
    # Only keep news from these categories
    news = news[news['category_name'].isin(categories_list)]
    # Create a dictionary that maps each unique category_name to a unique integer
    category_mapping = {category: i for i, category in enumerate(news['category_name'].unique())}
    # Apply the mapping to the 'category_name' column
    news['category'] = news['category_name'].map(category_mapping)
    categories_list = [*category_mapping]

news['Embedding'] = news['Embedding'].apply(ast.literal_eval)

news_embeddings_lda = embeddings_to_df(news['Embedding'])
news_embeddings_lda.index = news['news_id']

users_interest = pd.read_csv('data/' + dataset + '/categories_distribution.csv', index_col=0)
users_interest.columns = users_interest.columns.astype(int)

list_users = pd.read_csv('list_users_{}.csv'.format(dataset))['0'].tolist()
print('Nb of users:', len(list_users))

test_set = pd.read_csv('data/' + dataset + '/test_set.csv', index_col=0)
test_set = test_set.rename(columns={'UserID': 'user_id', 'NewsID': 'news_id', 'Score': 'score'})

datapath_recos = 'reco_scores_greedy/{}/{}/{}_{}_RESULTS_lambda_{}_ADF.csv'.format(dataset, algorithm, algorithm,
                                                                                   dataset,
                                                                                   str(alpha_value).replace('.', ''))
results = get_results_df(datapath_recos, news)

behaviors = pd.read_csv('data/{}/behaviors.csv'.format(dataset), index_col=0)
behaviors = behaviors.rename(columns={'UserID': 'user_id', 'NewsID': 'news_id', 'Score': 'score'})

# print('liste initiale :', len(users_list))
# results = results[results['user_id'].isin(users_list)]
# results = results.reset_index(drop=True)
# users_list = results['user_id'].unique().tolist()
# print('liste re-filtrée :', len(users_list))

print('Data correctly imported!')
print('Number of users: ', len(list_users))

print('Construct similarity matrix')
array_sim = cosine_similarity(news_embeddings_lda)
similarity_matrix = pd.DataFrame(array_sim)
similarity_matrix.index = news_embeddings_lda.index.tolist()
similarity_matrix.columns = news_embeddings_lda.index.tolist()
print('Similarity matrix : OK !')

print('Getting evaluation results...')
evaluation_results = get_all_results(results, test_set, list_users, news, users_interest, similarity_matrix,
                                     categories_list, ''.format(algorithm), k=20)
evaluation_results[0]['NDCG'] = evaluation_results[1]['NDCG'].fillna(0).mean().round(3)
print('Evaluation done !')

print('Saving evaluation results...')
datapath_results_global = 'evaluation_results/{}/greedy/{}/greedy_results_{}_lambda_{}.csv'.format(dataset, algorithm,
                                                                                                   algorithm,
                                                                                                   str(alpha_value).replace(
                                                                                                       '.', ''))
evaluation_results[0].to_csv(datapath_results_global)
datapath_results_indiv = 'evaluation_results/{}/greedy/{}/greedy_results_indiv_{}_lambda_{}.csv'.format(dataset,
                                                                                                        algorithm,
                                                                                                        algorithm,
                                                                                                        str(alpha_value).replace(
                                                                                                            '.', ''))
evaluation_results[1].to_csv(datapath_results_indiv)
print('Results saved!')

print(
    'Evaluation of results when greedy re-ranking is applied on {} dataset with {} algorithm, with lambda = {} done !'.format(
        dataset, algorithm, alpha_value))
