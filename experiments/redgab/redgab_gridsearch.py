"""Code for hyperparameter-tuning of thresholding and LGDE in redgab application"""

import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lgde import Thresholding, LGDE, evaluate_prediction

###################################
# data loading and pre-processing #
###################################

# load fine-tuned embeddings for different dimensions
embedding_50d = pd.read_pickle("data/glove_redgab_15_0.8_50d_1.0mu_df.pkl")
embedding_100d = pd.read_pickle("data/glove_redgab_15_0.8_100d_1.0mu_df.pkl")
embedding_300d = pd.read_pickle("data/glove_redgab_15_0.8_300d_1.0mu_df.pkl")

# retrieve word vectors at different dimensions
word_vecs_50d = np.zeros(
    (embedding_50d["word_vector"].shape[0], embedding_50d["word_vector"][0].shape[0])
)
for i in range(word_vecs_50d.shape[0]):
    word_vecs_50d[i, :] = embedding_50d["word_vector"][i]

word_vecs_100d = np.zeros(
    (embedding_100d["word_vector"].shape[0], embedding_100d["word_vector"][0].shape[0])
)
for i in range(word_vecs_100d.shape[0]):
    word_vecs_100d[i, :] = embedding_100d["word_vector"][i]

word_vecs_300d = np.zeros(
    (embedding_300d["word_vector"].shape[0], embedding_300d["word_vector"][0].shape[0])
)
for i in range(word_vecs_300d.shape[0]):
    word_vecs_300d[i, :] = embedding_300d["word_vector"][i]

word_vecs_all_dimensions = [word_vecs_50d, word_vecs_100d, word_vecs_300d]

# get word list
word_list = list(embedding_50d["word_string"])

# load redgab data and split into train and test
with open("data/redgab_pos_data.pkl", "rb") as handle:
    domain_data = pickle.load(handle)

X_train, X_test, y_train, y_test = train_test_split(
    domain_data.data,
    domain_data.target,
    test_size=0.25,
    stratify=domain_data.target,
    random_state=42,
)

# compute document-term matrix for train
vectorizer = CountVectorizer(vocabulary=word_list)
word_counts_train = vectorizer.fit_transform(X_train)
dt_train = pd.DataFrame(word_counts_train.toarray(), columns=word_list)

# seed dictionary is defined as top 5 most frequent hate keywords according to Qian et al.
seed_dict = ["nigger", "faggot", "retard", "retarded", "cunt"]

dimension_string = ["50", "100", "300"]

###############
# grid search #
###############

for dim_ind, dim in enumerate(dimension_string):

    word_vecs = word_vecs_all_dimensions[dim_ind]

    ###############################################
    # grid search for thresholding hyperparameter #
    ###############################################

    epsilons = np.arange(0.3, 1, 0.001)
    size_disc_eps = np.zeros_like(epsilons)
    precision_eps = np.zeros_like(epsilons)
    recall_eps = np.zeros_like(epsilons)
    fscore_eps = np.zeros_like(epsilons)

    # create Thresholding object
    thres = Thresholding(seed_dict, word_list, word_vecs)

    for i, epsilon in tqdm(enumerate(epsilons), total=len(epsilons)):
        # compute epsilon balls around keywords
        thres.expand(epsilon=epsilon)
        size_disc_eps[i] = len(thres.discovered_dict_)
        # evaluate expanded dictionary on training set
        p, r, f = evaluate_prediction(
            thres.expanded_dict_, y_train, dt_train, with_report=False
        )
        precision_eps[i] = p
        recall_eps[i] = r
        fscore_eps[i] = f

    results_th = {
        "size": size_disc_eps,
        "precision": precision_eps,
        "recall": recall_eps,
        "fscore": fscore_eps,
        "eps": epsilons,
    }

    ########################################
    # grid search for LGDE hyperparameters #
    ########################################

    times = np.arange(1, 11, dtype=int)
    ks = np.arange(3, 21, dtype=int)
    size_disc = np.zeros((len(ks), len(times)))
    precision = np.zeros((len(ks), len(times)))
    recall = np.zeros((len(ks), len(times)))
    fscore = np.zeros((len(ks), len(times)))
    communities = []

    # create LGDE object
    lgde = LGDE(seed_dict, word_list, word_vecs)

    # perform grid search
    for i, k in tqdm(enumerate(ks), total=len(ks)):
        lgde.construct_network(k=k)

        for j, t in tqdm(enumerate(times), total=len(times)):
            lgde.detect_local_communities(t=t, disable_tqdm=True)
            # store communities and compute size of discovered dictionary
            communities.append(lgde.semantic_communities_)
            size_disc[i, j] = len(lgde.discovered_dict_)
            # evaluate expanded dictionary on training set
            p, r, f = evaluate_prediction(
                lgde.expanded_dict_, y_train, dt_train, with_report=False
            )
            precision[i, j] = p
            recall[i, j] = r
            fscore[i, j] = f

    # store results
    results_lgde = {
        "size": size_disc,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "communities": communities,
        "times": times,
        "ks": ks,
    }

    # store thresholding and gbda results in dictionary of dictionaries
    results = {"th": results_th, "lgde": results_lgde}

    with open(
        f"data/gs_th_lgde_glove_redgab_15_0.8_{dim}d_1.0mu_df.pkl", "wb"
    ) as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
