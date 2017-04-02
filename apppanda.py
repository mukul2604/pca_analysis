from flask import Flask
from flask import render_template

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from bson import  json_util

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as SK_Metrics

from sklearn.manifold import MDS

app = Flask(__name__)

list_mds = ["euclidean", "correlation"]

@app.route("/")
def index():
    decimate_data("data/Letter_recognition.csv", False)
    return render_template("index.html")

@app.route("/lr/details")
def letter_recognition_details():
    return "details"


def find_mds(dataframe, type):
    dis_mat = SK_Metrics.pairwise_distances(dataframe, metric=type)
    mds = MDS(n_components=2, dissimilarity='precomputed')
    return pd.DataFrame(mds.fit_transform(dis_mat))


def stratified_sampling(kmean_obj, samples):
    cluster_numbers = np.array(kmean_obj.labels_).astype(int)
    clustered_data = []

    for i in range(kmean_obj.n_clusters):
        clustered_data.append([])

    # segregate the input random sample into their respective
    # clusters
    for i in cluster_numbers:
        clustered_data[i].append(samples[0])
        samples = np.delete(samples, 0, axis=0)

    """Random sampling on individual cluster set.
       Taking 20% random samples
    """
    strat_samples = []
    for i in range(kmean_obj.n_clusters):
        strat_samples.append([])

    for x in clustered_data:
        idx = clustered_data.index(x)
        x = np.array(x)
        mask = np.random.choice([False, True], len(x), p=[0.80, 0.20])
        strat_samples[idx].append(x[mask])
        print len(x[mask])

    # flatten the array
    flattened_arr = []
    samples = np.array(strat_samples).flatten()
    for s in samples:
        for item in s:
            flattened_arr.append(item)

    strat_samples = np.array(flattened_arr)
    return strat_samples


def decimate_data(datapath, doplot):
    df = pd.read_csv(datapath)
    df.columns = ["V" + str(i) for i in
                    range(1, len(df.columns) + 1)]
    df.V1 = df.V1.astype(str)
    X = df.loc[:, "V2":]  # independent variables data
    y = df.V1  # dependent variable data

    encoder = preprocessing.LabelEncoder()
    """
        Initial Random Sampling: takes 20% sample of total
        input sample
    """
    pctg = 0.20
    sample_len = int(len(df) * pctg)
    random_sample = X.take(np.random.permutation(len(df))[:sample_len])
    random_sample_encoded = random_sample.apply(encoder.fit_transform)

    """
        K-means clustering
    """
    x = np.array(random_sample_encoded)
    x = x.astype(int)
    # print len(x[0])
    ks = range(1, 16)
    kmeans = [KMeans(n_clusters=i, random_state=0) for i in ks]
    score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]
    score = [-score[i] for i in range(len(ks))]

    """Plot for evaluating the Elbow for k-Means clustering"""
    if doplot:
        colors = np.random.rand(100)
        plt.suptitle("Elbow Plot", fontsize=14, fontweight='bold')
        plt.scatter(ks, score, c=colors, alpha=0.5)
        plt.plot(ks, score)
        plt.ylabel('Objective Function Value')
        plt.xlabel('Number of clusters')
        plt.show()

    """
        from elbow plot, elbow is found at k = 4
        next do the stratified sampling on those 4 clusters.
    """
    k_elbow = 4

    decimated_data = stratified_sampling(kmeans[k_elbow-1], x)

    return StandardScaler().fit_transform(decimated_data.astype(float))


def squared_sum(arr):
    sqr_sum = 0.0
    for x in arr:
        sqr_sum += pow(x, 2)
    return sqr_sum


def highest_attributes(loading_attr):
    loading_attr = [(i, loading_attr[i]) for i in xrange(len(loading_attr))]
    loading_attr.sort(key=lambda x: x[1], reverse=True)
    return [loading_attr[0][0]] + [loading_attr[1][0]] + [loading_attr[2][0]]


def dump_data_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv("data/" + filename, index=False)


def dimension_reduction(datapath, draw_plots):
    decimated_data = decimate_data(datapath, draw_plots)
    pca = PCA()
    pca_trans = pca.fit_transform(decimated_data)
    components = range(pca.n_components_)
    components = [x+1 for x in components]

    """Scree Plot"""
    if draw_plots:
        colors = np.random.rand(100)
        plt.suptitle("Scree Plot", fontsize=14, fontweight='bold')
        plt.scatter(components, pca.explained_variance_, c=colors, alpha=0.5)
        plt.plot(components, pca.explained_variance_)
        line_data = np.array([1] * len(components))
        plt.plot(components, line_data, 'r--')
        plt.xticks([x for x in components])
        plt.ylabel('Eigen Values')
        plt.xlabel('PCA Components')
        plt.show()

    # if draw_plots:
    #     plt.plot(pca_trans[0:40, 0], pca_trans[0:40, 1], 'o', markersize=7, color='blue', alpha=0.5,
    #              label='class1')
    #     plt.plot(pca_trans[40:140, 0], pca_trans[40:140, 1], '^', markersize=7, color='red', alpha=0.5,
    #              label='class2')
    #     plt.xlabel('x_values')
    #     plt.ylabel('y_values')
    #     plt.xlim([-4, 4])
    #     plt.ylim([-4, 4])
    #     plt.legend()
    #     plt.title('PCA 2D scatter')
    #     plt.show()

    principal_components = 0
    for x in pca.explained_variance_:
        if x > 1:
            principal_components += 1

    component_matrix = pca.components_[:, :principal_components]
    eigenvalues = pca.explained_variance_[:principal_components]
    loading_matrix = component_matrix * [math.sqrt(x) for x in eigenvalues]
    loading_arr = [squared_sum(x) for x in loading_matrix]

    print eigenvalues
    print highest_attributes(loading_arr)

    dim_reduced_data = pca_trans[:, 0:principal_components]
    dump_data_to_csv(dim_reduced_data, 'dimension_reduced_data.csv')
    for mds_type in list_mds:
        dump_data_to_csv(find_mds(dim_reduced_data, mds_type), mds_type + '.csv')
    print("Contents Dumped")


def main():
    dimension_reduction("data/Letter_recognition.csv", False)
    #app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == "__main__":
    main()