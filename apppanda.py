from flask import Flask
from flask import render_template

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


@app.route("/")
def index():
    decimate_data("data/Letter_recognition.csv", False)
    return render_template("index.html")

@app.route("/lr/details")
def letter_recognition_details():
    return "details"


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
                    range(1, len(df.columns) + 1)]  # rename column names to be similar to R naming convention
    df.V1 = df.V1.astype(str)
    X = df.loc[:, "V2":]  # independent variables data
    y = df.V1  # dependent variable data

    encoder = preprocessing.LabelEncoder()
    """
        Initial Random Sampling: takes 20% sample of total
        input sample
    """
    pctg = 0.2
    sample_len = int(len(df) * pctg)
    random_sample = X.take(np.random.permutation(len(df))[:sample_len])
    random_sample_encoded = random_sample.apply(encoder.fit_transform)

    """
        K-means clustering
    """
    x = np.array(random_sample_encoded)
    x = x.astype(int)

    ks = range(1, 16)
    kmeans = [KMeans(n_clusters=i, random_state=0) for i in ks]
    score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]
    score = [-score[i] for i in range(len(ks))]

    if doplot:
        plt.suptitle("Elbow Plot", fontsize=14, fontweight='bold')
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
    #  return standardized data
    return StandardScaler().fit_transform(decimated_data.astype(float))


def dimension_reduction(datapath, draw_plots):
    decimated_data = decimate_data(datapath, draw_plots)
    pca = PCA()
    pca.fit(decimated_data)
    components = range(pca.n_components_)
    components = [x+1 for x in components]

    if draw_plots:
        colors = np.random.rand(100)
        plt.suptitle("Scree Plot", fontsize=14, fontweight='bold')
        plt.scatter(components, pca.explained_variance_, c=colors, alpha=0.5)
        plt.plot(components, pca.explained_variance_)
        line_data = np.array([1 for i in xrange(len(components))])
        plt.plot(components, line_data, 'r--')
        plt.xticks([x for x in components])
        plt.ylabel('Eigen Values')
        plt.xlabel('PCA Components')
        plt.show()
    print pca.explained_variance_


def main():
    dimension_reduction("data/Letter_recognition.csv", True)
    # decimate_data("data/Letter_recognition.csv", False)
    # app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == "__main__":
    main()