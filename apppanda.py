from flask import Flask
from flask import render_template

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


@app.route("/")
def index():
    decimated_data("data/Letter_recognition.csv", False)
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
        # print len(x[mask])
    # print strat_samples[0], len(strat_samples)
    return strat_samples


"""Pre-process the data"""
def decimated_data(datapath, doplot):
    df = pd.read_csv(datapath)
    encoder = preprocessing.LabelEncoder()
    """
        Initial Random Sampling: takes 20% sample of total
        input sample
    """
    pctg = 0.2
    sample_len = int(len(df) * pctg)
    random_sample = df.take(np.random.permutation(len(df))[:sample_len])
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
        plt.plot(ks, score)
        plt.ylabel('Objective function value')
        plt.xlabel('Number of clusters')
        plt.show()

    """from plot elbow is found at k = 4"""
    k_elbow = 4

    stratified_samples = stratified_sampling(kmeans[k_elbow-1], x)
    return stratified_samples


def main():
    decimated_data("data/Letter_recognition.csv", False)
    # app.run(host='127.0.0.1', port=5000, debug=True)

if __name__ == "__main__":
    main()