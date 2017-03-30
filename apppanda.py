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
    preprocess_data("data/Letter_recognition.csv", False)
    return render_template("index.html")

@app.route("/lr/details")
def letter_recognition_details():
    return "details"

def stratified_sampling():
    return 0


"""Pre-process the data"""
def preprocess_data(datapath, doplot):
    df = pd.read_csv(datapath)
    encoder = preprocessing.LabelEncoder()
    """
        Random Sampling: takes 20% sample of total
        input sample
    """
    pctg = 0.2
    sample_len = int(len(df) * pctg)
    random_sample = df.take(np.random.permutation(len(df))[:sample_len])
    random_sample_encoded = random_sample.apply(encoder.fit_transform)

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


def main():
    preprocess_data("data/Letter_recognition.csv", True)
    app.run(host='127.0.0.1', port=5000, debug=True)

if __name__ == "__main__":
    main()