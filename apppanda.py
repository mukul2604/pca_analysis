from flask import Flask
from flask import render_template

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


@app.route("/")
def index():
    preprocessdata("data/Letter_recognition.csv", False)
    return render_template("index.html")

@app.route("/lr/details")
def letter_recognition_details():
    return "details"


def preprocessdata(datapath, doplot):
    df = pd.read_csv(datapath)
    encoder = preprocessing.LabelEncoder()
    random_sample = df.take(np.random.permutation(len(df))[:1000])
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
    preprocessdata("data/Letter_recognition.csv", True)
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == "__main__":
    main()