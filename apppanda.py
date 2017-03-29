from flask import Flask
from flask import render_template

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")
    # return 'Hello'


@app.route("/lr/details")
def traffic_violation_details():
    df = pd.read_csv("data/Letter_recognition.csv")
    encoder = preprocessing.LabelEncoder()
    random_sample = df.take(np.random.permutation(len(df))[:5000])
    random_sample_encoded =random_sample.apply(encoder.fit_transform)

    x = np.array(random_sample_encoded)
    x = x.astype(int)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(x)

    # return str(x)
    return str(kmeans.labels_)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
