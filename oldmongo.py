from flask import Flask
from flask import render_template
from pymongo import MongoClient
import json
from bson import json_util
from bson.json_util import dumps
import numpy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DBS_NAME = 'tv'
COLLECTION_NAME = 'lrecords'
FIELDS = {'Make': True, 'Model': True}



@app.route("/")
def index():
    return render_template("visualize.html")
    # return 'Hello'


@app.route("/lr/details")
def letter_recog_details():
    # return "Hdeded"
    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = connection[DBS_NAME][COLLECTION_NAME]
    recs = collection.find(projection=FIELDS, limit=100000)
    # recs = collection.find(projection=FIELDS)
    json_recs = []
    for rec in recs:
        json_recs.append(rec)
        # json_recs.append("\n")
    rnd_sample = numpy.random.choice(json_recs, 1000)

    # kmeans = KMeans(n_clusters=10, random_state=0)
    # # kmeans.fit_predict(rnd_sample)
    # # print type(cluster_sample)
    # # rnd_sample = json.dumps(rnd_sample, default=json_util.default)
    # # stratified_sample = stratified_samples(rnd_sample, 10, 1000)
    connection.close()
    return str(rnd_sample)
    # KMeans(8)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
