from flask import Flask
from flask import render_template
from pymongo import MongoClient
import json
from bson import json_util
from bson.json_util import dumps

app = Flask(__name__)

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DBS_NAME = 'tv'
COLLECTION_NAME = 'tvrecords'
FIELDS = {'Make': True, 'Model': True}



@app.route("/")
def index():
    return render_template("index.html")
    # return 'Hello'


@app.route("/tv/details")
def traffic_violation_details():
    # return "Hdeded"
    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = connection[DBS_NAME][COLLECTION_NAME]
    recs = collection.find(projection=FIELDS, limit=100000)
    # recs = collection.find(projection=FIELDS)
    json_recs = []
    for rec in recs:
        json_recs.append(rec)
        json_recs.append("\n")
    json_recs = json.dumps(json_recs, default=json_util.default)
    connection.close()
    return json_recs


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
