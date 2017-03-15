from flask import Flask
from flask import render_template
from pymongo import MongoClient
import json

app = Flask(__name__)

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DBS_NAME = 'traffic_violation'
COLLECTION_NAME = 'details'
FIELDS = {'Make': True, 'Model': True}


@app.route("/")
def index():
    return render_template("index.html")
    # return 'Hello'

@app.route("/traffic_violation/details")
def traffic_violation_details():
    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = connection[DBS_NAME][COLLECTION_NAME]
    recs = collection.find(projection=FIELDS, limit=100000)
    # recs = collection.find(projection=FIELDS)
    json_recs = []
    for rec in recs:
        json_recs.append(rec)
        print rec
        print
        json_recs = json.dumps(json_recs, default=json.util.default)
    connection.close()
    return json_recs


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
