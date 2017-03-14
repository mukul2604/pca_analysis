from flask import Flask
from flask import render_template
from pymongo import MongoClient
import json
#from json import  json.util


app = Flask(__name__)

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DBS_NAME = 'traffic_violation'
COLLECTION_NAME = 'details'
FIELDS = {'Make': True, 'Model': True}

#
# //@app.route("/")
# //def index():
#     return render_template("index.html")


@app.route("/traffic_violation/details")
def traffic_violation_details():
    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = connection[DBS_NAME][COLLECTION_NAME]
    projects = collection.find(projection=FIELDS, limit=100000)
    #projects = collection.find(projection=FIELDS)
    json_projects = []
    for project in projects:
        json_projects.append(project)
 #   json_projects = json.dumps(json_projects, default=json_util.default)
    connection.close()
    return json_projects

if __name__ == "__main__":
    app.run(host='localhost',port=5000,debug=True)
