from flask import Flask
from flask import render_template
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as SK_Metrics
from sklearn.manifold import MDS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

app = Flask(__name__)

list_mds = ["euclidean", "correlation"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/visualize")
def letter_recognition_details():
    return render_template("visualize.html")


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
        sample = np.append(samples[0], i)
        clustered_data[i].append(sample)
        samples = np.delete(samples, 0, axis=0)

    """
       Random sampling on individual cluster set.
       Taking 20% random samples
    """
    strat_samples = []
    for i in range(kmean_obj.n_clusters):
        strat_samples.append([])

    print "Stratified sampling:"
    for x in clustered_data:
        idx = clustered_data.index(x)
        x = np.array(x)
        mask = np.random.choice([False, True], len(x), p=[0.80, 0.20])
        strat_samples[idx].append(x[mask])
        print "Samples in cluseter_id " + str(idx) + ": " + str(len(x[mask]))

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
    print "Initial number of samples: " + str(len(df))
    encoder = preprocessing.LabelEncoder()
    """
        Initial Random Sampling: takes 20% sample of total
        input sample
    """
    pctg = 0.20
    print "Random sampling rate: " + str(pctg*100) + "%"
    sample_len = int(len(df) * pctg)
    random_sample = X.take(np.random.permutation(len(df))[:sample_len])
    random_sample_encoded = random_sample.apply(encoder.fit_transform)
    print "Number of random samples: " + str(len(random_sample))

    """
        K-means clustering
    """
    x = np.array(random_sample_encoded)
    x = x.astype(int)

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
    standard_data = decimated_data[:, :-1]
    standard_data = StandardScaler().fit_transform(standard_data.astype(float))
    cluster_id_col = decimated_data[:, -1:]
    return standard_data, cluster_id_col


def squared_sum(arr):
    sqr_sum = 0.0
    for x in arr:
        sqr_sum += pow(x, 2)
    return sqr_sum


def highest_attributes(loading_attr, num):
    loading_attr = [(i, loading_attr[i]) for i in xrange(len(loading_attr))]
    loading_attr.sort(key=lambda x: x[1], reverse=True)

    attrs = []
    for x in xrange(num):
        attrs.append(loading_attr[x][0])
    return attrs


def dump_data_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv("static/data/" + filename, index=False)


def dimension_reduction(datapath, draw_plots):
    decimated_data, cluster_ids = decimate_data(datapath, draw_plots)
    print "Dimension reduction start..."
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

    principal_components = 0
    for x in pca.explained_variance_:
        if x > 1:
            principal_components += 1

    component_matrix = pca.components_.T[:, :principal_components]

    eigenvalues = pca.explained_variance_[:principal_components]
    print "EigenValues greater than or equal to 1: " + str(eigenvalues)
    loading_matrix = component_matrix * [math.sqrt(x) for x in eigenvalues]
    loading_arr = [squared_sum(x) for x in loading_matrix]

    highest_attrs = highest_attributes(loading_arr, 3)
    print "Highest Loaded Attributes: " + str(highest_attrs)

    transformed_data = pca_trans[:, 0:principal_components]
    dim_reduced_data = np.append(transformed_data, cluster_ids, 1)
    highest_attrs_data = np.append(decimated_data[:, highest_attrs], cluster_ids, 1)  # standardized data
    print "Dimension reduction ends"
    """
        Dump all the needed data for 
        D3 visualization
    """
    print "Dumping the data..."
    dump_data_to_csv(dim_reduced_data, 'dimension_reduced_data.csv')
    dump_data_to_csv(highest_attrs_data, 'highest_attrs_data.csv')
    for mds_type in list_mds:
        mds_data = find_mds(transformed_data, mds_type)
        dump_data_to_csv(np.append(mds_data, cluster_ids, 1), mds_type + '.csv')
    print"Dumped data"


def main():
    print "Pre-processing the data..."
    dimension_reduction("data/Letter_recognition.csv", False)
    print "Starting server.."
    app.run(host='localhost', port=2017, debug=True, use_reloader=False)

if __name__ == "__main__":
    main()