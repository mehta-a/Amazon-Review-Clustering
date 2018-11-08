# I. g) Perform k-means clustering on the rows of a matrix,
# given k and the matrix whose rows are to be clustered as input parameters
# (implement k-means from scratch and do not use any pre-existing packages
# that offer functionality to do this directly).
import math

import numpy as np


# Distance between two vectors
def euclidean_distance(vec_A, vec_B):
    ed = 0
    squared_distance = 0
    if len(vec_A)==len(vec_B):
        for i in range(len(vec_A)):
            squared_distance += (vec_A[i]-vec_B[i])**2
        ed = math.sqrt(squared_distance)
    else:
        print("Invalid input: vectors of different lengths encountered")
    return ed


# recalculate cluster centroid
def average_centroid(cluster, matrix):
    vectors = []
    for doc in cluster:
        vectors.append(matrix[doc])
    return np.mean(vectors, axis = 0)


def k_mean(matrix, k, num_iterations=100):
    cluster_centroids = []
    clusters = {}

    for i in range(k):
        cluster_centroids.append(matrix[i])  # Let first rows(docs) be intial centroids

    # number of iterations we want to run the algorithm of k means
    for _iter in range(num_iterations):

        for i in range(k):
            clusters[i] = []  # clusters are empty right now.

        # Assign documents to clusters
        for doc in range(len(matrix)):
            cluster_dist = []
            min_dist = euclidean_distance(cluster_centroids[0], matrix[doc])
            clostest_centroid = 0
            for centroid in range(1, len(cluster_centroids)):
                ed_1 = euclidean_distance(cluster_centroids[centroid], matrix[doc])
                if ed_1 <= min_dist:
                    clostest_centroid = centroid
                    min_dist = ed_1
            clusters[clostest_centroid].append(doc)

        # Print updated cluster
        # print(clusters)
        # print(cluster_centroids)

        # Take average and reset cluster centroids
        new_cluster_centroids = []
        for i in range(k):
            new_cluster_centroids.append(average_centroid(clusters[i], matrix))

        # if there is no change in clusters, then pause the iterations
        close_flag = True
        for i in range(k):
            if not (cluster_centroids[i] == new_cluster_centroids[i]).all():
                close_flag = False

        # Assign Cluster centroid as new centroids
        cluster_centroids = new_cluster_centroids

        if close_flag:
            break

    return clusters, cluster_centroids
