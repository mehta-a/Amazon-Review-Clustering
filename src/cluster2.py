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
def average_centroid(cluster, matrix, minPts):
    vectors = []
    if len(cluster)>minPts:
        for doc in cluster:
            vectors.append(matrix[doc])
        return np.mean(vectors, axis = 0)
    return None


def k_mean_modified(matrix, k, num_iterations=100, epsilon=2.0, min_pts=10):
    cluster_centroids = []
    clusters = {}

    for i in range(k):
        cluster_centroids.append(matrix[i])  # Let first rows(docs) be initial centroids

    # number of iterations we want to run the algorithm of k means
    for _iter in range(num_iterations):

        for i in range(k):
            clusters[i] = []  # clusters are empty right now.

        # Assign documents to clusters
        for doc in range(len(matrix)):
            cluster_dist = []
            min_dist = euclidean_distance(cluster_centroids[0], matrix[doc])
            closest_centroid = 0
            for centroid in range(1, len(cluster_centroids)):
                ed_1 = euclidean_distance(cluster_centroids[centroid], matrix[doc])
                if ed_1 <= min_dist and ed_1>epsilon: # is it a close enough point
                    closest_centroid = centroid
                    min_dist = ed_1
            clusters[closest_centroid].append(doc)

        # Print updated cluster
        # print(clusters)
        # print(cluster_centroids)

        # Take average and reset cluster centroids
        new_cluster_centroids = []
        for i in range(k):
            new_cluster_centroids.append(average_centroid(clusters[i], matrix, min_pts))

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


def get_epsilon_pts(matrix, point, epsilon):
    epsilon_pts = []
    for doc in range(len(matrix)):
        if point!=doc:
            point_dist = euclidean_distance(matrix[point], matrix[doc])
            if point_dist < epsilon:
                epsilon_pts.append(doc)
    return epsilon_pts


# Db scan
def db_scan_method(matrix, k, num_iterations=100, epsilon=2.0, min_pts=10):
    #cluster_centroids = []
    clusters = {}

    #for i in range(k):
    #    cluster_centroids.append(matrix[i])  # Let first rows(docs) be initial centroids

    # number of iterations we want to run the algorithm of k means
    for _iter in range(num_iterations):

        for i in range(k):
            clusters[i] = []  # clusters are empty right now.

        # Assumption: Let first point be 0'th row
        point = 0
        # Assign documents to clusters
        visited_pts = []
        core_pts = []
        for doc in range(1, len(matrix)):
            if point < k:
                if (doc in visited_pts) or (doc in core_pts) or (doc in clusters[point]): continue
                epsilon_pts = get_epsilon_pts(matrix, point, epsilon)
                # is this a core pt? if no, mark as noise
                if len(epsilon_pts)<min_pts:
                    visited_pts.append(point)
                    continue
                # if yes, start a cluster
                print(epsilon_pts)
                point = point + 1
                if point < k:
                    clusters[point].extend(epsilon_pts)
                    clusters[point] = list(set(clusters[point]))
                    for pt in epsilon_pts:
                        if pt in visited_pts: visited_pts.remove(pt)
                        if (pt in visited_pts) or (pt in core_pts): continue
                        # get this points epsilon region
                        epsilon_pts = get_epsilon_pts(matrix, pt, epsilon)
                        print(epsilon_pts)
                        if len(epsilon_pts)>=min_pts:
                            clusters[point].extend(epsilon_pts)
                            clusters[point] = list(set(clusters[point]))
                    core_pts.extend(clusters[point])
            '''
            closest_centroid = 0
            for centroid in range(1, len(cluster_centroids)):
                ed_1 = euclidean_distance(cluster_centroids[centroid], matrix[doc])
                if ed_1 <= min_dist and ed_1>epsilon: # is it a close enough point
                    closest_centroid = centroid
                    min_dist = ed_1
            clusters[closest_centroid].append(doc)
            '''
        # Print updated cluster
        # print(clusters)
        # print(cluster_centroids)

        # Take average and reset cluster centroids
        # new_cluster_centroids = []
        #for i in range(k):
        #    new_cluster_centroids.append(average_centroid(clusters[i], matrix, min_pts))

        # if there is no change in clusters, then pause the iterations
        #close_flag = True
        #for i in range(k):
        #    if not (cluster_centroids[i] == new_cluster_centroids[i]).all():
        #        close_flag = False

        # Assign Cluster centroid as new centroids
        #cluster_centroids = new_cluster_centroids

        #if close_flag:
        #   break

    return clusters, clusters.keys()

if __name__=="__main__":
    print(db_scan_method(np.array([[2, 2],[1,3]]), 2, 5, 0.1))