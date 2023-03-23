import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from PIL import Image as PIL_Image
from IPython.display import Image
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state
import random
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
random.seed(9)

def confusion_matrix(true_y, predicted_y):
    n_samples = len(true_y)
    label_counts = dict()
    for i in range(n_samples):
        if predicted_y[i] not in label_counts:
            label_counts[predicted_y[i]] = dict()
        if true_y[i] not in label_counts[predicted_y[i]]:
            label_counts[predicted_y[i]][true_y[i]] = 0
        label_counts[predicted_y[i]][true_y[i]] += 1
    
    most_common_labels = {k: max(v, key=v.get) for k, v in label_counts.items()}
    new_preds = [most_common_labels[predicted_y[i]] for i in range(n_samples)]
    
    labels = np.unique(true_y)
    n_labels = len(labels)
    confusion_matrix = np.zeros((n_labels, n_labels), dtype=int)
    for i in range(n_samples):
        true_label_index = np.argwhere(labels == true_y[i])[0][0]
        label_index = np.argwhere(labels == new_preds[i])[0][0]
        confusion_matrix[true_label_index][label_index] += 1

    return confusion_matrix

def select_centroids(X: np.ndarray, k: int):

    centroids = [X[np.random.choice(X.shape[0])]]

    for _ in range(1, k):
        D2 = np.array([min([np.linalg.norm(x-c)**2 for c in centroids]) for x in X])
        next_centroid = X[np.argmax(D2)]
        centroids.append(next_centroid)

    return np.array(centroids)

def kmeans_simple(X: np.ndarray, k: int, centroids=None, max_iter=50, tolerance=1e-2):
    n_samples, n_features = X.shape

    if centroids is None:
        centroids = X[np.random.choice(n_samples, k, replace=False)]
    else:
        k = centroids.shape[0]

    for i in range(max_iter):
        
        distances = np.zeros((n_samples, k))
        
        for j in range(k):
            distances[:, j] = np.linalg.norm(X - centroids[j], axis=1)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        centroids_change = np.linalg.norm(new_centroids - centroids)
        
        if centroids_change <= tolerance:
            break

        centroids = new_centroids

    return centroids, labels

def kmeans_wcss(X: np.ndarray, k: int, centroids=None, max_iter=50, tolerance=1e-2):
    n_samples, n_features = X.shape

    if centroids is None:
        centroids = X[np.random.choice(n_samples, k, replace=False)]
    else:
        k = centroids.shape[0]

    wcss = 0

    for i in range(max_iter):
        # Compute the distances between each data point and the centroids
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.linalg.norm(X - centroids[j], axis=1)
            
        labels = np.argmin(distances, axis=1)

        for j in range(k):
            wcss += np.sum((X[labels == j] - centroids[j])**2)
            
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        centroids_change = np.linalg.norm(new_centroids - centroids)
        if centroids_change <= tolerance:
            break

        centroids = new_centroids

    return centroids, labels, wcss


def kmeans(X: np.ndarray, k: int, centroids=None, max_iter=50, tolerance=1e-2):
    n_rows, n_cols, n_channels = X.shape
    n_samples = n_rows * n_cols
    X_reshape = X.reshape(n_samples, n_channels)

    if centroids is None:
        centroids = X_reshape[np.random.choice(n_samples, k, replace=False)]
    elif centroids == 'kmeans++':
        centroids = select_centroids(X_reshape, k)
    else:
        k = centroids.shape[0]

    for i in range(max_iter):
        distances = np.zeros((n_samples, k))
        for j in range(k):
            centroid = centroids[j].reshape(1, -1)
            distances[:, j] = np.sqrt(np.sum((X_reshape - centroid)**2, axis=1))

        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X_reshape[labels == i].mean(axis=0) for i in range(k)])

        centroids_change = np.linalg.norm(new_centroids - centroids)
        if centroids_change <= tolerance:
            break

        centroids = new_centroids

    return centroids, labels