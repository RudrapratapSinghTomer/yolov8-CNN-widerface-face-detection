import numpy as np
import pandas as pd

data = pd.read_excel(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\Scratch - Perceptron\student_maths.xlsb")

x = data.iloc[:,26:29].values
n_sample , n_features = x.shape


@staticmethod
def normalize(x):
    x = (x - np.mean(x))/np.std(x)
    return np.array(x)

@staticmethod
def euclidean(a,b):
    return np.sqrt((np.sum(a-b))**2)

@staticmethod
def manhattan(a, b):
    return np.sum(np.abs(a - b))

class agglomerative:
    def __init__(self, n_clusters=2, metric='euclidean', linkage='single'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.lable = None
        self.clusters = None

    def fit(self, x):
        clusters = [[x] for x in range(x)]
        best_dist = float("inf")
        best_i, best_j = None, None
        n = n_sample

        while len(clusters) > self.n_clusters:

            for i in range(len(x)):
                for j in range(i+1, len(x)):
                    d = self.distance_calc(x,
                                           clusters[i], 
                                                clusters[j], 
                                                    metric=self.metric, 
                                                        linkage=self.linkage)

                    if d < best_dist:
                        best_dist = d
                    best_i, best_j = i, j

            merged = best_i + best_j

            clusters.pop(j)
            clusters.pop(i)
            clusters.append(merged)

        labels = np.empty(n, dtype=int)
        for k, cl in enumerate(clusters):
            for idx in cl:
                labels[idx] = k  


        self.labels_ = labels
        self.clusters_ = clusters
        return self


    def distance_calc(self, x, cluster_a, cluster_b, metric='euclidean', linkage='single'):
        distance = []
        for i in range(cluster_a):
            for j in range(cluster_b):
                distance.append(euclidean(x[i],x[j]))
        return min(np.array(distance))
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_