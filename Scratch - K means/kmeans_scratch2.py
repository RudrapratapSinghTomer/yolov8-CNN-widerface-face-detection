import pandas as pd
import numpy as np

data = pd.read_excel(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\Scratch - Perceptron\student_maths.xlsb")

x = data.iloc[:,26:29].values
y = data.iloc[:,11:12].values
n_sample , n_features = x.shape

@staticmethod
def normalize(x):
    x = (x - np.mean(x))/np.std(x)
    return x 

class Kmean():
    def __init__(self, k=5, epoch=1000):
        self.k = k
        self.epoch = epoch
        self.centroids = None


    def predit(self, x):
        old_centroids = np.random.choice(n_sample, self.k, replace=False)
        for i in range():
            cluster_group = self.assign_cluster(x)
            self.centroids = self.move_centroids()
        pass

    def assign_cluster(self, x):
        cluster_group = []
        distance = []

        for row in x():
            for centroid in self.centroids:
                distance.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distance = min(distance)
            distance_index = distance.index(min_distance)
            cluster_group.append(distance_index)
            distance.clear()
        return np.array(cluster_group)

    def move_centroids(self, x, cluster_group):
        new_centroids = []

        centroid_type = np.unique(cluster_group)

        for type in centroid_type:
            new_centroids.append(centroid_type == type).mean(axis=0)
        return np.array(new_centroids)