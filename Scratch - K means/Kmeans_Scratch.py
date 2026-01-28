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

@staticmethod
def WCSS(self, x, centroids):
    sum = 0
    for i, point in enumerate(range(x)):
        sum += np.sum(np.dot((x[i]-centroids[i]))**2)        
    return sum

class Kmean():
    def __init__(self, k=5, epoch=1000):
        self.k = k
        self.epoch = epoch
        self.centroids = None

    def predict(self, x):
        random_index = np.random.sample(range(0,x.shape[0]),self.n_clusters)
        self.centroids = x[random_index]

        for i in range(self.epoch):
            cluster_group = self.assign_cluster(x)
            old_centroids = self.centroids

            self.centroids = self.move_centroids(x, cluster_group)

            if (old_centroids == self.centroids).all():
                break

        return cluster_group

    def assign_cluster(self):
        cluster_group = []
        distance = []

        for row in x:
            for centroid in self.centroids:
                distance.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distance = min(distance)
            index_pos = distance.index(min_distance)
            cluster_group.append(index_pos)
            distance.clear
        np.array(cluster_group)


    def move_centroids(self, x, cluster_group):
        new_centroids = []

        cluster_type = np.unique(self.cluster_group)

        for type in cluster_type:
            new_centroids.append(x[cluster_type == type].mean(axis=0))
        
        return np.array(new_centroids)