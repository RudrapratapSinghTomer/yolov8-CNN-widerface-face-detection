import numpy as np
import pandas as pd
from collections import Counter

data = pd.read_csv(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\IPL Predication Code-\Scratch_Decision_Tree_\california_housing_test_1.csv", sep=",")

x = data.iloc[:,0:-1]
y = data.iloc[:,-1]

X = np.asarray(x)
y = np.asarray(y)
# print(x,y)

n_feature, n_sample = np.shape(X)

class RandomForest():
    def __init__(self, trees, max_depth=10, min_sample_split=2):
        self.n_trees = []
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        n_feature, n_sample = n_feature, n_sample
        self.trees = trees
        
    def fit(self):
        trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressor(min_sample_split=self.min_sample_split, max_depth=self.max_depth, n_features=n_feature)
            x_samples, y_samples = self.random_samples(x,y, n_sample=n_sample)
            tree.fit(x_samples,y_samples)
            trees.append(tree)
    
    def random_samples(self, x,y, n_sample):
        indx = np.random.choice(n_sample, n_sample, replace=True)
        return x[indx], y[indx]

    def most_common_lable(self, y):
        counter = Counter(y)
        most_common_counter = counter.most_common(1)[0][0]
        return most_common_counter

    def predict(self, x):
        prediction = np.array([tree.predict(x) for tree in self.trees])
        swape_indx = np.swapaxes(prediction,0,1)
        prediction = np.array([self.most_common_lable(pred) for pred in swape_indx])
        return prediction