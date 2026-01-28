import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\IPL Predication Code-\Scratch_Decision_Tree_\california_housing_test_1.csv", sep=",")

x = data.iloc[:,0:-1]
y = data.iloc[:,-1]

X = np.asarray(x)
y = np.asarray(y)
# print(x,y)

n_feature, n_sample = np.shape(X)


class TrainTestSplit_:
    def traintestsplit(self, x, y, split_ratio=0.8, seed=42):
        np.random.seed(seed)
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

        indices = np.arange(len(x))
        np.random.shuffle(indices)
        split_index = int(len(x) * split_ratio)

        train_indices = indices[:split_index]
        test_indices  = indices[split_index:]

        self.x_train, self.y_train = x.iloc[train_indices], y.iloc[train_indices]
        self.x_test , self.y_test  = x.iloc[test_indices] , y.iloc[test_indices]
        return self.x_train, self.y_train, self.x_test, self.y_test



class SVM_Scratch:
    def __init__(self, w,b, lr,lambda_param=0.001, epochs=1000):
        self.w = np.ones(n_feature)
        self.b = 0
        self.lr = 0.01
        self.epochs = epochs
        self.lambda_param = lambda_param

    #StandardScaler
    def y_(y):
        return -1 if y <=0 else 1

    def fit(self, x, y):
        for idx, y_ in enumerate(self.epochs):
            condition = y_[idx] * (np.dot(self.w, x[idx]) - self.b)
            if condition:
                self.w -= self.lr * (2 *self.lambda_param * self.w)
            else:
                self.w -=  self.lr * (2 * (self.lambda_param * self.w) - (np.dot(x[idx], y_[idx])))
                self.b -= self.lr * y_[idx]
        return self.w, self.b

    def predict(self, x):
        prediction = np.sign(np.dot(self.w, x) - self.b)
        return prediction
    

    def accuracy(self, y_, prediction):
        corr_pred = 0
        total_pred = len(y_)
        for y_ in range(len(y_)):
            if y_ == prediction:
                corr_pred += 1
            else:
                continue
        return (corr_pred / total_pred)*100