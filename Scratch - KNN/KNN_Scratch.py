import numpy as np
import pandas as pd

data = pd.read_excel(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\Scratch - Perceptron\student_maths.xlsb")
x = data.iloc[:,26:29].values
y = data.iloc[:,11:12].values
n_sample , n_features = x.shape

class TrainTestSplit_:
    def traintestsplit(self, x, y, split_ratio=0.8, seed=42):
        np.random.seed(seed)
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        split_index = int(len(x) * split_ratio)

        train_indices = indices[:split_index]
        test_indices  = indices[split_index:]

        self.x_train, self.y_train = x.iloc[train_indices], y.iloc[train_indices]
        self.x_test , self.y_test  = x.iloc[test_indices] , y.iloc[test_indices]
        return self.x_train, self.y_train, self.x_test, self.y_test

class KNN():
    def __init__(self):
        pass

    @staticmethod
    def normalize(x):
        x = (x - np.mean(x))/np.std(x)
        return x 

    def euclidean(self, x):
            euc_dis_l = []
            for j in range(len(x)):
                euc_dis = np.sqrt(np.sum(x-x[j])**2)
                euc_dis_l.append(euc_dis)
            return euc_dis_l.sorted()

    def manhattan(self, x):
        man_dis_l = []
        for j in range(len(x)):
            man_dis = np.sum(abs(x-x[j]))
            man_dis_l.append(man_dis)
        return man_dis_l.sorted()
         
    def cross_validation(self, n_sample):
        x = np.round(np.sqrt(len(n_sample)))
        for i in range(1, x+1):
            n_neighbors = i
            if n_neighbors % 2 == 0:
                n_neighbors += 1
            else:
                continue
        return n_neighbors

    def algo(self, algo, x):
        euc_list = []
        man_list = []
        if algo == "euclidean":
            for i in range(len(x)):
                euc_list.append(self.euclidean(x))
            return euc_list
    
        elif algo == "manhattan":
            for i in range(len(x)):
                man_list.append(self.manhattan(x))
            return man_list

    def fit(self, x, y, epochs=2000, lr=0.1, verbose_every=200):
        for i in range(epochs):
            self.normalize(x)
            self.cross_validation(x)
            self.algo(algo="manhattan")
            if verbose_every and i % verbose_every == 0:
                return 

    def predict_reg(self, algo, x, n_neighbour):
        reg_list = []
        class_list = []
        half = np.round(n_neighbour/2)
        left, right = half, half-1
        if algo == "euclidean":
            for i in self.euc_list(left, right):
                reg_list.append(self.euc_list[i])
                new_dis_x = self.euclidean(x)
                reg_list.append(new_dis_x)
                reg_list = reg_list.sorted()
                return np.average(reg_list)
        
        if algo == "manhattan":
            for i in self.man_list(left, right):
                class_list.append(self.man_list[i])
                new_dis_x = self.manhattan(x)
                class_list.append(new_dis_x)
                class_list = class_list.sorted()
                return np.average(class_list)
        
        else:
        # Handle invalid algorithm choice
            return "Error: Invalid algorithm choice."

    # def predict_class(self, n_neighbour, x):
    #     half = np.round(n_neighbour/2)
    #     left, right = half, half-1
    #     for i in range(len(x)-left, len(x+right)):
    #         unique = [np.unique(x)]
    #         for j in range(len(unique)):
    #             count = unique.count[j]
    #     return count

    # def predict_reg(self, n_neighbour, x):
    #     reg_list = []
    #     half = np.round(n_neighbour/2)
    #     left, right = half, half-1
    #     for i in range(len(x)-left, len(x+right)):
    #         reg_list.append(i) 
    #         average = np.average(reg_list)
    #     return average