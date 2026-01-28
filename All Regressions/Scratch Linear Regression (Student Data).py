import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\IPL Predication Code-\Linear Regression\student_maths.xlsb")

x = data.iloc[:,26].values
y = data.iloc[:,28].values

class Slr:

    def __init__(self):
        self.m = None
        self.b = None

    def traintestsplit(self, x, y):
        # define percentage
        split_ratio = 0.8 

        indices = np.arange(len(x))
        np.random.shuffle(indices)

        split_index = int(len(x) * split_ratio)

        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        self.x_train , self.y_train = x[train_indices], y[train_indices]
        self.x_test , self.y_test = x[test_indices], y[test_indices]
    #formula = (Xi-Xmean)*(Yi-Ymean)/(Xi-Xmean)**
    def fit(self, X_train, Y_train):
        num = 0
        den = 0

        for i in range(len(X_train)):

            num = num + (X_train[i] - X_train.mean()) * (Y_train[i] - Y_train.mean())
            den = den + (X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean())

            self.m = num/den
            self.b = Y_train.mean() - (self.m * X_train.mean())
    
    #formula = mx + b
    def predict(self,X_test):
        return self.m * X_test + self.b

model = Slr()
model.traintestsplit(x, y)

model.fit(model.x_train , model.y_train)

model.predict(model.x_test)

print("Slope (m):", model.m)
print("Intercept (b):", model.b)


plt.scatter(data['G1'],data['G2'])
plt.plot(model.x_train,model.predict(model.x_train),color='red')
plt.xlabel('G1')
plt.ylabel('G2')
plt.show()