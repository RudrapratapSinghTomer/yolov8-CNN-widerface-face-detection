import numpy as np
import pandas as pd
import random

data = pd.read_excel(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\IPL Predication Code-\Linear Regression\student_maths.xlsb")

x = data.iloc[:,26:28].values
y = data.iloc[:,28].values

#Hyperparameters
x = np.insert(x,0,1, axis=1)
w = np.ones(x.shape[1])
lr = 0.1
epoch = 1000

class MeraLR:
    def __init__(self):
        pass

    #Activitation Funcation: Step
    def step(self, z):
        # lambda: 1 if z > 0 else 0
        return 1 if z > 0 else 0
        # return z

    #Activitation Funcation: Sigmoid
    def sigmoid(self, z):
        z = 1/(1+np.exp(-z))
        return z
    
    #Gradient Descent
    def gd(self, x, y, w, lr, epoch):
        for i in range(epoch):
            y_hat = self.sigmoid(np.dot(x, w))
            # Old weights + Learning Rate * (Dot product of original value - predicted value) with x & divided with total number of rows
            U_w = w - lr*(np.dot((y-y_hat),x)/x.shape[0])
            # U_w = w + self.N_lr*(np.dot((y-y_hat),x)/x.shape[0])
            return

    #Stochastic Gradient Descent (SGD)
    def S_Trick(self, x, y, w, lr, epoch):
        for i in range(epoch):
            j = np.random.randint(1,100)
            y_hat = self.step(np.dot(x[j], w))
            U_w = w+lr*(y[j] - y_hat)*x[j]
            # U_w = w+lr*(y[j] - y_hat)*x[j]

            return U_w[0] , U_w[1:]
        
    #From Chat GPT
    #Binary Cross-Entropy Loss Funcation  
    def LF(self):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
        m = len(y)
        loss = - (1/m) * np.sum(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    )
        return loss
    
    def MAE(self, y, y_hat):
        absolute_error = np.abs(y - y_hat)
        mae = np.mean(absolute_error)
        return mae

    def MSE(self, y, y_hat):
        mse = np.square(y - y_hat)
        return mse

model = MeraLR()
bias1, weights1 = model.S_Trick(x, y, w, lr, epoch)
bias2, weights2 = model.gd(x, y, w, lr, epoch)

print("Bias (intercept):", bias1)
print("Bias (intercept):", bias2)
print("Weights:", weights1)
print("Weights:", weights2)