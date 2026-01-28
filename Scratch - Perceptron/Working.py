import numpy as np
import pandas as pd


data = pd.read_excel(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\Scratch - Perceptron\student_maths.xlsb")
x = data.iloc[:,26:29].values
y = data.iloc[:,11:12].values
# n_sample , n_features = x.shape

# # print(data.head(10))
# for x in x(n_sample):
#     x = x - np.mean(x)/np.std(x)
# y = data.iloc[:,11:12].values
# for y in y(n_sample):
#     y = y - np.mean(y)/np.std(y)
# x = np.asarray(x)
# y = np.asarray(y).ravel()


def normalization(x):
    x_ = x - (np.mean(x)/np.std(x))
    return np.array(x_)

x = normalization(x)
y = normalization(y)

x = np.asarray(x)
y = np.asarray(y).ravel()

print(x)
print(y)