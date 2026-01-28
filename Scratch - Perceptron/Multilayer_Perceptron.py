import numpy as np
import pandas as pd


data = pd.read_excel(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\Scratch - Perceptron\student_maths.xlsb")
x = data.iloc[:,26:29].values
y = data.iloc[:,11:12].values
n_sample , n_features = x.shape

# print(data.head(10))
# for x in x(n_sample):
#     x = x - np.mean(x)/np.std(x)
# for y in y(n_sample):
#     y = y - np.mean(y)/np.std(y)

def normalization(x):
    x_ = x - (np.mean(x)/np.std(x))
    return np.array(x_)

x = normalization(x)

x = np.asarray(x)
y = np.asarray(y).ravel()

epoch = 100000
lr = 0.01
w1 = np.ones(n_features)
b1 = 0.0
w2 = np.ones(n_features)
b2 = 0.0

class Perceptron_Main:
    def __init__(self):
        self.w = None
        self.b = None

    def normalization(x):
        x_ = x - (np.mean(x)/np.std(x))
        return np.array(x_)

    def step(self, x):
        return 1 if x >= 0 else 0
    

    def relu(self, z):
        return np.maximum(0, z)

    @staticmethod
    def drelu(z):
        # derivative wrt z (not wrt h!)
        return (z > 0).astype(z.dtype)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dsigmoid(a):
        # given a = sigmoid(z)
        return a * (1 - a)

    def forward(self, x, w1, b1, b2, epoch, n_sample):
        for i in range(epoch):
            j = np.random.randint(n_sample)
            l1 = np.dot(w1,x[j])+b1
            af1 = self.relu(l1.T)
            l2 = np.dot(l1,x[j])+b2
            af2 = self.sigmoid(l2)
            self.af1 = af1
            #updated Weights & bises
            # w = w + lr * ((y[j] - y_) * x[j])
            # b = b + lr * (y[j] - y_)
            # self.w, self.b = w, b
        return af2

    def mse(y, y_):
        return np.mean((y - y_)**2)

    def backprop(self, y_):
        dL_dyhat = -2 * (y - y_)
        dyhat_daf2 = self.dsigmoid(self.af2)
        dl_daf2_ = dL_dyhat * dyhat_daf2

        gw2 = np.dot(self.af2.T, dl_daf2_)
        gb2 = np.sum(dl_daf2_)

        # w2 -= self.lr * gw2
        # b2 -= self.lr * gb2
        
        dl_dh =  dl_daf2_ * self.w2.T
        dh_daf1 = self.drelu(self.af1)
        dl_daf1 = dl_dh * dh_daf1

        gw1 = np.dot(self.af1.T, dl_daf1)
        gb1 = np.sum(dl_daf1)

        w1 -= self.lr * gw1
        b1 -= self.lr * gb1
    
    def step(self, lr=0.01):
        self.w2 -= lr * self.gw2
        self.b2 -= lr * self.gb2
        self.w1 -= lr * self.gw1
        self.b1 -= lr * self.gb1

    def fit(self, x, y, epochs=2000, lr=0.1, verbose_every=200):
        for ep in range(epochs):
            y_hat = self.forward(x)
            loss = self.mse(y, y_hat)

            self.backprop(y)
            self.step(lr)

            if verbose_every and ep % verbose_every == 0:
                print(f"Epoch {ep:4d} | Loss: {loss:.4f}")

    def predict(self, X, threshold=0.5):
        probs = self.forward(X)
        return (probs >= threshold).astype(int)

    # def predict(self, x):
    #     y_list = []
    #     for i in range(len(x)):
    #         y_ = np.dot(x[i],self.w)+self.b
    #         y_ = self.step(y_)
    #         y_list.append(y_)
    #     return y_list

    # def correct_predication(self, y_list, y):
    #     corr_prid = 0
    #     for i in range(len(y_list)):
    #         if y_list[i] == y[i]:
    #             corr_prid += 1
    #     return corr_prid
    

#     def all_method(self, y_list , y):
#         TP = 0
#         TN = 0
#         FP = 0
#         FN = 0
#         for i in range(len(y_list)):
#             if y_list == 0 and y == 0:
#                 TN += 1
#             elif y_list == 1 and y == 1:
#                 TP += 1
#             elif y_list == 0 and y == 1:
#                 FN += 1
#             else:
#                 FP += 1
#             self.TP = TP
#             self.TN = TN
#             self.FP = FP
#             self.FN = FN
#         return TP, TN, FP, FN
#         print(TP, TN, FP, FN)

#     def accuracy(self, y_list, y):
#         accu_score = self.correct_predication(y_list, y)/len(y)*100
#         # accu_score2 = (self.TP + self.TN)/(self.TP + self.TN + self.FP + self.FN)*100
#         return accu_score
#         # return accu_score2

#     def precision(self):
#         pre_score = self.TP/(self.TP + self.FP)
#         return pre_score
    
#     def recall(self):
#         self.TP = TP
#         self.FN = FN
#         recall_score = self.TP/(self.TP + self.FN)
#         return recall_score
    
#     def f1_score(self):
#         f1_score_ = 2 * (self.pre_score * self.recall_score)/(self.pre_score + self.recall_score)
#         return f1_score_


# model = Perceptron_Main()
# final_w, final_b = model.algo(x, y, w, b, epoch, lr, n_sample)
# y_pred = model.predict(x)

# acc = model.accuracy(y_pred, y)
# TP, TN, FP, FN = model.all_method(y_pred, y)
# prec = model.precision()
# # rec = model.recall()
# # f1 = model.f1_score()

# print(f"Final Weights: {final_w}")
# print(f"Final Bias: {final_b}")
# print(f"Accuracy: {acc:.2f}%")
# print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
# print(y_pred)
# print(f"Precision: {prec:.4f}")
# # print(f"Recall: {rec:.4f}")
# # print(f"F1 Score: {f1:.4f}")

model = Perceptron_Main()
# def forward(self, x, w1, b1, b2, epoch, n_sample):
model.forward(x, w1, b1, b2, epoch, n_sample)
# def fit(self, x, y, epochs=2000, lr=0.1, verbose_every=200):
model.fit(x, y, epochs=2000, lr=0.1, verbose_every=400)
preds, probs = model.predict(x)
print("\nFinal probabilities:", np.round(probs.ravel(), 4))
print("Final predictions  :", preds.ravel())
print("\nWeights and biases:")
print("w1:", np.round(model.w1, 4))
print("b1:", np.round(model.b1, 4))
print("w2:", np.round(model.w2, 4))
print("b2:", np.round(model.b2, 4))