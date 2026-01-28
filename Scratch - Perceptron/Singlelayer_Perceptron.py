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
w = np.ones(n_features)
b = 0.0

class Perceptron_Main:
    def __init__(self):
        self.w = None
        self.b = None

    def normalization(x):
        x_ = x - (np.mean(x)/np.std(x))
        return np.array(x_)

    def step(self, x):
        return 1 if x >= 0 else 0

    def algo(self, x, y, w, b, epoch, lr, n_sample):
        for i in range(epoch):
            j = np.random.randint(n_sample)
            y_ = np.dot(w,x[j])+b
            #updated Weights & bises
            w = w + lr * ((y[j] - y_) * x[j])
            b = b + lr * (y[j] - y_)
            self.w, self.b = w, b
        return w, b

    def predict(self, x):
        y_list = []
        for i in range(len(x)):
            y_ = np.dot(x[i],self.w)+self.b
            y_ = self.step(y_)
            y_list.append(y_)
        return y_list

    def correct_predication(self, y_list, y):
        corr_prid = 0
        for i in range(len(y_list)):
            if y_list[i] == y[i]:
                corr_prid += 1
        return corr_prid
    

    def all_method(self, y_list , y):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(y_list)):
            if y_list == 0 and y == 0:
                TN += 1
            elif y_list == 1 and y == 1:
                TP += 1
            elif y_list == 0 and y == 1:
                FN += 1
            else:
                FP += 1
            self.TP = TP
            self.TN = TN
            self.FP = FP
            self.FN = FN
        return TP, TN, FP, FN
        print(TP, TN, FP, FN)

    def accuracy(self, y_list, y):
        accu_score = self.correct_predication(y_list, y)/len(y)*100
        # accu_score2 = (self.TP + self.TN)/(self.TP + self.TN + self.FP + self.FN)*100
        return accu_score
        # return accu_score2

    def precision(self):
        pre_score = self.TP/(self.TP + self.FP)
        return pre_score
    
    def recall(self):
        self.TP = TP
        self.FN = FN
        recall_score = self.TP/(self.TP + self.FN)
        return recall_score
    
    def f1_score(self):
        f1_score_ = 2 * (self.pre_score * self.recall_score)/(self.pre_score + self.recall_score)
        return f1_score_


model = Perceptron_Main()
final_w, final_b = model.algo(x, y, w, b, epoch, lr, n_sample)
y_pred = model.predict(x)

acc = model.accuracy(y_pred, y)
TP, TN, FP, FN = model.all_method(y_pred, y)
prec = model.precision()
# rec = model.recall()
# f1 = model.f1_score()

print(f"Final Weights: {final_w}")
print(f"Final Bias: {final_b}")
print(f"Accuracy: {acc:.2f}%")
print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print(y_pred)
print(f"Precision: {prec:.4f}")
# print(f"Recall: {rec:.4f}")
# print(f"F1 Score: {f1:.4f}")