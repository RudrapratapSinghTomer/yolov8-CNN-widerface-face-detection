import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import joblib


# data = pd.read_csv("student_maths.xlsb", sep=";") # This line is incorrect as the file is in xlsb format, not csv
# Correct way to read xlsb file using pyxlsb
data = pd.read_excel("student_maths.xlsb", engine="pyxlsb")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "guardian", "health", "sex", "age", "internet"]]
predict = "G3"
print(data.head())

# Giveing the data to model
x = np.array(data.drop(columns=[predict]))
y = np.array(data[predict])


best=0
for _ in range(1000):

    # Split data into 80% train and 20% test
    # x, x_leftover, y, y_leftover = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    #fitting the model.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # predictions = model.predict(x_test)
    # accuracy = model.score(x_test, y_test)
    # print("accuracy (r^2 score):", accuracy)

    # Create a linear regression model
    our_Training = linear_model.LinearRegression()
    our_Training.fit(x_train, y_train)

    # Make predictions on the test set
    our_accuracy=our_Training.score(x_test, y_test)
    print(our_accuracy)
    if our_accuracy > best:
        best = our_accuracy
        # Save the model
        joblib.dump(our_Training, "linear_model_student.pkl")


# knowing about how are model will change the data
# Coefficients are the weights assigned to each feature in the linear regression model.
# The intercept is the value of the target variable when all features are zero.
print("Coefficients:\n", our_Training.coef_)
print("Intercept:\n", our_Training.intercept_)

# Use the trained model to make predictions on the test data
predicated_grade = our_Training.predict(x_test)

for i in range(len(predicated_grade)):
    print(predicated_grade[i], y_test[i], x_test[i])


# referencing the actual variables (predicted, actual)
predicted = predicated_grade
actual = y_test

# Plotting the predicted vs actual grades
plt.plot(predicted, label="Predicted Grades", marker='o')
plt.plot(actual, label="Actual Grades", marker='x')
plt.title("Predicted vs Actual Student Grades")
plt.xlabel("Student Index")
plt.ylabel("Grade (G3)")
plt.legend()
plt.grid(True)
plt.show()