import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\IPL Predication Code-\Scratch_Decision_Tree_\california_housing_test_1.csv", sep=",")
# data = data[0]*2/2
# data = data.drop(['median_house_value', 'ocean_proximity'], axis=1)
# data = data.drop('ocean_proximity', axis=1)

# print(data.head(10))
x = data.iloc[:,0:-1]
y = data.iloc[:,-1]

x = np.asarray(x)
y = np.asarray(y)
print(x,y)

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

# TTSplit = TrainTestSplit_()
# x_train, y_train, x_test, y_test = TTSplit.traintestsplit(x,y)
# print(x_train, x_test, y_train, y_test)


# class evaluation_functions:
    # def MSE(self, y_test, x_test):
    #     if len(y_test) != len(x_test):
    #         raise ValueError("input arrays must have equal lengths")
        
    #     y_test = np.array(y_test)
    #     x_test = np.array(x_test)
    #     mse = np.mean((y_test - x_test)**2)

    #     return mse

    # def r2_score(self, y_test, x_test):
    #     if len(y_test) != len(x_test):
    #         raise ValueError("input arrays must have equal lengths")
        
    #     y_test = np.array(y_test)
    #     x_test = np.array(x_test)
    #     mean_x_test = np.mean(x_test)

    #     SSE = np.sum((y_test - x_test)**2)
    #     SST = np.sum((y_test - mean_x_test)**2)
    #     r2_score = 1 - (SSE/SST)

    #     return r2_score
    
class EvaluationFunctions:
    def MSE(self, y_test, y_pred):
        if len(y_test) != len(y_pred):
            raise ValueError("Input arrays must have equal lengths")

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        # Formula: (1/n) * Σ (y_test - y_pred)^2
        mse = np.mean((y_test - y_pred) ** 2)
        return mse


    def r2_score(self, y_test, y_pred):
        if len(y_test) != len(y_pred):
            raise ValueError("Input arrays must have equal lengths")

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        # Formula: R² = 1 - (Σ(y_test - y_pred)² / Σ(y_test - mean_y_test)²)
        mean_y_test = np.mean(y_test)
        SSE = np.sum((y_test - y_pred) ** 2)
        SST = np.sum((y_test - mean_y_test) ** 2)

        r2_score = 1 - (SSE / SST)
        return r2_score
    

class DecisionTreeRegressor:
    def __init__(self, n_features, min_sample_split=100, max_depth=2):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features

    def mse_(self, y):
        mean_y = np.mean(y)
        mse_value = np.mean((y - mean_y) ** 2)
        return mse_value

    def split(self, x, y):
        n_sample, n_features = np.shape(x)
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

        for j in range(n_features):
            # sort by feature j and keep X–y aligned
            order = np.argsort(x[:, j])
            x_sorted = x[order, j]
            y_sorted = y[order]

            # try all cut points between consecutive distinct values
            for i in range(1, n_sample):
                # skip if no real split at this boundary
                if x_sorted[i] == x_sorted[i - 1]:
                    continue

                # enforce min samples on both sides (if provided)
                # if (i < self.min_sample_split) or ((n_sample - i) < self.min_sample_split):
                #     continue

                # use TARGET slices for impurity (not X)
                y_left = y_sorted[:i]
                y_right = y_sorted[i:]

                # weights by proportion of samples
                w_mse_l_node = i / n_sample
                w_mse_r_node = 1 - w_mse_l_node

                # node impurities
                l_mse = self.mse_(y_left)
                r_mse = self.mse_(y_right)

                weighted_mse = (w_mse_l_node * l_mse) + (w_mse_r_node * r_mse)

                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = j
                    # midpoint between adjacent sorted feature values
                    best_threshold = (x_sorted[i - 1] + x_sorted[i]) / 2.0
                    best_left_idx = order[:i]
                    best_right_idx = order[i:]

        return {
            "feature_index": best_feature,
            "threshold": best_threshold,
            "weighted_mse": best_mse,
            "left_indices": best_left_idx,
            "right_indices": best_right_idx,
        }
    
    def _make_leaf(self, y):
    # Node dict for a leaf
        return {
            "is_leaf": True,
            "prediction": float(np.mean(y)) if len(y) > 0 else 0.0,
            "feature_index": None,
            "threshold": None,
            "left": None,
            "right": None,
            "n_samples": int(len(y)),
            "impurity": float(self.mse_(y)) if len(y) > 0 else 0.0,
        }
    
    def build_tree(self, X, y, depth=2):
        if depth >= self.max_depth or len(y) == 0 or self.mse_(y) < 1e-12:
            return self._make_leaf(y)
        
        split_info = self.split(X, y)
        j = split_info["feature_index"]
        t = split_info["threshold"]
        left_idx = split_info["left_indices"]
        right_idx = split_info["right_indices"]

        if j is None or t is None or left_idx is None or right_idx is None:
            return self._make_leaf(y)
        
        x_left, y_left = X[left_idx], y[left_idx]
        x_right, y_right = X[right_idx], y[right_idx]

        if len(y_left) == 0 or len(y_right) == 0:
            return self._make_leaf(y)
        
        left_child = self.build_tree(x_left, y_left, depth + 1)
        right_child = self.build_tree(x_right, y_right, depth + 1)

        return {
        "is_leaf": False,
        "prediction": None,
        "feature_index": int(j),
        "threshold": float(t),
        "left": left_child,
        "right": right_child,
        "n_samples": int(len(y)),
        "impurity": float(self.mse_(y)),
    }
    
    def fit(self, X, y):
        self.root = self.build_tree(np.asarray(X), np.asarray(y), depth=0)
        return self.root

    def _predict_one(self, x_row, node):
        if node["is_leaf"]:
            return node["prediction"]
        j = node["feature_index"]
        t = node["threshold"]
        if x_row[j] <= t:
            return self._predict_one(x_row, node["left"])
        else:
            return self._predict_one(x_row, node["right"])

    def predict(self, x):
        # Vectorized loop over rows
        return np.array([self._predict_one(x[i], self.root) for i in range(x.shape[0])], dtype=float)
    









# tree = DecisionTreeRegressor(n_features=x.shape[1], max_depth=3)
# tree = DecisionTreeRegressor(n_features=x.shape[1], max_depth=6)
# tree = DecisionTreeRegressor(n_features=x.shape[1], max_depth=8)
tree = DecisionTreeRegressor(n_features=x.shape[1], max_depth=20)
tree.fit(x, y)
y_pred = tree.predict(x)
print("Predictions:", y_pred[:10])          # first 10 predictions
print("Actual:", y[:10])                    # first 10 actual values

metrics = EvaluationFunctions()
mse_value = metrics.MSE(y, y_pred)
r2_value = metrics.r2_score(y, y_pred)

print("MSE:", mse_value)
print("R2 Score:", r2_value)