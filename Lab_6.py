# A1–A7 Combined Implementation

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Dataset
np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=4,
                          n_informative=4, n_redundant=0,
                          random_state=42)


# A1: Entropy + Binning
def entropy(y):
    p = np.bincount(y) / len(y)
    return -np.sum(p[p > 0] * np.log2(p[p > 0]))

def bin_data(x, bins=4, mode='width'):
    if len(np.unique(x)) <= 10:
        return x
    edges = np.linspace(x.min(), x.max(), bins + 1) if mode == 'width' \
        else np.quantile(x, np.linspace(0, 1, bins + 1))
    return np.digitize(x, edges[:-1])


# A2: Gini Index
def gini(y):
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p**2)


# A3: Root Feature (Info Gain)
def info_gain(x, y, bins=4, mode='width'):
    x = bin_data(x, bins, mode)
    return entropy(y) - sum(
        (len(sub := y[x == v]) / len(y)) * entropy(sub)
        for v in np.unique(x)
    )

def find_root(X, y):
    return max(range(X.shape[1]), key=lambda i: info_gain(X[:, i], y))


# A5: Decision Tree Module
class Node:
    def __init__(self, f=None, t=None, l=None, r=None, v=None):
        self.f, self.t, self.l, self.r, self.v = f, t, l, r, v

class DecisionTree:
    def __init__(self, depth=4, min_split=5, bins=4, mode='width'):
        self.depth, self.min_split = depth, min_split
        self.bins, self.mode = bins, mode

    def fit(self, X, y):
        self.nf = X.shape[1]
        self.root = self._build(X, y, 0)

    def _best_split(self, X, y):
        best = (-1, None, None)
        for i in range(self.nf):
            x = bin_data(X[:, i], self.bins, self.mode)
            for t in np.unique(x):
                l, r = y[x <= t], y[x > t]
                if len(l) < self.min_split or len(r) < self.min_split:
                    continue
                gain = entropy(y) - (
                    len(l)/len(y)*entropy(l) +
                    len(r)/len(y)*entropy(r)
                )
                if gain > best[0]:
                    best = (gain, i, t)
        return best[1], best[2]

    def _build(self, X, y, d):
        if d == self.depth or len(y) < self.min_split or len(set(y)) == 1:
            return Node(v=Counter(y).most_common(1)[0][0])
        f, t = self._best_split(X, y)
        if f is None:
            return Node(v=Counter(y).most_common(1)[0][0])
        x = bin_data(X[:, f], self.bins, self.mode)
        return Node(f, t,
                    self._build(X[x <= t], y[x <= t], d+1),
                    self._build(X[x > t], y[x > t], d+1))

    def predict(self, X):
        return np.array([self._pred(x, self.root) for x in X])

    def _pred(self, x, node):
        if node.v is not None:
            return node.v
        val = bin_data(np.array([x[node.f]]), self.bins, self.mode)[0]
        return self._pred(x, node.l if val <= node.t else node.r)


# A6: Tree Visualization
def print_tree(node, depth=0):
    if node.v is not None:
        print("  "*depth + f"Leaf:{node.v}")
        return
    print("  "*depth + f"X{node.f} <= {node.t}")
    print_tree(node.l, depth+1)
    print_tree(node.r, depth+1)

def visualize_sklearn_tree(X, y):
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X, y)
    plt.figure(figsize=(10,6))
    plot_tree(clf, filled=True)
    plt.show()

# A7: Decision Boundary
def plot_boundary(model, X, y):
    X2 = X[:, :2]
    x_min, x_max = X2[:, 0].min()-1, X2[:, 0].max()+1
    y_min, y_max = X2[:, 1].min()-1, X2[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel(), np.zeros((len(xx.ravel()), 2))]
    Z = model.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X2[:, 0], X2[:, 1], c=y)
    plt.show()

# Execution

# A1, A2
print("Entropy:", round(entropy(y), 4))
print("Gini:", round(gini(y), 4))

# A3
root = find_root(X, y)
print("Root Feature:", root + 1)

# Train/Test
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3)

# A5
tree = DecisionTree()
tree.fit(Xtr, ytr)

pred = tree.predict(Xte)

# Metrics (extra for report)
print("Accuracy:", round(np.mean(pred == yte), 4))
print("Confusion Matrix:\n", confusion_matrix(yte, pred))
print("Report:\n", classification_report(yte, pred))

# A6
print("\nCustom Tree Structure:")
print_tree(tree.root)

visualize_sklearn_tree(Xtr, ytr)

# A7
plot_boundary(tree, X, y)