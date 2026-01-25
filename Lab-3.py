import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import minkowski


#A1
def dot_product(a, b):
    s = 0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def euclidean_norm(v):
    s = 0
    for x in v:
        s += x * x
    return s ** 0.5


#A2
def mean(data):
    return sum(data) / len(data)


def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def std_dev(data):
    return variance(data) ** 0.5


def dataset_stats(matrix):
    return np.mean(matrix, axis=0), np.std(matrix, axis=0)


def centroid(class_data):
    return np.mean(class_data, axis=0)


#A3
def feature_stats(feature):
    return mean(feature), variance(feature)


#A4
def minkowski_distance_custom(a, b, p):
    s = 0
    for i in range(len(a)):
        s += abs(a[i] - b[i]) ** p
    return s ** (1 / p)


#A6
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


#A10
def knn_predict_custom(X_train, y_train, test_point, k):
    distances = []
    for i in range(len(X_train)):
        d = euclidean_distance(X_train[i], test_point)
        distances.append((d, y_train[i]))
    distances.sort(key=lambda x: x[0])
    labels = [label for _, label in distances[:k]]
    return max(set(labels), key=labels.count)


#A13
def confusion_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return cm, accuracy, precision, recall, f1

np.random.seed(0)

class0 = np.random.normal(2, 1, (50, 2))
class1 = np.random.normal(6, 1, (50, 2))

X = np.vstack((class0, class1))
y = np.array([0]*50 + [1]*50)

a = X[0]
b = X[1]

dp_custom = dot_product(a, b)
dp_np = np.dot(a, b)

norm_custom = euclidean_norm(a)
norm_np = np.linalg.norm(a)

centroid0 = centroid(class0)
centroid1 = centroid(class1)
centroid_distance = np.linalg.norm(centroid0 - centroid1)

feature = X[:, 0]
hist_data = np.histogram(feature, bins=10)

plt.hist(feature, bins=10)
plt.show()

p_values = range(1, 11)
distances = [minkowski_distance_custom(a, b, p) for p in p_values]

plt.plot(p_values, distances)
plt.show()

custom_mink = minkowski_distance_custom(a, b, 3)
scipy_mink = minkowski(a, b, 3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
accuracy_knn = knn.score(X_test, y_test)

pred_knn = knn.predict(X_test)

pred_custom = np.array([knn_predict_custom(X_train, y_train, x, 3) for x in X_test])

cm, acc, prec, rec, f1 = confusion_metrics(y_test, pred_knn)

k_vals = range(1, 12)
accs = []
for k in k_vals:
    preds = np.array([knn_predict_custom(X_train, y_train, x, k) for x in X_test])
    accs.append(np.mean(preds == y_test))

plt.plot(k_vals, accs)
plt.show()

