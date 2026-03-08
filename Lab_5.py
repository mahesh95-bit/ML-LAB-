# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt


# Function to load dataset
def load_data(path):
    data = pd.read_excel(path)
    return data



def preprocess_data(data):

   
    le = LabelEncoder()
    data['Label'] = le.fit_transform(data['Label'])

    
    X = data.filter(regex='glove_')

    y = data['Label']

    return X, y



def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



def train_linear_regression(X_train, y_train):

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model



def regression_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2 = r2_score(y_true, y_pred)

    return mse, rmse, mape, r2



def perform_kmeans(X, k):

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    return kmeans, labels, centers



def clustering_scores(X, labels):

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    return sil, ch, db



def evaluate_k_values(X):

    k_values = range(2, 10)

    sil_scores = []
    ch_scores = []
    db_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        sil_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))

    return k_values, sil_scores, ch_scores, db_scores

def elbow_method(X):

    distortions = []

    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    return distortions


# ===========================
# MAIN PROGRAM
# ===========================

data = load_data("Clarity_Text_student_teacher_with_glove.xlsx")

X, y = preprocess_data(data)

X_train, X_test, y_train, y_test = split_data(X, y)

# A1 Linear Regression
model = train_linear_regression(X_train[['glove_0']], y_train)

y_train_pred = model.predict(X_train[['glove_0']])
y_test_pred = model.predict(X_test[['glove_0']])


# A2 Metrics
train_metrics = regression_metrics(y_train, y_train_pred)
test_metrics = regression_metrics(y_test, y_test_pred)

print("Train Metrics (MSE, RMSE, MAPE, R2):", train_metrics)
print("Test Metrics (MSE, RMSE, MAPE, R2):", test_metrics)


# A3 Regression using all attributes
model_all = train_linear_regression(X_train, y_train)

train_pred_all = model_all.predict(X_train)
test_pred_all = model_all.predict(X_test)

print("Train Metrics (All Features):", regression_metrics(y_train, train_pred_all))
print("Test Metrics (All Features):", regression_metrics(y_test, test_pred_all))


# A4 K-Means clustering
kmeans, labels, centers = perform_kmeans(X_train, 2)

print("Cluster Centers Shape:", centers.shape)


# A5 Clustering Scores
scores = clustering_scores(X_train, labels)

print("Silhouette Score:", scores[0])
print("Calinski Harabasz Score:", scores[1])
print("Davies Bouldin Score:", scores[2])


# A6 Evaluate different k values
k_values, sil_scores, ch_scores, db_scores = evaluate_k_values(X_train)

plt.plot(k_values, sil_scores)
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")
plt.show()


# A7 Elbow method
distortions = elbow_method(X_train)

plt.plot(range(2,20), distortions)
plt.xlabel("k")
plt.ylabel("Distortion")
plt.title("Elbow Method")
plt.show()