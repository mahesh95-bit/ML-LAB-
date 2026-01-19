import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder



#A1
#Purchase Data
df = pd.read_excel(file_name, sheet_name="Purchase data")

# column names
cols = list(df.columns)

# Feature matrix
X = df[[cols[1], cols[2], cols[3]]].values
y = df[cols[4]].values

# Conversion to numeric
X = X.astype(float)
y = y.astype(float)

print("A1: Purchase Data")
print("Number of features:", X.shape[1])
print("Number of samples:", X.shape[0])

# Rank of the matrix
rank = np.linalg.matrix_rank(X)
print("Rank of feature matrix:", rank)

# Pseudo-inverse and cost calculation
X_pinv = np.linalg.pinv(X)
cost = X_pinv.dot(y)

print("Cost of each product:", cost)

#A2
print("A2: Customer Classification")

category = []

for amount in y:
    if amount > 200:
        category.append("RICH")
    else:
        category.append("POOR")


# 3

def load_stock_data(file_path):
    return pd.read_excel(file_path, sheet_name="IRCTC Stock Price")


def manual_mean(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)


def manual_variance(data):
    mean = manual_mean(data)
    total = 0
    for value in data:
        total += (value - mean) ** 2
    return total / len(data)


def execution_time(func, data):
    times = []
    for _ in range(10):
        start = time.time()
        func(data)
        times.append(time.time() - start)
    return sum(times) / len(times)



#4

def load_thyroid_data(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def explore_data(df):
    info = df.info()
    description = df.describe(include="all")
    missing = df.isnull().sum()
    return info, description, missing





# MAIN FUNCTION

def main():
    file_path = "Lab Session Data.xlsx"

    # 1
    X, y = load_purchase_data(file_path)
    print("Dimensionality:", X.shape[1])
    print("Number of vectors:", X.shape[0])
    print("Rank of matrix:", calculate_rank(X))
    print("Cost of products:\n", calculate_cost_pinv(X, y))

    # 2
    labels = create_labels(y)
    model = train_classifier(X, labels)
    print("Classifier trained successfully")

    # 3
    stock_df = load_stock_data(file_path)
    price = stock_df.iloc[:, 3]

    print("Mean (numpy):", np.mean(price))
    print("Variance (numpy):", np.var(price))
    print("Mean (manual):", manual_mean(price))
    print("Variance (manual):", manual_variance(price))
    print("Time numpy mean:", execution_time(np.mean, price))
    print("Time manual mean:", execution_time(manual_mean, price))

    wed_prices = stock_df[stock_df["Day"] == "Wednesday"].iloc[:, 3]
    print("Wednesday Mean:", np.mean(wed_prices))

    april_prices = stock_df[stock_df["Month"] == "Apr"].iloc[:, 3]
    print("April Mean:", np.mean(april_prices))

    loss_prob = len(list(filter(lambda x: x < 0, stock_df["Chg%"]))) / len(stock_df)
    print("Probability of loss:", loss_prob)

    # Scatter plot
    plt.scatter(stock_df["Day"], stock_df["Chg%"])
    plt.title("Chg% vs Day")
    plt.show()

    #4
    thyroid_df = load_thyroid_data(file_path)
    _, _, missing = explore_data(thyroid_df)
    print("Missing values:\n", missing)



# RUN
if __name__ == "__main__":

    main()
