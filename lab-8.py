import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# core units
def net(x, w, b): return np.dot(x, w) + b

def step(x): return 1 if x >= 0 else 0
def bipolar(x): return 1 if x >= 0 else -1
def sigmoid(x): return 1/(1+np.exp(-x))
def relu(x): return max(0, x)

def train(X, y, w, b, lr, act, max_ep=1000):
    errors = []
    for ep in range(max_ep):
        e = 0
        for i in range(len(X)):
            o = act(net(X[i], w, b))
            err = y[i] - o
            w += lr * err * X[i]
            b += lr * err
            e += err**2
        errors.append(e)
        if e <= 0.002: break
    return w, b, errors, ep+1

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])
y_xor = np.array([0,1,1,0])

# A2
w0 = np.array([0.2, -0.75])
b0 = 10
_, _, err_and, ep_and = train(X, y_and, w0.copy(), b0, 0.05, step)

# A3
acts = [bipolar, sigmoid, relu]
act_results = {}
for a in acts:
    _, _, _, ep = train(X, y_and, w0.copy(), b0, 0.05, a)
    act_results[a.__name__] = ep

# A4
lrs = np.arange(0.1, 1.1, 0.1)
lr_epochs = []
for lr in lrs:
    _, _, _, ep = train(X, y_and, w0.copy(), b0, lr, step)
    lr_epochs.append(ep)

# A5
_, _, _, ep_xor = train(X, y_xor, w0.copy(), b0, 0.05, step)

# A6
Xc = np.array([
    [20,6,2,386],[16,3,6,289],[27,6,2,393],
    [19,1,2,110],[24,4,2,280],[22,1,5,167],
    [15,4,2,271],[18,4,2,274],[21,1,4,148],
    [16,2,4,198]
])
yc = np.array([1,1,1,0,1,0,1,1,0,0])
w_c = np.random.rand(4)
b_c = np.random.rand()
_, _, _, ep_cust = train(Xc, yc, w_c, b_c, 0.01, sigmoid)

# A7
def pseudo_inv(X, y):
    Xb = np.c_[np.ones(len(X)), X]
    return np.linalg.pinv(Xb) @ y

pinv_w = pseudo_inv(X, y_and)

# A8
def backprop(X, y, lr=0.05, ep=1000):
    w1 = np.random.uniform(-0.5,0.5,(2,2))
    w2 = np.random.uniform(-0.5,0.5,(2,1))
    errs = []
    for e in range(ep):
        tot = 0
        for i in range(len(X)):
            h = sigmoid(np.dot(X[i], w1))
            o = sigmoid(np.dot(h, w2))
            err = y[i] - o
            tot += err**2
            d_o = err * o * (1-o)
            d_h = h*(1-h)*(w2.flatten()*d_o)
            w2 += lr * np.outer(h, d_o)
            w1 += lr * np.outer(X[i], d_h)
        errs.append(tot)
        if tot <= 0.002: break
    return errs, e+1

_, ep_bp_and = backprop(X, y_and)

# A9
_, ep_bp_xor = backprop(X, y_xor)

# A10
def encode(y):
    return np.array([[1,0] if i==0 else [0,1] for i in y])

encoded = encode(y_and)

# A11
mlp_and = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000).fit(X, y_and)
mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000).fit(X, y_xor)

# A12
sc = StandardScaler()
Xc_scaled = sc.fit_transform(Xc)
mlp_cust = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000).fit(Xc_scaled, yc)

# outputs
print("AND epochs:", ep_and)
print("Activation comparison:", act_results)
print("LR vs epochs:", list(zip(lrs, lr_epochs)))
print("XOR perceptron epochs:", ep_xor)
print("Customer epochs:", ep_cust)
print("Pseudo-inverse weights:", pinv_w)
print("Backprop AND epochs:", ep_bp_and)
print("Backprop XOR epochs:", ep_bp_xor)
print("Encoded outputs:", encoded)
print("MLP AND acc:", mlp_and.score(X, y_and))
print("MLP XOR acc:", mlp_xor.score(X, y_xor))
print("MLP customer trained")

plt.plot(lrs, lr_epochs)
plt.xlabel("Learning Rate")
plt.ylabel("Epochs")
plt.show()