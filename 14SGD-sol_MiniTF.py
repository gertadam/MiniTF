"""
Check out the new network architecture and dataset!

Notice that the weights and biases are
generated randomly.

No need to change anything, but feel free to tweak
to test your network, play around with the epochs, batch size, etc!
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from miniflow import *

# Load data
data = load_boston()
X_ = data['data']
y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(y, l2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

learn=0.008
epochs = 610
num_lines=10
devider=epochs/num_lines

# Total number of examples
m = X_.shape[0]
batch_size = 12
steps_per_epoch = m // batch_size

lastloss=0
last2=0
converging=0

graph = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

print("Epochs-i*batch*steps:",epochs*batch_size*steps_per_epoch)
print("Total number of examples = {}".format(m))

# Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        forward_and_backward(graph)

        # Step 3
        sgd_update(trainables,learn)

        loss += graph[-1].value
    lastloss = loss/steps_per_epoch
    if (i%devider==0):
        print("Epoch: {}, learnrate: {:.5f}, last:{:.2f}, Loss: {:.2f}".format(i+1, learn, last2, lastloss))
        if (lastloss <= last2):
            converging = 1
        last2 = lastloss
        if (converging==1):
            learn *= 0.80
            converging=0
    