from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


def sigmoid_activation(x):
    # compute the sigmoid activatin function for a given input
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    # calculates the derivative of the sigmoid function
    # assuming X has already been passed into the sigmoid
    # activation function
    return x * (1-x)# e = np.exp(-x)
    #return e / (1 + e)**2

def predict(X, W, threshold=0.5):
    # compute prediction by passing the dot product of W and X
    # into the sigmoid activation funcrtion and the converting
    # the outputs to binary class labels
    preds = sigmoid_activation(X.dot(W))
    preds[preds <= threshold] = 0
    preds[preds > threshold] = 1

    return preds

def next_batch(X, y, batch_size):
    # loop over dataset and return batches of it
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i : i + batch_size], y[i : i + batch_size])

BATCH = 64
EPOCHS = 100
LEARNING_RATE = 0.01

# generates a 2 class classification dataset with 1000 data points
# wheere is data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)


# do the bias trick by adding an extra column
# of ones to the feature matrix and reshapes
# the class vectors to 2D vectors
X = np.c_[X, np.ones((X.shape[0]))]
y = y.reshape((-1, 1))

# split the datasets into 50% training and 50% testing
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.5, random_state=42)

# initializing the weight matrix and list of losses
W = np.random.randn(X.shape[1], 1)
losses = []

print("[INFO]: Training ....")
for epoch in np.arange(0, EPOCHS):
    # initialize the list to record loss for each epoch
    epoch_loss = []
    for (x_batch, y_batch) in next_batch(x_train, y_train, batch_size=BATCH):
        # compute the predictions by passing the
        #  dot producr of the training batch data and the weight
        # into the sigmoid function 
        predictions = sigmoid_activation(x_batch.dot(W))

        # compute the error: how far out prediction and
        # actual classes are far apart from eachother
        error = predictions - y_batch
        epoch_loss.append(np.sum(error ** 2))

        # compute the gradient
        delta = error * sigmoid_deriv(predictions)
        gradient = x_batch.T.dot(delta)

        # wieght update
        W += -LEARNING_RATE * gradient

    # update loss history by taking the average loss across
    # all batches
    loss = np.average(epoch_loss)
    losses.append(loss)
    # display info during training
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"[INFO]: loss={loss} epoch={epoch+1}")

# evaluating model
pred = predict(x_test, W)
print("[INFO]: evaluating model")
print(classification_report(y_test, pred))

# plotting the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("DAta")
plt.scatter(x_test[:, 0], x_test[:, 1], marker="o", c=y_test[:, 0], s=30)


# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS, ), losses)
plt.title("Training Loss")
plt.xlabel("Epoch number")
plt.ylabel("Loss")
plt.show()