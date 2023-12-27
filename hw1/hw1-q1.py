#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import utils


def relu(z):
    return z * (z > 0)


def relu_d(z):
    return 1. * (z > 0)


def softmax(z):
    z = z - np.max(z)
    return np.exp(z) / np.sum(np.exp(z))


def cross_entropy(YPred, Y):
    return -(np.log(Y.T @ YPred + 1e-8))


def cross_entropy_w_softmax_d(YPred, Y):
    return YPred - Y


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        y_hat_i = self.predict(x_i)

        if y_hat_i != y_i:
            self.W[y_i] += x_i
            self.W[y_hat_i] -= x_i


class LogisticRegression(LinearModel):

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        y_hat_i = softmax(np.dot(self.W, x_i.T))

        n_classes = self.W.shape[0]

        y_hot_i = np.zeros(n_classes)
        y_hot_i[y_i] = 1

        z_grad = y_hat_i - y_hot_i  # (n_classes)

        w_grad = np.outer(z_grad, x_i)

        self.W -= learning_rate * w_grad


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        units = [n_features, hidden_size, n_classes]
        self.W = [None] * len(units)
        self.B = [None] * len(units)

        self.W[0] = np.empty(0)
        for i in range(1, len(units)):
            self.W[i] = np.random.normal(loc=0.1, scale=0.1, size=(units[i], units[i - 1]))

        self.B[0] = np.empty(0)
        for i in range(1, len(units)):
            self.B[i] = np.zeros((units[i], 1))

    def forward(self, x_i):
        z = [None] * len(self.W)
        h = [None] * len(self.W)

        h[0] = x_i
        z[0] = x_i

        for i in range(1, len(self.W)):
            # W[i] -> (num_outputs_i, num_inputs_i)
            # H[i-1] -> (num_inputs_i, 1)
            z[i] = self.W[i] @ h[i - 1] + self.B[i]

            if i == len(self.W) - 1:
                h[i] = softmax(z[i])
            else:
                h[i] = relu(z[i])

        return z, h

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        predicted_labels = []

        for x_i in X:
            x_i = np.expand_dims(x_i, axis=1)

            _, h = self.forward(x_i)
            # h[-1] (n_classes)
            predicted_labels += [np.argmax(h[-1])]

        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def backward(self, y, z, h):
        z_grad = cross_entropy_w_softmax_d(h[-1], y)  # (num_outputs_i, 1)
        W_grad = [None] * len(self.W)  # (num_outputs)
        B_grad = [None] * len(self.W)

        for i in reversed(range(1, len(self.W))):
            W_grad[i] = z_grad @ h[i - 1].T

            B_grad[i] = z_grad
            h_grad = self.W[i].T @ z_grad
            z_grad = h_grad * relu_d(
                z[i - 1])  # z_grad '[0]' should use input linear layer derivative, but since it's never used, doesn't matter

        return W_grad, B_grad

    def optimize(self, W_Grad, B_Grad, eta):
        for i in range(1, len(self.W)):
            self.W[i] -= eta * W_Grad[i]
            self.B[i] -= eta * B_Grad[i]

    def train_epoch(self, X, y, learning_rate=0.001):
        total_loss = 0
        for x_i, y_i in zip(X, y):
            x_i = np.expand_dims(x_i, axis=1)
            y_hot_i = np.zeros((self.W[-1].shape[0], 1))
            y_hot_i[y_i] = 1

            z, h = self.forward(x_i)

            total_loss += cross_entropy(h[-1], y_hot_i).item()

            W_grad, B_grad = self.backward(y_hot_i, z, h)

            self.optimize(W_grad, B_grad, learning_rate)

        return total_loss


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()


def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []

    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )

        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
    ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
