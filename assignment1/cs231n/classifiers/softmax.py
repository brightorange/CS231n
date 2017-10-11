import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores = X[i].dot(W)
    probs = np.exp(scores-np.max(scores)) / np.sum(np.exp(scores-np.max(scores)))
    correct_class_prob = probs[y[i]]
    loss += -np.log(correct_class_prob)

    for j in range(num_classes):
      dW[:, j] += (probs[j] - (j == y[i])) * X[i, :]

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]

  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  maxes = np.max(scores, axis=1)[:, np.newaxis]
  probs = np.exp(scores - maxes) / np.sum(np.exp(scores - maxes), axis=1)[:, np.newaxis]
  loss = np.sum(-np.log(probs[np.arange(num_train), y]))
  loss /= num_train
  loss += reg * np.sum(W * W)

  dscores = probs
  dscores[range(num_train), y] -= 1
  dscores /= num_train
  dW = np.dot(X.T, dscores)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

