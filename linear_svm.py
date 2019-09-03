from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero
    #print('W.shape: ',W.shape)
    #print('dW.shape: ',dW.shape)
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):#500장
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes): # 10개 class
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] +=X[i,:] # 다른 label에서의 W도함수
                dW[:,y[i]] -=X[i,:] # 맞는 label에서의 W도함수
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W) # 최종 loss 식 
    dW += reg*2*W # 최종 도함수 식
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W) # 500,10
    correct_class_score = scores[np.arange(scores.shape[0]),y] #500,10 중에 correct label 선택해서 저장
    margin = np.maximum(0,scores - np.matrix(correct_class_score).T +1)
    margin[np.arange(num_train),y]=0 #마진에서 crrect label에 0 저장
    loss = np.mean(np.sum(margin,axis =1))
    #print(type(W)) # matrix
    W = np.array(W)
    #print(type(W)) # array
    loss += reg *np.sum(W*W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    binary = margin
    binary[margin>0] =1 #binary에 마진이 0보다 크면 1로 저장
    row_sum = np.sum(binary,axis=1) # 행으로 다 더함(axis =1)
    binary[np.arange(num_train),y] = -np.matrix(row_sum).T # binary matrix의 correct label에 - row_sum 넣어줌(sum만큼 미분횟수 진행)
    dW = np.dot(X.T,binary) # X.transpose 와 binary matrix 곱 해줌 -> 도함수값
    dW = dW/num_train +2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW
