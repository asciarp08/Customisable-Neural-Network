#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 19:45:22 2019

@author: anjali
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NN(object):
    """
    A network that uses Sigmoid activation function.
    """

    def __init__(self):
        
        self.nodes = []
        self.layers = {}
        self.weights = {}
        self.n_classes = 0
        
        # For Debugging
        self.grads = {}
        self.regs = {}
        self.dels = {}
        
        # For analysis
        self.history = {}

    def add_layer(self,node = []):
        self.nodes = node[:-1]
        self.n_classes = node[-1]

    def sigmoid(self, z):
        """
        Calculates the sigmoid activation function.
        """

        return 1 / (1 + np.exp(-z))

    def predict(self, x, predict=True, argmax=True, rand_weights=False):
        """
        Performs a forward and backward propagation.
        weights are to be initialised randomaly.
        if predict is set as true ,the trained weights are used
        and returns predictions as a single vector 
        (if argmax is set true)
        """
        
        np.random.seed(1)
        nodes = self.nodes
        layers = {}
        weights = {}

        #  input layer
        m = x.shape[0]
        x = np.append(np.ones(m).reshape(m, 1), x, axis=1)
        layers['a%d' % 1] = x
        
        #  dense layers
        if rand_weights:
            for i in range(len(nodes)):
                m, n = x.shape
                w = np.random.randn(nodes[i], n)
                z = x.dot(w.T)
                a = np.append(np.ones(m).reshape(m, 1), self.sigmoid(z), axis=1)

                if not predict:
                    layers['a%d' % (i + 2)] = a
                    weights['w%d' % (i + 1)] = w
                x = a

            # output layer
            m, n = x.shape
            w = np.random.randn(self.n_classes, n)
            z = x.dot(w.T)
            a = self.sigmoid(z)
            x = a

            if not predict:
                layers['a%d' % (len(layers) + 1)] = a
                weights['w%d' % (len(weights) + 1)] = w

        else:
            
            #  dense layers
            for i in range(len(nodes)):
                m, n = x.shape
                w = self.weights['w%d' % (i + 1)]
                z = x.dot(w.T)
                a = np.append(np.ones(m).reshape(m, 1), self.sigmoid(z), axis=1)
                if not predict:
                    layers['a%d' % (i + 2)] = a
                    weights['w%d' % (i + 1)] = w

                x = a

            # output layer
            w = self.weights['w%d' % (len(nodes) + 1)]
            z = x.dot(w.T)
            a = self.sigmoid(z)
            x = a

            if not predict:
                layers['a%d' % (len(layers) + 1)] = a
                weights['w%d' % (len(weights) + 1)] = w

        output = x
        if predict:
            return np.argmax(output, axis=1) if argmax else output
        elif rand_weights:
            self.layers = layers
            self.weights = weights
        else:
            return layers, weights

    def cost(self, x, y, lamda=0):
        """ Calculates the cost for the given data , using the trained weights."""
        
        m,n = x.shape
        weights = self.weights
        # _  for the throw away variable weights as weights have already been called.
        layers , _ = self.predict(x,predict = False)
        
        sum = np.sum
        log = np.log
        reg2 = 0
        for i in range(len(weights)):
            reg2 += sum(weights['w%d' % (i + 1)][:, 1:] ** 2)

        j = (-1 / m) * sum(y.T.dot(log(layers['a%d' % (len(layers))])) +
                              (1 - y).T.dot(log(1 - layers['a%d' % (len(layers))]))) + (lamda / (2 * m)) * reg2
        return j
    
    def fit(self, data, labels, test=[], test_labels=[], alpha=0.01, lamda=0, epochs=50):
        """
        Performs specified number of epochs. 
        One epoch = one pass of forward propagation + one pass of backpropagation. 
        """

        #  Training
        self.predict(data, predict=False, rand_weights=True)
        j_history = []
        j_test_history = []

        for epoch in range(epochs):
            layers, weights = self.predict(data, predict=False, rand_weights=False)
            m, n = data.shape

            #  Calculating del terms for outer layer
            delta1 = layers['a%d' % (len(layers))] - labels
            delta = delta1.dot(weights['w%d' % (len(weights))]) * layers['a%d' % (len(layers) - 1)] * \
                    (1 - layers['a%d' % (len(layers) - 1)])

            dels = {
                'del%d' % (len(layers)): delta1,
                'del%d' % (len(layers) - 1): delta
            }
            #for inner layers
            for i in range(len(weights) - 2):
                delta = delta[:, 1:].dot(weights['w%d' % (len(layers) - 2 - i)]) * \
                        layers['a%d' % (len(layers) - 2 - i)] * (1 - layers['a%d' % (len(layers) - 2 - i)])
                dels['del%d' % (len(layers) - 2 - i)] = delta

            #  Calculating grad and regularization terms
            grads = {
                'grad%d' % (len(weights)): (1 / m) * (
                    dels['del%d' % (len(layers))].T.dot(layers['a%d' % (len(weights))]))
            }

            regs = {
                'reg%d' % (len(weights)): (lamda / m) * weights['w%d' % (len(weights))]
            }

            for i in range(len(weights) - 1):
                grad = (1 / m) * \
                       (dels['del%d' % (len(layers) - 1 - i)][:, 1:].T.dot(layers['a%d' % (len(weights) - 1 - i)]))

                grads['grad%d' % (len(weights) - 1 - i)] = grad

                reg = (lamda / m) * weights['w%d' % (len(weights) - 1 - i)]
                reg[:, 0] = 0
                regs['reg%d' % (len(weights) - 1 - i)] = reg

            #  for debugging later on
            self.grads = grads
            self.regs = regs
            self.dels = dels

            #  Updating Parameters
            for i in range(1, len(weights) + 1):
                weights['w%d' % i] = weights['w%d' % i] - alpha * grads['grad%d' % i] - regs['reg%d' % i]

            self.layers = layers
            self.weights = weights

            #  Analysis steps

            j = float(self.cost(data, labels, lamda=lamda))
            j_history.append(j)

            print("\r" + "{}% |".format(int(100 * i / epochs) + 1) + '#' * int((int(100 * i / epochs) + 1) / 5) +
                  ' ' * (20 - int((int(100 * i / epochs) + 1) / 5)) + '|',
                  end="") if not i % (epochs / 100) else print("", end="")

            acc = 100 * np.sum(np.argmax(layers['a%d' % (len(layers))], axis=1) == np.argmax(labels, axis=1)) / m
            
            if len(test):
                m1, n1 = test.shape
                j_test = float(self.cost(test, test_labels, lamda=lamda))
                j_test_history.append(j_test)
                test_prediction = self.predict(test)
                acc_test = 100 * np.sum(test_prediction == np.argmax(test_labels, axis=1)) / m1
                print(acc_test, type(acc_test))
                print('Train cost: %0.2f\tTrain acc.: %0.2f%%\tTest cost: %0.2f\tTest acc.: %0.2f%%' % (j, acc, j_test, acc_test))
                
            else:
                print('cost: %0.2f\tacc.: %0.2f%%' % (j, acc))

        print("Displaying Cost vs Iterations graph...")
        
        if len(test):
            plot_test, =plt.plot(range(epochs), j_test_history, 'r')
        
        plot_train, =plt.plot(range(epochs), j_history, 'b')        
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        
        if len(test):
            plot_test, =plt.plot(range(epochs), j_test_history, 'r')
            plt.legend([plot_test, plot_train], ['Test', "Train"])
        else:
            plt.legend([plot_train], ["Train"])
        
        if len(test):
            print('Final train cost: %0.2f\tFinal train acc.: %0.2f%%\tFinal test cost: %0.2f\tFinal test acc.: %0.2f%%' % (j, acc, j_test, acc_test))
        else:
            print('Final cost: %0.2f\tFinal acc.: %0.2f%%' % (j, acc))

        self.history['run%d' % (len(self.history) + 1)] = {
            'layers': self.nodes,
            'alpha': alpha,
            'lambda': lamda,
            'epochs': epochs,
            'Final train acc.': acc,
            'Final train cost': j,
            'Final test acc.': acc_test if len(test) else 'N/A',
            'Final test cost': j_test if len(test) else 'N/A',
            
        }
        
#  Sample run

if __name__ == "__main__":
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [9, 8, 7], [6, 5, 4], [3, 2, 1], [7, 5, 3]])  # Test data
    y = np.array([[0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]])  # Test labels

    model = NN()
    model.add_layer([70,70,2])

    model.fit(x, y, epochs=1000)
