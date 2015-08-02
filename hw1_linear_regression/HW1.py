# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 12:55:05 2015

@author: ennever
"""

import numpy as np
import matplotlib.pyplot as plot

ex1 = '/home/ennever/Projects/coursera_machinelearning/hw1_linear_regression/machine-learning-ex1/ex1/ex1data1.txt'
f = open(ex1)
alldata = ''
for line in f:
    alldata += line.strip('\n')
    alldata += ';'

alldata=alldata.strip(';')
data = np.matrix(alldata)

x = data[:,0]
y = data[:,1]


#cost function for gradient descent 
def cost_fcn(x_data, y_data, theta):
    total = 0
    m = len(y_data)
    for index in range(m):
        y_test = theta[0] + theta[1] * x_data[index]
        total += (y_data[index] - y_test) ** 2
    return total/(2*m)

print cost_fcn(x, y, [0,0])

def difference_total(x_data, y_data, theta):
    m = len(x_data)
    total = [0,0]
    for index in range(m):
        y_test = theta[0] + theta[1] * x_data[index]
        total[1] += (y_test - y_data[index]) * (x_data[index])
        total[0] += (y_test - y_data[index])
    total[0] = total[0]/m
    total[1] = total[1]/m
    return total

def gradient_descent(x_data, y_data, theta_guess = [0,0], its = 1000, alpha = 0.01, cost_array=cost_array):
    its_counter = 0
    theta = theta_guess
    while its_counter < its:
        its_counter += 1
        cost_array.append(float(cost_fcn(x, y, theta)))
        cost = difference_total(x_data, y_data, theta)
        theta[0] -= float(cost[0] * alpha)
        theta[1] -= float(cost[1] * alpha)
    
    return theta

 
#print gradient_descent(x, y, its = 10)  
y_test = []
cost_array = []
nits = 2000
alpha = 0.02
theta = gradient_descent(x, y, its = nits, cost_array=cost_array, alpha = alpha)  
m = len(x)
for index in range(m):
    y_test.append(theta[0] + theta[1] * float(x[index]))
plot.plot(x, y, 'rx')
plot.plot(x, y_test, 'b')
plot.ylabel('Profit in $10,000s')
plot.xlabel('Population of City in 10,000s')
plot.show()

plot.plot(range(nits), cost_array)
plot.xlabel('Itterations')
plot.ylabel('Cost Function')
plot.ylim(4, 5)
plot.show()
    
    


