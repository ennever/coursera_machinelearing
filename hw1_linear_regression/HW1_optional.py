# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 18:07:46 2015

@author: ennever
"""
import numpy as np
import matplotlib.pyplot as plot


ex1 = '/home/ennever/Projects/coursera_machinelearning/hw1_linear_regression/machine-learning-ex1/ex1/ex1data2.txt'
f = open(ex1)
alldata = ''
for line in f:
    alldata += line.strip('\n')
    alldata += ';'

alldata=alldata.strip(';')
data = np.matrix(alldata)

x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]
"""
plot.plot(x1, y, 'rx')
plot.ylabel('Profit in $s')
plot.xlabel('Size of House [square ft]')
plot.show()

plot.plot(x2, y, 'rx')
plot.ylabel('Profit in $s')
plot.xlabel('Number of Bedrooms')
plot.show()
"""

class feature:
    
    def __init__(self, x):
        self.x = x
        self.meanval = np.mean(x)
        self.stdval = np.std(x)
        self.normalized = False
        
    def normalize(self):
        if ~self.normalized:
            xnew = []
            for element in self.x:
                xnew.append((float(element) - self.meanval)/self.stdval)
            self.x = xnew
            self.normalized = True
        else:
            print("Already Normalized")
        
    
    def denormalize(self):
        if self.normalized:
            xnew = []
            for element in self.x:
                xnew.append((float(element)*self.stdval) + self.meanval)
            self.x = xnew
            self.normalized = False
        else:
            print("Not Normalized")

class features:
   
    def __init__(self):
        self.features = []
        self.n = 0 #number of features
        
    def add_feature(self, feature):
        self.features.append(feature)
        self.m = len(feature.x) #number of examples
        self.n += 1
    
    def compute_cost(self, y_data, theta):
        total = 0
        for mindex in range(self.m):
            y_test = theta[0]
            for nindex in range(self.n):
                feature = self.features[nindex].x
                y_test += float(theta[nindex+1]) * float(feature[mindex])
            total += (y_data[mindex] - y_test) ** 2
        return total/(2*self.m)
    
    def difference_total(self, y_data, theta):
        total = np.zeros(self.n + 1)
        for mindex in range(self.m):
            y_test = theta[0]
            
            for nindex in range(self.n):
                feature = self.features[nindex].x
                y_test += float(theta[nindex+1]) * float(feature[mindex])
            total[0] += (y_test - y_data[mindex])
            for nindex in range(self.n):
                feature = self.features[nindex].x
                total[nindex+1] += (y_test - y_data[mindex]) * (feature[mindex])
        total[0] = total[0]/self.m
        for nindex in range(self.n):
            total[nindex+1] = total[nindex+1]/self.m
        return total
     
     
    def gradient_descent(self, y_data, theta_guess = None, alpha = 0.01, its = 100, plot_cost = False):
        its_counter = 0
        theta = theta_guess or np.zeros(self.n+1) #make the default this way since it depends on self
        if plot_cost:
            cost_array = []
        while its_counter < its:
            its_counter += 1
            if plot_cost:
                cost_array.append(float(self.compute_cost(y_data, theta)))
            diff_totals = self.difference_total(y_data, theta)
            theta[0] -= float(diff_totals[0] * alpha)
            for nindex in range(self.n):
                theta[nindex+1] -= float(diff_totals[nindex+1] * alpha)
                
        if plot_cost:
            cost_array.append(float(self.compute_cost(y_data, theta)))
            plot.plot(range(its + 1), cost_array, 'g')
            plot.ylabel('Cost Function')
            plot.xlabel('Number of Itterations')
            plot.show()
            
        return theta  
        
    def plot
        
        
x1new = feature(x1)
x2new = feature(x2)
ynew = feature(y)

x1new.normalize()
x2new.normalize()
ynew.normalize()

x = features()
x.add_feature(x1new)
x.add_feature(x2new)

theta = x.gradient_descent(ynew.x, plot_cost=True, alpha = 0.1, its = 100)
