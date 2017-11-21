#!/usr/local/bin/python3
# preds = O^T * X
preds = X.dot(theta)
print ("preds:", preds)
return preds

# COST
m = len(y)
# cost = ( 1/ 2m   ) *    SUM[          ( h_theta(x^i)     - y^i) ^2 ]                          ]
cost = ( 1/(2*m) ) * np.sum(np.power( ( predict(X, theta) - y ), 2 ))
print ("cost", cost)
return cost

# GRAD 
m=len(y) 
# grad =   (1/m) * SUM [ h_theta(x^i)  - y^i ]* x^ij
grad = ( (1/m) * ((predict(X, theta) - y).dot(X)) )
print ("grad", grad)
return grad

#------------------------------------------
def addQuadraticFeature(X):
# Given feature vector [x_1,x_2] as input, extend this to
# [x_1,x_2,x_1*x_1] i.e. add a new quadratic feature
##### insert your code here #####
return np.append(X, np.power(X[:, [0]], 2), axis=1)

def computeScore(X,y,preds):
# for training data X,y it calculates the number of correct predictions made by the model
##### replace the next line with your code #####
score = len(y) - np.count_nonzero(y-preds)
return score

def predict(X,theta):
# calculates the prediction h_theta(x) for input(s) x contained in array X
##### replace the next line with your code #####
pred=np.sign(X.dot(theta))
return pred

def computeCost(X, y, theta):
# function calculates the cost J(theta) and returns its value
##### replace the next line with your code #####
# m = len(y)
cost = 1/(len(y)) * np.sum ( np.log ( 1 + np.exp( (-y) * (X.dot(theta))) ) )
return cost

def computeGradient(X,y,theta):
# calculate the gradient of J(theta) and return its value
##### replace the next lines with your code #####
m = len(y)
# tope = np.exp( (-y) * (X.dot(theta)))
# bote = 1 + np.exp( (-y) * (X.dot(theta)))
# tob = tope/bote
# yo = (-y) * tob
# xy = X.T.dot(yo)
grad = X.T.dot( (-y) * ((np.exp( (-y) * (X.dot(theta))))/(1 + np.exp( (-y) * (X.dot(theta)))))) / (len(y))
return grad
