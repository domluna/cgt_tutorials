from __future__ import print_function
import cgt
import numpy as np
from cgt import nn
from sklearn import datasets
from sklearn.cross_validation import train_test_split

boston = datasets.load_boston()
data = boston.data
targets = boston.target

nfeats = data.shape[1]

# scale data
# scaler = StandardScaler().fit(data, targets)
# scaled_data = scaler.transform(data, targets)

# split data
X_train, X_test, Y_train, Y_test = train_test_split(data, targets,
        test_size=.2, random_state=0) 

# hyperparams
#
# Be careful when setting alpha! If it's too large
# here the cost will blow up.
alpha = 1e-7
epochs = 100

# Linear regression model
np.random.seed(0)
X = cgt.matrix('X', fixed_shape=(None, nfeats)) 
Y = cgt.vector('Y')
w = cgt.shared(np.random.randn(nfeats) * 0.01)

# prediction
ypred = cgt.dot(X, w)

# cost
cost = cgt.square(Y - ypred).mean()

# derivative with respect to w
dw = cgt.grad(cost=cost, wrt=w)
updates = [(w, w - dw * alpha)]

# training function
trainf = cgt.function(inputs=[X, Y], outputs=[], updates=updates)
# cost function, no updates
costf = cgt.function(inputs=[X, Y], outputs=cost) 

for i in xrange(epochs):
    trainf(X_train, Y_train)
    C = costf(X_test, Y_test)
    print("epoch {} cost = {}".format(i+1, C))

wval = w.op.get_value()
print("Linear Regression ", wval)

# closed form solution
wclosed = np.linalg.lstsq(data, targets)[0] 
print("Closed form ", wclosed)

# Tests, linreg_err ~= closed_err
linreg_err = np.square(np.dot(X_test, wval) - Y_test).mean()
closed_err = np.square(np.dot(X_test, wclosed) - Y_test).mean()
print("Linear Regression error = ", linreg_err)
print("Closed Form error = ", closed_err)


