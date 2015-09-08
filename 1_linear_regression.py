from __future__ import print_function
import cgt
import numpy as np

np.random.seed(0)

# training data
Xtrain = np.linspace(-1, 1, 100)
ytrain = 2 * Xtrain + np.random.randn(*Xtrain.shape) * 0.33

X = cgt.scalar('X')
Y = cgt.scalar('Y')

# Linear regression model
def model(X, w):
    return X * w

w = cgt.shared(np.random.randn() * 0.01)
b = cgt.shared(1)
y = model(X, w)

# learning rate
alpha = 0.001
epochs = 100

# gradient descent
cost = cgt.mean(cgt.square(y - Y))
dw = cgt.grad(cost=cost, wrt=w)
updates = [(w, w - dw * alpha)]

train = cgt.function([X, Y], outputs=cost, updates=updates)

for i in xrange(epochs):
    for x,y in zip(Xtrain, ytrain):
        train(x, y)

print(w.op.get_value())

# closed form
A = np.vstack([Xtrain, np.ones(len(Xtrain))]).T
m,c = np.linalg.lstsq(A, ytrain)[0] # y = mx + b
print(m, c)
