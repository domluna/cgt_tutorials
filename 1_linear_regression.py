from __future__ import print_function
import cgt
import numpy as np

# training data
trX = np.linspace(-1, 1, 100)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

X = cgt.scalar('X')
Y = cgt.scalar('Y')
b = cgt.scalar('b')

# Linear regression model
def model(X, w, b):
    return X * w + b

w = cgt.shared(np.random.randn())
b = cgt.shared(np.random.randn())
y = model(X, w, b)

# learning rate
alpha = 0.001

# gradient descent
cost = cgt.mean(cgt.square(y - Y))
dw, db = cgt.grad(cost=cost, wrt=[w, b])
updates = [(w, w - dw * alpha), (b, b - db * alpha)]

train = cgt.function([X, Y], outputs=cost, updates=updates)

for i in xrange(100):
    for x,y in zip(trX, trY):
        train(x, y)

print(w.op.get_value(), b.op.get_value()) # ~2.0


