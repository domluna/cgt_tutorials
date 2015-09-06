from __future__ import print_function, absolute_import
import cgt
from cgt import nn
from cgt.distributions import categorical
import numpy as np
from load import load_mnist
import time

epochs = 50
batch_size = 128
learning_rate = 1e-3

Xtrain, Xtest, ytrain, ytest = load_mnist(onehot=False)

# shuffle the data
np.random.seed(42)
sortinds = np.random.permutation(Xtrain.shape[0])
Xtrain = Xtrain[sortinds]
ytrain = ytrain[sortinds]

# Model:
# Two linear/affine layers with a ReLU activation in between
# followed by a logsoftmax.
X = cgt.matrix('X', fixed_shape=(None, 784))
y = cgt.vector('y', dtype='i8')

layer1 = nn.Affine(784, 400, weight_init=nn.XavierUniform(np.sqrt(2)))(X)
act1 = nn.rectify(layer1)
layer2 = nn.Affine(400, 10, weight_init=nn.XavierUniform(np.sqrt(2)))(act1)
act2 = nn.rectify(layer2)
probs = nn.softmax(act2)

y_preds = cgt.argmax(probs, axis=1)
cost = -cgt.mean(categorical.loglik(y, probs))
err = cgt.cast(cgt.not_equal(y, y_preds), cgt.floatX).mean()

params = nn.get_parameters(cost)
updates = nn.sgd(cost, params, learning_rate) # train via sgd

# training function
f = cgt.function(inputs=[X, y], outputs=[], updates=updates)
# compute the cost and error
cost_and_err = cgt.function(inputs=[X, y], outputs=[cost, err])

for i in xrange(epochs):
    t0 = time.time()
    for start in xrange(0, Xtrain.shape[0], batch_size):
        end = batch_size + start
        f(Xtrain[start:end], ytrain[start:end])
    elapsed = time.time() - t0
    costval, errval = cost_and_err(Xtest, ytest)
    print("Epoch {} took {}, test cost = {}, test error = {}".format(i, elapsed, costval, errval))




