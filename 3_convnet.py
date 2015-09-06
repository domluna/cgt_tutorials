from __future__ import print_function, absolute_import
import cgt
from cgt import nn
from cgt.distributions import categorical
import numpy as np
from load import load_mnist
import time

epochs = 10
batch_size = 128

Xtrain, Xtest, ytrain, ytest = load_mnist(onehot=False)

# shuffle the data
np.random.seed(42)
sortinds = np.random.permutation(Xtrain.shape[0])
Xtrain = Xtrain[sortinds]
ytrain = ytrain[sortinds]

# reshape for convnet
Xtrainimg = Xtrain.reshape(-1, 1, 28, 28)
Xtestimg = Xtest.reshape(-1, 1, 28, 28)

# Model:
# Make it VGG-like
# VGG nets have 3x3 kernels with length 1 padding and max-pooling has all 2s.
#
# VGG is a large model so here well just do a small part of it.
X = cgt.tensor4('X', fixed_shape=(None, 1, 28, 28))
y = cgt.vector('y', dtype='i8')

conv1 = nn.rectify(
        nn.SpatialConvolution(1, 32, kernelshape=(3,3), stride=(1,1), pad=(1,1), weight_init=nn.IIDGaussian(std=.1))(X)
        )
pool1 = nn.max_pool_2d(conv1, kernelshape=(2,2), stride=(2,2))

conv2 = nn.rectify(
        nn.SpatialConvolution(32, 32, kernelshape=(3,3), stride=(1,1), pad=(1,1), weight_init=nn.IIDGaussian(std=.1))(pool1)
        )
pool2 = nn.max_pool_2d(conv2, kernelshape=(2,2), stride=(2,2))
d0, d1, d2, d3 = pool2.shape

flat = pool2.reshape([d0, d1*d2*d3])
nfeats = cgt.infer_shape(flat)[1]
probs = nn.softmax(nn.Affine(nfeats, 10)(flat))
cost = -categorical.loglik(y, probs).mean()

y_preds = cgt.argmax(probs, axis=1)
err = cgt.cast(cgt.not_equal(y, y_preds), cgt.floatX).mean()

params = nn.get_parameters(cost)
updates = nn.sgd(cost, params, 1e-3) 

# training function
f = cgt.function(inputs=[X, y], outputs=[], updates=updates)
# compute the cost and error
cost_and_err = cgt.function(inputs=[X, y], outputs=[cost, err])

for i in xrange(epochs):
    t0 = time.time()
    for start in xrange(0, Xtrain.shape[0], batch_size):
        end = batch_size + start
        f(Xtrainimg[start:end], ytrain[start:end])
    elapsed = time.time() - t0
    costval, errval = cost_and_err(Xtestimg, ytest)
    print("Epoch {} took {}, test cost = {}, test error = {}".format(i, elapsed, costval, errval))




