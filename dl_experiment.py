from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math

from nets import *
from random import * 

placeholder_dict = {}

"""
s = 10
m = 1000 # hidden vector
n = 500 # observed vector
q = s/m
"""
s = 3
m = 50
n = 25
q = s/m

# The neural net that learns the dictionary
model = compile_net(Net(
    [Scope("layer1",[
        InitVar("W", [n,m], tf.truncated_normal_initializer(stddev=1/n)),
        InitVar("b", [m]),
        Apply(relu_layer)]),
     Scope("layer2",[
         InitVar("W", [m,1], tf.truncated_normal_initializer(stddev=1/math.sqrt(m))),
        InitVar("b", [1]),
        Apply(linear_layer)])]))
"""
compile_net(Net(
    [relu_layer_with_dims(n, m, scope="layer1"),
     linear_layer_with_dims(m, 1, scope="layer2")]))
"""

def _dl_fs(x, y_):
    """
    net = Net(
        [Apply(relu_layer_with_dims, {"m":n, "n":m, "scope":"layer1"}),
         Apply(linear_layer_with_dims, {"m":m, "n":1, "scope":"layer2"})])
    print(type(net))
    print(net.__class__.__name__)
    ans = model(x)
    print(type(ans))
    print(ans.__class__.__name__)
    print(type(y_))"""
    ans = model(x)
    loss = tf.reduce_mean(tf.maximum(0.0, 1 - y_*ans))
    pred = tf.sign(ans)
    # accuracy = tf.reduce_sum((tf.sign(ans)*y_ + 1)/2)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(pred, y_), tf.float32))
    return {"inference":ans, "loss":loss, "accuracy":accuracy, "losses":[]}  

def dl_fs(batch_size):
    global placeholder_dict
    x = variable_on_cpu("x", shape=(batch_size,n), var_type="placeholder")
    y_ = variable_on_cpu("y_", shape=(batch_size), var_type="placeholder")
    placeholder_dict["x"] = x
    placeholder_dict["y_"] = y_
    return _dl_fs(x,y_)

def rand_pm1():
    if randint(0,1)==0:
        return -1
    else:
        return 1

def make_data_set(A,size,f):
    xs = np.zeros((size,m))
    ys = np.zeros(size)
    for i in range(size):
        for j in range(n):
            if random() < q:
                xs[i,j] = f()
        if np.sign(np.sum(xs[:,j]))>=0:
            ys[i]=1
        else:
            ys[i]=-1
    return (np.dot(xs,A), ys)
#generate random...

def make_data_set_pm1(A, size):
    return make_data_set(A,size,rand_pm1)

def refresh_f(A,bf):
    (xs, ys) = make_data_set_pm1(A,bf.num_examples)
    bf.xs = xs
    bf.ys = ys

def init_close(A, sigma):
    dims = A.shape
    R= 1/math.sqrt(dims[0]) * sigma * np.random.randn(*dims)
    return A + R

def main(argv=None):
  print(s,m,n,q)
  #batch_size = 32
  #train_size = 2**15
  max_steps = 10000
  eval_steps = 10000      
  test_size = 1024
  for alpha in [1e-6, 1e-5, 1e-4]:
      for batch_size in [16, 32]:
          for train_size in [2**10, 2**15]:
              # initialize A randomly
              A = 1/math.sqrt(m) * np.random.randn(m,n)
              print(A)
              print(batch_size, train_size, max_steps, eval_steps, alpha)
              train_dir = "data/"
              (xs,ys) = make_data_set_pm1(A,train_size)
              train_data = make_batch_feeder(xs, ys)
              #  train_data = make_batch_feeder_with_refresh(None, None, lambda bf: refresh_f(A, bf), train_size)
              test_data = make_batch_feeder_with_refresh(None, None, lambda bf: refresh_f(A, bf), test_size)
              step = lambda fs, global_step: (
                  train_step(fs["loss"], fs["losses"], global_step, 
                             lambda gs: tf.train.GradientDescentOptimizer(alpha)))
              #print(tf.get_collection(tf.GraphKeys.VARIABLES))
              with tf.Graph().as_default():
                  sess=train(lambda: dl_fs(batch_size), 
                             step, 
                             max_steps=max_steps, 
                             eval_steps=eval_steps,
                             train_dir=train_dir,
                             batch_size=batch_size,
                             train_data=train_data,
                             # validation_data=data_sets.validation,
                             test_data=test_data,
                             x_pl = "x:0",
                             y_pl = "y_:0") 
                  #  print(tf.get_collection(tf.GraphKeys.VARIABLES))
                  #  with tf.Graph().as_default():
                  B = tf.get_default_graph().get_tensor_by_name("layer1//W:0").eval(session=sess)  
                  Bt = np.transpose(B)
                  (mins, argmins, closestRows) = getClosestRows(A,Bt)
                  print(mins)
                  Btn = [bi/np.linalg.norm(bi) for bi in Bt]
                  (mins, argmins, closestRows) = getClosestRows(A,Btn)
                  print(mins)
                  #  print(A.eval())
                  #  type(A.eval())

def getClosestRows(A,B):
    dists = [[np.linalg.norm(ai-bi) for bi in B] for ai in A]
    argmins = [np.argmin(row) for row in dists]
    mins = [np.min(row) for row in dists]
    closestRows = [B[i] for i in argmins]
    return (mins, argmins, closestRows)  

if __name__ == '__main__':
  #tf.get_variable_scope().reuse_variables()
  tf.app.run()
