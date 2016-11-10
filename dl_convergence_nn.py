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
from dl_convergence import *


""" Norms """
#x :: R^{batch * size}
def l2(x, reduction_indices=[1]): 
    return tf.reduce_sum(tf.square(x), reduction_indices=reduction_indices)
#note this is squared.

#h :: R^{batch * size}
def l1(h, reduction_indices=[1]):
    return tf.reduce_sum(tf.abs(h), reduction_indices=reduction_indices)

"""
Classical DL, Lee and Seung
"""
def dl_loss(x, A, h):
    x_t = tf.matmul(h,A)
    loss = tf.reduce_sum(l2(x - x_t))
    return loss

# Classic DL, Lee & Seung. Do AM with this. 
def _dl_fs(x, A, h, la):
    loss = dl_loss(x,A,h) + la*l1(h, [0,1])
    #tf.get_collection('A')
    #tf.add_to_collection('A', A)
    #tf.get_collection('h')
    #tf.add_to_collection('h', h)
    return {"loss":loss}

def dl_fs(A, la, batch_size):
    global placeholder_dict
    h = variable_on_cpu("h", shape=(batch_size,m), initializer = tf.truncated_normal_initializer(stddev=1/math.sqrt(m)*s/m))
    x = variable_on_cpu("x", shape=(batch_size,n), var_type="placeholder")
    placeholder_dict["x"] = x
    return _dl_fs(x, A, h, la)

def refresh_f(A,m,s,bf,settting=""):
    #(xs, ys) 
    if setting == "y":
        (xs,ys) = make_data_set_pm1(A,bf.num_examples,m,s,"y")
        bf.xs = xs
        bf.ys = ys
    else:
        xs = make_data_set_pm1(A,bf.num_examples,m,s)
        bf.xs = xs

"""
AM step
"""
"""
def am_step(fs, global_step):
    def _am_step(gs):
        tf.train.GradientDescentOptimizer(alpha).minimize(
            train_step(fs["loss"], fs["losses"], global_step, 
                             lambda gs: tf.train.GradientDescentOptimizer(alpha)))
"""

def main(argv=None):
    np.set_printoptions(threshold=np.inf)
    am_dl()

def am_dl():
    f = open('am_dl.txt', 'w')
    print(s,m,n,q)
    #batch_size = 32
    #train_size = 2**15
#    sigma = 0.05
    max_steps = 10000
    eval_steps = 10000      
    la = 0.01
    alpha_list = [1e-2] #[1e-4, 1e-3, 1e-2]
    #batch_list = [16, 32]
    #train_list = [2**10, 2**15]
    batch_size = 1024 #32
    train_size = 2**10 #actually, number of examples
    test_size = 2**10
    #[(lambda M: init_close(M,0), "0")]:
    for (init,st) in [(lambda M: init_close(M, 0.05), "0.05"),
                      (lambda M: init_close(M, 0.1), "0.1"),
                      (lambda M: init_close(M, 0.2), "0.2"),
                      (lambda M: init_close(M, 0.5), "0.5"),
                      (lambda M: 1/math.sqrt(n)*np.random.randn(m,n).astype(np.float32), "random"), 
                      (lambda M: make_data_set_pm1(A,m,m,q), "samples"),
                      (lambda M: make_data_set_pm1(A,2*m,m,q), "oversamples")]:
      for alpha in alpha_list:
        A = 1/math.sqrt(n) * np.random.randn(m,n)
        A = [ai/np.linalg.norm(ai) for ai in A]
#np.transpose([ai/np.linalg.norm(ai) for ai in np.transpose(A)])
        print(A)
        print(train_size, batch_size, max_steps, eval_steps, alpha, st)
        f.write(str((train_size, batch_size, max_steps, eval_steps, alpha, st)))
        f.write("\n")
        train_dir = "data/"
#        (xs,hs) = make_data_set_pm1(A,train_size,m,q)
        #              train_data = make_batch_feeder(xs, ys)
        train_data = make_batch_feeder_with_refresh(None, None, lambda bf: refresh_f(A, m,q,bf), train_size)
        test_data = make_batch_feeder_with_refresh(None, None, lambda bf: refresh_f(A, m,q,bf), test_size)
        with tf.Graph().as_default():
            with tf.variable_scope("def"):
                #, shape=(m,n) #do this for tensorflow 7 not 11.
                A0 = variable_on_cpu("A", initializer=init(A), dtype=tf.float32)
                      # x = tf.constant(xs)
                funcs = dl_fs(A0, la, batch_size)
            step = lambda fs, global_step: (
                train_step(fs["loss"], [], global_step,
                              lambda gs: tf.train.GradientDescentOptimizer(alpha)))
            sess=train2(funcs, 
                        step, 
                        max_steps=max_steps, 
                        eval_steps=0, #no evaluation
                        train_dir=train_dir,
                        batch_size=batch_size, #no batches
                        train_data=train_data,
                        test_data=test_data,
                        x_pl="def/x:0")
                        #fn = lambda sess: printAB(A, sess) )
#                        train_feed={"def/x:0": xs}) 
                  #  print(tf.get_collection(tf.GraphKeys.VARIABLES))
                  #  with tf.Graph().as_default():
            B = tf.get_default_graph().get_tensor_by_name("def/A:0").eval(session=sess)  
            #Bt = np.transpose(B)
            (mins, argmins, closestRows) = getClosestRows(A,B)
            print(A)
            print(B)
            print(mins)
            f.write(str(mins))
            f.write("\n")
            Bn = [bi/np.linalg.norm(bi) for bi in B]
            # np.transpose([bi/np.linalg.norm(bi) for bi in np.transpose(B)])

            (mins, argmins, closestRows) = getClosestRows(A,Bn)
            print(mins)
            f.write(str(mins))
            f.write("\n")
            #(mins, argmins, closestRows) = getClosestRows(Bn,A)
            #f .write(str(mins))
            #f.write("\n")
            #  print(A.eval())
            #  type(A.eval())
            AB = np.dot(A,np.transpose(B))
            print(np.shape(A))
            print(np.shape(B))
            print(np.shape(AB))
            print(AB)
#            ABt = np.less(0.5, np.dot
#            print(np.transpose(A)*A)
            print(np.dot(A,np.transpose(A)))
    f.close()

if __name__ == '__main__':
  #tf.get_variable_scope().reuse_variables()
  tf.app.run()
