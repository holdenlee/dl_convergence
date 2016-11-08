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
import dl_feed
from load_mnist import *
import spams

# Apply different types of DL to images

placeholder_dict = {}

def dl_loss(x, A, h):
    x_t = tf.matmul(h,A)
    loss = l2(x - x_t)
    return loss

#batch * size
def l2(x, reduction_indices=[1]): 
    return tf.reduce_sum(tf.square(x), reduction_indices=reduction_indices)
#note this is squared.

#batch * size
def l1(h, reduction_indices=[1]):
    return tf.reduce_sum(tf.abs(h), reduction_indices=reduction_indices)
    #check that this maps correctly.

"""
def l1l2(h, r2=[-1], r1=[-1,-2]):
    return ll(l2(h, r2),r1)
"""

# Classic DL, Lee & Seung. Do AM with this. 
def dl_fs(x, A, h, la):
    loss = dl_loss(x,A,h) + la*l1(h)
    return {"loss":loss}

# here A is not flattened, f_h*f_w*c_in*c_out NO
# A is actually the opposite of convolution!
# I don't know how to do this!
# need to reverse A. Let's not reverse it now...
# x is batch*c_in
def dl2_loss(x, A, h):
    # h::batch*c_in
    # A::f_h*f_w*c_in*c_out
    # x_t:: f_h*f_w*c_out
    # A_shuffle :: c_in*f_h*f_w*c_out
    A_shuffle = tf.transpose(h, [2,0,1,3])
    # aargh!
    # https://github.com/tensorflow/tensorflow/issues/216
    # https://github.com/tensorflow/tensorflow/pull/4378
    dims2 = A_shuffle.get_shape().as_list()[1:3]
    A_reshape = tf.reshape(A_shuffle, [None,-1])
    x_t = tf.reshape(tf.matmul(x, A_reshape), [None]+dims2)
               #CHECK concat
    loss = l2(x-x_t,[-1,-2])
    return loss

def dl2_fs(x, A, h, la):
    loss = dl2_loss(x,A,h) + l1(l2(x - x_t, [-1]), [-1])
    return {"loss":loss}

def dl_fs(x, A, h, la):
    loss = dl_loss(x,A,h) + la*l1(h)
    return {"loss":loss}

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]
"""
def tmul(A,B):
    A_dims = A.get_shape().as_list()[:-2] #check indexing
    B_dims = B.get_shape().as_list()[2:]
    A_reshape = tf.reshape(A_shuffle, [None,-1])
    dims2 = A_shuffle.get_shape().as_list()[1:3]
    A_reshape = tf.reshape(A_shuffle, [None,-1])
    tf.matmul(x, A_reshape)
"""
"""
def tensormul(A,B,Ainds=[-1], Binds=[0]):
    A_left = diff(range(A.rank), Ainds)
    B_left = diff(range(B.rank), Binds)
    A_shuffle = 
    B.get_shape().as_list()
"""

#loss from non-smoothness of filter. Here A is 2-D.
def deriv_loss(A, d):
    #https://www.tensorflow.org/versions/r0.11/api_docs/python/constant_op.html#constant
    #[1,-1]
    #I think you need to pass in the dimension... annoying
    A_flat = tf.reshape(A, [1, None, None, -1])
    I = np.eye(d)
    horiz = tf.constant([[I, -I]])
    vert = tf.constant([[I], [-I]])
    #horiz = tf.constant([1, -1])
    #vert = tf.constant([1, -1], shape=[1,2])
    #do we want to include the edges? same = yes, valid = no
    hd = tf.nn.conv2d(A_flat, horiz, [1,1], 'SAME')
    hdn = l2(hd, [-1,-2,-3])
    vd = tf.nn.conv2d(A_flat, vert, [1,1], 'SAME')
    vdn = l2(hd, [-1,-2,-3])
    return hdn + vdn
#do I need to replicate the [1,1]?

# Convolutional DL. 
def conv_dl_loss(x, A, h):
    x_t = tf.nn.conv2d(h, A, [1,1], 'SAME')
    loss = l2(x - x_t, [1,2])
    return loss

def conv_dl_fs(x, A, h, la):
    loss = conv_dl_loss(x,A,h) + la*l1(h,[1,2])
    return {"loss":loss}

if __name__ == "__main__":
    images, labels = load_mnist()
    psize = 6 
    X = (get_all_patches_as_matrix(arr, psize, lambda x:x)[0:2000] - 177.5)/177.5
    param = { 'K' : 100, # learns a dictionary with 100 elements
          'lambda1' : 0.15, 'numThreads' : 4, 'batchsize' : 400,
          'iter' : 1000}

    ########## FIRST EXPERIMENT ###########
    tic = time.time()
    D = spams.trainDL(np.transpose(X),**param)
    tac = time.time()
    t = tac - tic
    print('time of computation for Dictionary Learning: %f' % t)

    A = np.transpose(D)
    save_dict_as_pics(A, psize, psize, unnorm_f= lambda li:(li - min(li))/((max(li)-min(li)))*255):


    ##param['approx'] = 0
    # save dictionnary as dict.png
    # _objective(X,D,param,'dict')

  


