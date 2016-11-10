import tensorflow as tf
from nets import *

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
    #global placeholder_dict
    h = variable_on_cpu("h", shape=(batch_size,m), initializer = tf.truncated_normal_initializer(stddev=1/math.sqrt(m)*s/m))
    x = variable_on_cpu("x", shape=(batch_size,n), var_type="placeholder")
    #placeholder_dict["x"] = x
    return _dl_fs(x, A, h, la)

def nndl_loss(x, A, y):
    h = tf.matmul(x, A)
    #print("nndl_loss")
    #print(y)
    yh = tf.reduce_sum(h, reduction_indices=[1])
    #print(yh)
    return l2(yh - y, [0])

def nndl_fs(A, n, batch_size):
    #global placeholder_dict
    x = variable_on_cpu("x", shape=(batch_size,n), var_type="placeholder")
    y = variable_on_cpu("y", shape=batch_size, var_type="placeholder")
    #placeholder_dict["x"] = x
    return {"loss" : nndl_loss(x, A, y), "accuracy" : nndl_loss(x,A,y)}
