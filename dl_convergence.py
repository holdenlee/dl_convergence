from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import math

#from nets import *
from random import * 

placeholder_dict = {}

"""
Sample generation
"""
def rand_pm1():
    if randint(0,1)==0:
        return -1
    else:
        return 1

def make_data_set(A,size,m,q,f):
    n = np.shape(A)[1]
    hs = np.zeros((size,m))
    for i in range(size):
        for j in range(n):
            if random() < q:
                hs[i,j] = f()
    return np.dot(hs,A).astype(np.float32)
#(np.dot(hs,A).astype(np.float32), hs)

def make_data_set_pm1(A, size,m,q):
    return make_data_set(A,size,m,q,rand_pm1)

def get_batch(A,m,q, num_examples):
    return make_data_set_pm1(A,num_examples,m,q)

def init_close(A, sigma):
    dims = np.shape(A)
    R= 1/math.sqrt(dims[1]) * sigma * np.random.randn(*dims)
    return (A + R).astype(np.float32)

def am_step(xs, A, eta, th, verbosity=0):
    #decoding
    hs = [[x if x>th or x<-th else 0 for x in li] for li in np.dot(xs,np.transpose(A))]
#map(lambda x: x if x>th or x<-th else 0, np.dot(xs,np.transpose(A)))
    diff = (np.dot(hs, A) - xs)
    grad = np.dot( np.transpose(hs), diff)/np.shape(xs)[0]
    #print(np.shape(A))
    #print(np.shape(grad))
    #print(grad)
    #print("am_step")
    #print(hs[0], diff[0], eta, grad[0])
    A = A - eta * grad
    if verbosity==1:
        loss = np.mean(np.sum(np.square(diff), 1))
        #np.sum(np.square(diff))
        print("Loss: %f" % loss)
    return A

def eval_step(xs, A, th):
    hs = [[x if x>th or x<-th else 0 for x in li] for li in np.dot(xs,np.transpose(A))]
#map(lambda x: x if x>th or x<-th else 0, np.dot(xs,np.transpose(A)))
    diff = xs - np.dot(hs, A)
    loss = np.mean(np.sum(np.square(diff), 1))
    print("Eval loss: %f" % loss)

def train_dl(A, B, m, q,batch_size, steps, eta, th, eval_steps):
    xs = get_batch(A, m, q, batch_size)
    eval_step(xs, B, th)
    for i in range(1,steps+1):
        xs = get_batch(A, m, q, batch_size)
        B = am_step(xs, B, eta, th, 1 if i % eval_steps == 0 else 0)
        #print(B[0])
        if i % eval_steps == 0:
            xs = get_batch(A, m, q, batch_size)
            eval_step(xs, B, th)
    return B

def am_dl():
    f = open('am_dl_3_50_25.txt', 'w')
    s = 3
    m = 50 # hidden vector
    n = 25 # observed vector
    q = s/m
    print(s,m,n,q)
    max_steps = 1000
    eval_steps = 100      
    alpha_list = [1e-2] #[1e-4, 1e-3, 1e-2]
    batch_size = 256 # \Om(m * s)
    th = 0.5
    eta = 0.1 # 0.1 * m/s
    A = 1/math.sqrt(n) * np.random.randn(m,n)
    A = [ai/np.linalg.norm(ai) for ai in A]
    print("A:")
    print(A)
    print("AA^T:")
    print(np.dot(A,np.transpose(A)))
    #[(lambda M: init_close(M,0), "0")]:
    for (init,st) in [(init_close(A, 0.05), "0.05"),
                      (init_close(A, 0.1), "0.1"),
                      (init_close(A, 0.2), "0.2"),
                      (init_close(A, 0.5), "0.5"),
                      (1/math.sqrt(n)*np.random.randn(m,n).astype(np.float32), "random"), 
                      (make_data_set_pm1(A,m,m,q), "samples"),
                      (make_data_set_pm1(A,2*m,m,q), "oversamples")]:
        for alpha in alpha_list:

            print(batch_size, max_steps, eta, st)
            f.write(str((batch_size, max_steps, eta, st)))
            f.write("\n")

            A0 = init
            B = train_dl(A, A0, m, q, batch_size, max_steps, eta, th, eval_steps)
            print("A:")
            print(A)
            print("B:")
            print(B)
            
            (mins, argmins, closestRows) = getClosestRows(A,B)
            print("Distance from rows of A:")
            print(mins)
            f.write("Distance from rows of A:\n")
            f.write(str(mins))
            f.write("\n")
            
            Bn = [bi/np.linalg.norm(bi) for bi in B]

            (mins, argmins, closestRows) = getClosestRows(A,Bn)
            print("Distance from rows of A (after normalization):")
            print(mins)
            f.write("Distance from rows of A (after normalization):\n")
            f.write(str(mins))
            f.write("\n")

            (mins, argmins, closestRows) = getClosestRows(Bn, A)
            print("Distance from rows of B (after normalization):")
            print(mins)
            f.write("Distance from rows of B (after normalization):\n")
            f.write(str(mins))
            f.write("\n")
            
            AB = np.dot(A,np.transpose(B))
            print("AB^T")
            print(AB)
    f.close()

"""
Is learned dictionary close to real dictionary?
"""

def getClosestRows(A,B):
    dists = [[np.linalg.norm(ai-bi) for bi in B] for ai in A]
    argmins = [np.argmin(row) for row in dists]
    mins = [np.min(row) for row in dists]
    closestRows = [B[i] for i in argmins]
    return (mins, argmins, closestRows)  

if __name__=="__main__":
    np.set_printoptions(threshold=np.inf)
    am_dl()
