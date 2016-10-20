from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import math
from utils import *

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

def make_data_set(A,size,m,s,f):
    n = np.shape(A)[1]
    hs = np.zeros((size,m))
    for i in range(size):
        coords = sample_without_replacement(n,s)
        for j in coords:
            hs[i,j] = f()
    return np.dot(hs,A).astype(np.float32)
#(np.dot(hs,A).astype(np.float32), hs)

def make_data_set_pm1(A, size,m,s):
    return make_data_set(A,size,m,s,rand_pm1)

def get_batch(A,m,s, num_examples):
    return make_data_set_pm1(A,num_examples,m,s)

def init_close(A, sigma):
    dims = np.shape(A)
    R= 1/math.sqrt(dims[1]) * sigma * np.random.randn(*dims)
    return (A + R).astype(np.float32)

def am_step(xs, A, eta, th, verbosity=1):
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
        printv("Loss: %f" % loss, verbosity, 1)
    return A

def eval_step(xs, A, th, verbosity=1):
    hs = [[x if x>th or x<-th else 0 for x in li] for li in np.dot(xs,np.transpose(A))]
#map(lambda x: x if x>th or x<-th else 0, np.dot(xs,np.transpose(A)))
    diff = xs - np.dot(hs, A)
    loss = np.mean(np.sum(np.square(diff), 1))
    printv("Eval loss: %f" % loss, verbosity, 1)
    return loss

def train_dl(A, B, m, s,batch_size, steps, eta, th, eval_steps):
    xs = get_batch(A, m, s, batch_size)
    eval_step(xs, B, th)
    for i in range(1,steps+1):
        xs = get_batch(A, m, s, batch_size)
        B = am_step(xs, B, eta, th, 1 if i % eval_steps == 0 else 0)
        #print(B[0])
        if i % eval_steps == 0:
            xs = get_batch(A, m, s, batch_size)
            loss = eval_step(xs, B, th)
    return (B, loss)

def make_A(m,n,verbosity=1):
    A = 1/math.sqrt(n) * np.random.randn(m,n)
    A = [ai/np.linalg.norm(ai) for ai in A]
    printv("A:", verbosity,2)
    printv(A, verbosity, 2)
    printv("AA^T:", verbosity, 2)
    printv(np.dot(A,np.transpose(A)), verbosity, 2)
    return A

def am_dl(verbosity=1):
    f = open('am_dl_3_50_25.txt', 'w')
    s = 3
    m = 50 # hidden vector
    n = 25 # observed vector
    q = s/m
    printv((s,m,n,q),verbosity,1)
    max_steps = 2000
    eval_steps = 100      
    #alpha_list = [1e-2] #[1e-4, 1e-3, 1e-2]
    batch_size = 256 # \Om(m * s)
    th = 0.5
    eta = 0.1 # 0.1 * m/s
    A = make_A(m,n,verbosity)
    #[(lambda M: init_close(M,0), "0")]:
    for (init,st) in [(init_close(A, 0.05), "0.05"),
                      (init_close(A, 0.1), "0.1"),
                      (init_close(A, 0.2), "0.2"),
                      (init_close(A, 0.5), "0.5"),
                      (1/math.sqrt(n)*np.random.randn(m,n).astype(np.float32), "random"), 
                      (1/math.sqrt(n)*np.random.randn(2*m,n).astype(np.float32), "overrandom"), 
                      (make_data_set_pm1(A,m,m,s), "samples"),
                      (make_data_set_pm1(A,2*m,m,s), "oversamples")]:
        train_dl_and_eval(A, m, s, batch_size, max_steps, eval_steps, eta, f, init, st, th=0.5, verbosity=verbosity)
    f.close()

def train_dl_and_eval(A, m, s, batch_size, max_steps, eval_steps, eta, f=None, init=None, st="", th=0.5, verbosity=1):
    #for alpha in alpha_list:

#    printv((batch_size, max_steps, eta, st), verbosity,1)
#    f.write(str((batch_size, max_steps, eta, st)))
    printv(st, verbosity, 1)
    if f!=None:
        f.write(st)
        f.write("\n")

    A0 = init
    (B, loss) = train_dl(A, A0, m, s, batch_size, max_steps, eta, th, eval_steps)
    printv("A:", verbosity, 2)
    printv(A, verbosity, 2)
    printv("B:", verbosity, 2)
    printv(B, verbosity, 2)

    if f!=None:
        f.write("Loss:\n")
        f.write(str(loss))
        f.write("\n")

    (mins1, argmins1, closestRows1) = getClosestRows(A,B)
    printv("Distance from rows of A:", verbosity, 1)
    printv(mins1, verbosity, 1)
    if f!=None:
        f.write("Distance from rows of A:\n")
        f.write(str(mins1))
        f.write("\n")

    Bn = [bi/np.linalg.norm(bi) for bi in B]

    (mins2, argmins2, closestRows2) = getClosestRows(A,Bn)
    printv("Distance from rows of A (after normalization):", verbosity, 1)
    printv(mins2, verbosity, 1)
    if f!=None:
        f.write("Distance from rows of A (after normalization):\n")
        f.write(str(mins2))
        f.write("\n")

    (mins3, argmins3, closestRows3) = getClosestRows(Bn, A)
    printv("Distance from rows of B (after normalization):", verbosity, 1)
    printv(mins3, verbosity, 1)
    if f!=None:
        f.write("Distance from rows of B (after normalization):\n")
        f.write(str(mins3))
        f.write("\n")

    AB = np.dot(A,np.transpose(B))
    printv("AB^T", verbosity, 2)
    printv(AB, verbosity, 2)
    return (loss, B, Bn, mins1, mins2, mins3, AB)
    
"""
Is learned dictionary close to real dictionary?
"""

def getClosestRows(A,B):
    dists = [[np.linalg.norm(ai-bi) for bi in B] for ai in A]
    argmins = [np.argmin(row) for row in dists]
    mins = [np.min(row) for row in dists]
    closestRows = [B[i] for i in argmins]
    return (mins, argmins, closestRows)  

def am_dls(verbosity=1):
    max_steps = 2000
    eval_steps = 100      
    th = 0.5
    eta = 1 # 0.1 * m/s
    data = []
    batch_size = 256 # \Om(m * s)
    f = open('am_dls.txt', 'w')
    for m in [50*2**k for k in range(5)]:
        n = m/2
        for s in [2**k for k in range(int(math.ceil(math.log(m,2)))+1)]:
            q = s/m
            printv((s,m,n,q),verbosity,1)
            A = make_A(m,n,verbosity)
            for (init,st) in [(init_close(A, 0.05), "0.05"),
                      (init_close(A, 0.1), "0.1"),
                      (init_close(A, 0.2), "0.2"),
                      (init_close(A, 0.5), "0.5"),
                      (1/math.sqrt(n)*np.random.randn(m,n).astype(np.float32), "random"), 
                      (make_data_set_pm1(A,m,m,s), "samples"),
                      (make_data_set_pm1(A,2*m,m,s), "oversamples")]:
                (loss, B, Bn, mins1, mins2, mins3, AB) = \
                    train_dl_and_eval(A, m, s, batch_size, max_steps, eval_steps, eta, f, init, st, 
                                      th=0.5, verbosity=verbosity)
                data.append((m,n,s, loss, mins1, mins2, mins3, AB))
    f.close()

if __name__=="__main__":
    np.set_printoptions(threshold=np.inf,precision=2)
    am_dls(verbosity=2)
