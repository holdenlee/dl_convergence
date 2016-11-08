from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
#import tensorflow as tf
import math

#from nets import *
from random import * 
import utils
from PIL import Image
#import mnist

# [index, y, x]
# deal with depth=1 for now
def get_all_patches_as_matrix(arr, psize, norm_f = lambda x:x):
    dims = np.shape(arr)
    mat = np.zeros((dims[0] * (dims[1] - psize + 1) * (dims[2] - psize + 1), psize * psize))
    r = 0
    for i in range(dims[0]):
        for j in range(dims[1] - psize + 1):
            for k in range(dims[2] - psize + 1):
                mat[r] = norm_f(np.ndarray.flatten(arr[i,j:j+psize, k:k+psize]))
                r = r+1
    return mat

def save_dict_as_pics(A, x, y, unnorm_f= lambda x:x, filename = 'dict'):
    rows = np.shape(A)[0]
    Ar = np.reshape(A, (rows, x, y))
    for i in range(rows):
        img = Image.fromarray(np.ndarray.astype(unnorm_f(Ar[i]), np.dtype(np.uint8)), 'L')
        img.save('%s_%d.png' % (filename, i))

