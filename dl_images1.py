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
from dl_feed import *
from load_mnist import *
import spams


if __name__ == "__main__":
    images, labels = load_mnist()
    psize = 6 
    X = (get_all_patches_as_matrix(images, psize)[0:2000] - 177.5)/177.5
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
    save_dict_as_pics(A, psize, psize, unnorm_f= lambda li:(li - np.min(li))/((np.max(li)-np.min(li)))*255)


    ##param['approx'] = 0
    # save dictionnary as dict.png
    # _objective(X,D,param,'dict')

  


