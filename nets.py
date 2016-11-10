import tensorflow as tf
from functools import *
import itertools
import inspect
from datetime import datetime
import re
import os.path
import time
import numpy as np
from six.moves import xrange

def add_bias(x,b,name=None):
    return tf.add(x,b,name)

def linear_layer(x,W,b, name=None):
    return tf.add(tf.matmul(x, W), b, name)

def linear_layer_with_dims(m,n,scope=None,name=None):
    li = [InitVar("W", [m,n]),
          InitVar("b", [n]),
          Apply(linear_layer, {"name":name})]
    if scope==None:
        return Net(li)
        #return li
    else:
        return Net([Scope(scope, li)])
        #return Scope(scope,li)

#softmax over batches, l is the number in the batch. (For a single one, l=1.)
def softmax_layer(x,W,b, name=None):
    """R^{m*n}, R^n, R^{l*m} -> R^{l*n}"""
    return tf.nn.softmax(tf.matmul(x,W) + b, name)
#Alternatively use to add b:
#tf.nn.bias_add(value, bias, data_format=None, name=None)

def relu_layer(x,W,b,name=None):
    """R^{m*n}, R^n, R^{l*m} -> R^{l*n}"""
    return tf.nn.relu(tf.matmul(x,W) + b, name)

#TODO: abstract this
def relu_layer_with_dims(m,n,scope=None,name=None):
    li = [InitVar("W", [m,n]),
          InitVar("b", [n]),
          Apply(relu_layer, {"name":name})]
    if scope==None:
        return Net(li)
        # return li
    else:
        return Net([Scope(scope, li)])
        #return Scope(scope, li)

def cross_entropy(y, yhat):
    """R^{l*n}, R^{l*n} -> R"""
    return tf.reduce_mean(-tf.reduce_sum(y * tf.log(yhat), reduction_indices=[1]))
    #-sum(y *. log(yhat)) where sum is along first axis
    #now take mean

def correct_prediction(y, yhat):
    """R^{l*n}, R^{l*n} -> Bool^l"""
    return tf.equal(tf.argmax(yhat,1), tf.argmax(y,1))

def accuracy(y, yhat):
    """R^{l*n}, R^{l*n} -> R^l"""
    return tf.reduce_mean(tf.cast(correct_prediction(y,yhat), tf.float32))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, r, c):
  return tf.nn.max_pool(x, ksize=[1, r, c, 1],
                        strides=[1, r, c, 1], padding='SAME')

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'
def activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    x: Tensor (same thing)
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
  return x

"""
Example multi-layer
reduce(softmax_layer, [(W1,b1),(W2,b2),(W3,b3)], inp)
makes a 3-layer neural net.
"""

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z

"""
def compose(fs, var_dict):
    def f(x):
        scope = ""
        for entry in fs:
            if isinstance(entry, tuple):
                (f,args) = entry
                with tf.variable_scope(scope):
                    x = eval_with_missing_args(f, x, args, var_dict)
                    #tf.get_variable_scope().reuse_variables()
                    #f(x,**args)
            elif isinstance(entry, str):
                scope = entry
            else:
                entry(x) #side effect
        return x
    return f
"""
"""
def fold(f, args):
    return (lambda x: reduce(f,args,x))
"""
#compose(zip(len(args)*[f], args))
def fold(f, args, x):
    return reduce(lambda y, args: f(y,**args), x)

def weight_decay(var,wd,add=True):
    loss = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    if add:
        tf.add_to_collection('losses', loss)
    return loss

join = itertools.chain.from_iterable

def eval_with_missing_args(f,x,d,var_dict):
    t = inspect.getargspec(f)
    vars_ = t[0]
    specs = t[1:]
    #print("attempt eval")
    #print(vars_)
    #print(t)
    if t[-1]==None:
        l = 0
    else:
        l=len(t[-1])
    reqs = vars_[1:len(vars_)-l]
    #tf.get_variable_scope().reuse_variables()
    #http://stackoverflow.com/questions/6486450/python-compute-list-difference
    sc = tf.get_variable_scope().name
    d2=dict([(y, var_dict[sc+"/"+y]) for y in reqs if y not in d.keys()])
    #variable_on_cpu(y)
    d2.update(d)
    return f(x,**d2)

def mapkw(f, li):
    return [f(*l) for l in li]

def add_loss_summaries(total_loss, losses=[], decay=0.9):
  """Add summaries for losses in model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  # shadow_variable = decay * shadow_variable + (1 - decay) * variable
  loss_averages = tf.train.ExponentialMovingAverage(decay, name='avg')
  # losses = tf.get_collection('losses')
  # Instead pass in `losses` as an argument.
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))
  return loss_averages_op

def train_step(total_loss, losses, global_step, optimizer, 
               summary_f=None):
#lambda gs: tf.train.ExponentialMovingAverage(0.9999, gs)
  """Train model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # decayed_learning_rate = learning_rate *
  #                     decay_rate ^ (global_step / decay_steps)
  # Variables that affect learning rate.
  
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = add_loss_summaries(total_loss, losses)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #must compute loss_averages_op before executing this---Why?
    opt = optimizer(global_step)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  hist_trainables()

  # Add histograms for gradients.
  hist_grads(grads)

  deps = [apply_gradient_op]
  
  apply_summary_f(summary_f, global_step, deps)

  #lump them together
  return get_train_op(deps)

def hist_trainables():
  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

def hist_grads(grads):
  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

def apply_summary_f(summary_f, global_step, deps):
    if summary_f!=None:
        # Track the moving averages of all trainable variables.
        #print('Trainables')
        #for i in tf.trainable_variables():
        #    print(i, i.name)
        variable_averages = summary_f(global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        deps.append(variables_averages_op)

def get_train_op(deps):
    with tf.control_dependencies(deps):
        train_op = tf.no_op(name='train')
    return train_op

def valid_pos_int(n):
    return n!=None and n>0

def train2(funcs, step_f, output_steps=10, summary_steps=100, save_steps=1000, eval_steps = 1000, max_steps=1000000, train_dir="/", log_device_placement=False, batch_size=128,train_data=None,validation_data=None, test_data=None, train_feed={}, eval_feed={}, x_pl=None, y_pl=None, batch_feeder_args=[], fn=None):
    global_step = tf.Variable(0, trainable=False)
    loss = funcs["loss"]
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = step_f(funcs, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session() #config=tf.ConfigProto(
        #log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
    print("Initialized.")
    for step in xrange(max_steps):
      start_time = time.time()
      feed_dict = map_feed_dict(merge_two_dicts(
          fill_feed_dict(train_data, batch_size, 
                         x_pl,y_pl), train_feed))
#                     placeholder_dict["y_"]),
#      {"fc/keep_prob:0": 0.5})
#      if feed_dict_fun!=None:
#          feed_dict = map_feed_dict(feed_dict_fun())
      #print feed_dict
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
      duration = time.time() - start_time

      if np.isnan(loss_value):
          print('Model diverged with loss = NaN')
          return None
      if valid_pos_int(output_steps) and step % output_steps == 0:
         output_info(batch_size, step, loss_value, duration)
      if valid_pos_int(summary_steps) and step % summary_steps == 0:
        print("Running summary...")
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
      # Save the model checkpoint periodically.
      if (valid_pos_int(save_steps) and step % save_steps == 0) or (step + 1) == max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        print("Saving as %s" % checkpoint_path)
        saver.save(sess, checkpoint_path, global_step=step)
      if ((valid_pos_int(eval_steps) and (step + 1) % (eval_steps) == 0)) or (eval_steps!=None and (step + 1) == max_steps):
        for (data, name) in zip([train_data,validation_data,test_data], ["Training", "Validation", "Test"]):
            if data!=None:
                print('%s Data Eval:' % name)
                do_eval(sess,
                        funcs["accuracy"],
                        data,
                        batch_size,
                        x_pl,
                        y_pl,
                        batch_feeder_args,
                        eval_feed)
      if fn != None:
        fn(sess)
#    li = tf.get_collection(tf.GraphKeys.VARIABLES)
#    for i in li:
#        print(i.name)
#    print(tf.get_default_graph().get_tensor_by_name("layer1//W:0"))
    return sess
    

def train(fs, step_f, output_steps=10, summary_steps=100, save_steps=1000, eval_steps = 1000, max_steps=1000000, train_dir="/", log_device_placement=False, batch_size=128,train_data=None,validation_data=None, test_data=None, train_feed={}, eval_feed={}, x_pl=None, y_pl=None, batch_feeder_args=[]):
    """
    Train model.
  
    Args:
      fs: Inference function and loss function.
      step: Function to execute at each training step,
        takes arguments `loss` and `global_step`
    Returns:
      None.
    """
    #global counter
#   with tf.Graph().as_default():
    print("Building graph...")
    funcs = fs()
    return train2(funcs, step_f, output_steps, summary_steps, save_steps, eval_steps, max_steps, train_dir, log_device_placement, batch_size,train_data,validation_data, test_data, train_feed, eval_feed, x_pl, y_pl, batch_feeder_args)

def _conv2d_dims(inp, kern, filt, padding='SAME'):
    if padding == 'SAME':
        return ceil(float(inp)/float(strides))
    else:
        return ceiling(float(inp - kern + 1)/float(strides))

def get_conv2d_dims(input_dim, filter_dim, strides, padding='SAME'):
    in_height, in_width, in_channels = input_dim
    filter_height, filter_width, _, out_channels = filter_dim
    _, stride_height, stride_width, _ = strides
    return [_conv2d_dims(in_height, filter_height, stride_height),
            _conv2d_dims(in_width, filter_width, stride_width),
            out_channels]

def variable_on_cpu(name, shape=None, initializer=None,dtype=tf.float32, var_type="variable"):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
      if var_type == "variable":
          var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
      else: #"placeholder"
          #print(dtype, shape, name)
          var = tf.placeholder(dtype, shape=shape, name=name)
  return var

class Net:
    def __init__(self, li):
        self.inputs = [1]
        self.outputs = [1]
        self.d = {1:(li, [])}

class Scope:
    def __init__(self, s, li):
        self.scope = s
        self.li = li

class Apply:
    def __init__(self, f, args={}):
        self.f = f
        self.args = args

class Var:
    def __init__(self, name, shape=None, initializer=None, scope="", dtype=tf.float32, var_type="variable"):
        self.name=name
        self.shape=shape
        self.initializer=initializer
        self.scope=scope
        self.dtype=tf.float32
        self.var_type=var_type
    def get(self):
        return (self.name,self.shape,self.initializer,self.scope, self.dtype, self.var_type)
"""
class InitVar(Var):
    def __init__(self, name, shape=None, initializer=None, scope="", dtype=tf.float32):
        super(InitVar, self).__init__(name, shape, initializer, scope, dtype)
        self.var_type="var"

class Placeholder(Var):
    def __init__(self, name, shape=None, initializer=None, scope="", dtype=tf.float32):
        super(Placeholder, self).__init__(name, shape, initializer, scope, dtype)
        self.var_type="placeholder"
"""
def InitVar(name, shape=None, initializer=None, scope="", dtype=tf.float32):
    return Var(name, shape, initializer, scope, dtype, "variable")

def Placeholder(name, shape=None, scope="", dtype=tf.float32):
    return Var(name, shape, None, scope, dtype, "placeholder")

#don't use this
class InitVars:
    def __init__(self, li):
        self.li = li

def concat_scopes(s1, s2):
    if s2!="":
        if s1!="":
            return s1 + "/" + s2
        else: 
            return s2
    else:
        return s1

#d is actually unnecessary because you can get tensors by name.
def _compile_list(li, d, scope):
    def f(x,d):
        sc = scope
        for entry in li:
            if isinstance(entry, Scope):
                with tf.variable_scope(entry.scope):
                    g = _compile_list(entry.li, d, concat_scopes(sc, entry.scope))
                    x, d = g(x)
            elif isinstance(entry, Var):
                (name, shape, initializer, scope1, dtype, var_type) = entry.get()
                scope2 = concat_scopes(sc, scope1)
                with tf.variable_scope(scope1):
                    n = concat_scopes(scope2, name)
                    #print(n, shape)
                    if var_type=="variable":
                        d[n] = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
                    else:
                        d[n] = tf.placeholder(dtype, shape=shape, name=name)
#                    d[n] = variable_on_cpu(name, shape, initializer=initializer, dtype=dtype, var_type=var_type)
                    #print(d[n])
                    #print(tf.get_default_graph().get_tensor_by_name(n+":0"))
            elif isinstance(entry, InitVars):
                for (name, shape, initializer, scope1) in entry.li:
                    scope2 = concat_scopes(sc, scope1)
                    #same as above.
                    with tf.variable_scope(scope1):
                        n = concat_scopes(scope2, name)
                        d[n] = variable_on_cpu(name, shape, initializer=initializer)
            elif isinstance(entry, Apply):
                f = entry.f
                args = entry.args
                #print(scope)
                #print(type(scope))
                x = eval_with_missing_args(f, x, args, d)
                #tf.get_variable_scope().reuse_variables()
                #f(x,**args)
            elif isinstance(entry, Net):
                g = compile_net_with_vars(entry, d, sc)
                x, d = g(x)
        return x,d
    return (lambda x : f(x,d))

def compile_net_with_vars(net, d={}, scope=""):
    #fix this
    return _compile_list(net.d[1][0], d, scope)

def compile_net(net, d={},scope=""):
    return (lambda x: compile_net_with_vars(net, d, scope)(x)[0])

class BatchFeeder(object):

  def __init__(self, xs, ys, num_examples, next_batch_fun):
      self.xs = xs
      self.ys = ys
      self.index = 0
      self.num_examples = num_examples
      self.epochs_completed = 0
      self.next_batch_fun = next_batch_fun

  def next_batch(self, batch_size, *args):
      return self.next_batch_fun(self, batch_size, *args)

def fill_feed_dict(batch_feeder, batch_size, x_pl, y_pl=None, args = []):
  """Fills the feed_dict for training the given step. Args should be a list.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }"""
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  if x_pl == None:
      return {}
  if y_pl == None:
      x_feed = batch_feeder.next_batch(batch_size, *args)
      feed_dict = {x_pl: x_feed}
  else:
      x_feed, y_feed = batch_feeder.next_batch(batch_size, *args)
      feed_dict = {
          x_pl: x_feed,
          y_pl: y_feed,
      }
  return feed_dict

def do_eval(sess,
            eval_correct,
            batch_feeder, 
            batch_size,
            xs_placeholder,
            ys_placeholder=None,
            args=[],
            eval_feed={}):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    xs_placeholder: The images placeholder.
    ys_placeholder: The labels placeholder.
    batch_feeder: The set of xs and ys to evaluate.
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = batch_feeder.num_examples // batch_size
  num_examples = steps_per_epoch * batch_size
  #print(batch_size)
  #print(steps_per_epoch)
  for step in xrange(steps_per_epoch):
    #if x_pl==None:
    #    feed_dict = {}
    #else: 
    feed_dict = fill_feed_dict(batch_feeder,
                                   batch_size,
                                   xs_placeholder,
                                   ys_placeholder, args)
    feed_dict.update(map_feed_dict(eval_feed))
    x = sess.run(eval_correct, feed_dict=feed_dict)
    #print(feed_dict["x:0"])
    #print(len(feed_dict["x:0"]))
    #print(len(feed_dict["x:0"][0]))
    #print(x)
    true_count += x
  precision = float(true_count) / float(num_examples)
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def map_keys(f, d):
    return {f(k): v for (k,v) in d.items()}

def map_feed_dict(feed_dict):
    #print(feed_dict)
    return map_keys(lambda x: tf.get_default_graph().get_tensor_by_name(x) if isinstance(x,str) else x, feed_dict)
#http://stackoverflow.com/questions/394809/does-python-have-a-ternary-conditional-operator
#http://stackoverflow.com/questions/644178/how-do-i-re-map-python-dict-keys

def batch_feeder_f(bf, batch_size):
    start = bf.index
    bf.index += batch_size
    # for simplicity, discard those at end.
    if bf.index > bf.num_examples:
      # Finished epoch
      bf.epochs_completed += 1
      # Shuffle the data
      perm = np.arange(bf.num_examples)
      np.random.shuffle(perm)
      bf.xs = bf.xs[perm]
      if bf.ys != None:
          bf.ys = bf.ys[perm]
      # Start next epoch
      start = 0
      bf.index = batch_size
      assert batch_size <= bf.num_examples
    end = bf.index
    if bf.ys!=None:
        return bf.xs[start:end], bf.ys[start:end]
    else:
        return bf.xs[start:end]

def batch_feeder_f_with_refresh(bf, batch_size, refresh_f):
    if bf.xs == None:
        refresh_f(bf)
    start = bf.index
    bf.index += batch_size
    # for simplicity, discard those at end.
    if bf.index > bf.num_examples:
      # Finished epoch
      bf.epochs_completed += 1
      # Shuffle the data
      refresh_f(bf)
      # Start next epoch
      start = 0
      bf.index = batch_size
      assert batch_size <= bf.num_examples
    end = bf.index
    if bf.ys!=None:
        return bf.xs[start:end], bf.ys[start:end]
    else:
        return bf.xs[start:end]

def make_batch_feeder(xs, ys):
    return BatchFeeder(xs, ys, len(xs), batch_feeder_f)

def make_batch_feeder_with_refresh(xs, ys, refresh_f, num_examples = None):
    if num_examples==None:
        l = len(xs)
    else:
        l = num_examples
    return BatchFeeder(xs, ys, l, (lambda bf, batch_size: batch_feeder_f_with_refresh(bf, batch_size, refresh_f)))
    
"""
based off code at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
"""

def am_train_step(total_loss, losses, global_step, optimizer, c1, c2,
               summary_f=None):
    loss_averages_op = add_loss_summaries(total_loss, losses)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        #must compute loss_averages_op before executing this---Why?
        opt = optimizer(global_step)
        grads = opt.compute_gradients(total_loss, var_list = c1 if global_step % 2 == 0 else c2)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    hist_trainables()

    # Add histograms for gradients.
    hist_grads(grads)

    deps = [apply_gradient_op]
  
    apply_summary_f(summary_f, global_step, deps)

    #lump them together
    return get_train_op(deps)


def reuse_var(scope, var):
    with tf.variable_scope(scope, reuse=True):
    #scope.reuse_variables()
        return tf.get_variable(var)

      
def output_info(batch_size, step, loss_value, duration):
    examples_per_sec = batch_size / duration
    sec_per_batch = float(duration)
    format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                  'sec/batch)')
    print (format_str % (datetime.now(), step, loss_value,
                         examples_per_sec, sec_per_batch))

def is_gtr(x, c):
    return (tf.sign(x-c) + 1)/2

def is_less(x,c):
    return (1-tf.sign(x-c))/2
