# -*- coding: utf-8 -*-
from itertools import islice
from sklearn.metrics import log_loss,roc_auc_score
from scipy.stats import logistic
import sys
import copy
import random
#import numpy as np
from collections import defaultdict
import numpy as np, os, re, itertools, math
#from tensorflow_addons.optimizers import AdamW
import pandas as pd
import pickle
import math

def sigmoid(x):
  res=[]
  for i in range(0,len(x)):
     res.append(1 / (1 + math.exp(-x[i])))
  return res

def evaluate(model, input_data_x, input_data_y ,args, sess):
  score=[]
  label=[]
  l=[]
  #test_batch = 1
  num_test_batches = int (len(input_data_x)/ args.batch_size)
  step = 0
  #print('Number of test batches....', num_test_batches)

  while step < len(input_data_x) :
  #while step < 10 :
      if step+args.batch_size <  len(input_data_x):
        ids = np.arange(step, step+ args.batch_size, 1)
        step+=args.batch_size
        #step+=test_batch
      else:
        ids = np.arange(step, len(input_data_x), 1)
        #step+=args.batch_size
        step+=args.batch_size

      input_attributes_all = [] # selecting article from all packages for all users in batch
      target_labels_all =[]
      
      for id in ids:
          input_attr = [] # selecting article from all packages for all users in batch
          target_labels=[]
          # =======================================preparing targets================================
          input_attr.append(input_data_x[id])
          target_labels.append(input_data_y[id])

          input_attributes_all.append(input_attr)
          target_labels_all.append(target_labels)

      sco, lab, loss = model.predict(sess, input_attributes_all ,target_labels_all )
      #print('Score before appending..',sco)
      sco= sco.tolist()
      lab= lab.tolist()
      flat_sco = [item for sublist in sco for item in sublist]
      flat_lab = [item for sublist in lab for item in sublist]
      score.extend(flat_sco)
      #print('Score After appending..',score)
      label.extend(flat_lab)
      #print(loss)
      l.append(loss)

      #auc_all= auc_all+ auc

  #print('Label:: ',len(label))
  #print('Score:: ',score[0])
  auc = roc_auc_score(label, score)
  Final_loss = np.mean(l)
  #print('Sklearn logloss...', label,'        ', score)
  print('Model_loss..', Final_loss)
  score= sigmoid(score)
  logloss= log_loss(label, score)

  #print('Sklearn logloss...', log_loss(label, score))
  return auc, logloss

"""# **Hierarchical Sampler**"""

import numpy as np
from multiprocessing import Process, Queue

import sys


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(input_data_X, input_data_Y, params, result_queue, SEED):
    def sample():
        #print('Preparing...')
        id = np.random.randint(0, len(input_data_X) )
        #input_user= tf.expand_dims(input_user, 1) # reshape it to be (None, Number_of_user_packs))
        #print('input_user.......',input_user)
        input_attributes = [] # selecting article from all packages for all users in batch
        target_labels=[]
        
        input_attributes.append(input_data_X[id])
        # =======================================preparing targets================================
        target_labels.append(input_data_Y[id])

        return (input_attributes ,target_labels)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(params.batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, input_data_X, input_data_Y, params, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(
                                                      input_data_X,
                                                      input_data_Y,
                                                      params,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

"""#Sampler

#Modules
"""

import tensorflow as tf
import numpy as np


def positional_encoding(dim, sentence_length, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(value=encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(x=inputs, axes=[-1], keepdims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs

def embedding(inputs,
              vocab_size,
              num_units,
              Name,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None
              ):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        lookup_table = tf.compat.v1.get_variable('lookup_table'+Name,
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       #initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.keras.regularizers.l2(0.5 * (l2_reg)))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(params=lookup_table, ids=inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t: return outputs,lookup_table
    else: return outputs




def multihead_attention_Item(queries,
                        keys,
                        resweight_MultiHead,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        res=False,
                        with_qk=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].  4d tensor [Batches, baskets, items_in_the_basket, item embeddings] on normalized seq
      keys: A 3d tensor with shape of [N, T_k, C_k].     4d tensor [Batches, baskets, items_in_the_basket, item embeddings]
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)   4d tensor (Batches, baskets, items_in_the_basket, C)
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.compat.v1.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, B, T_q, C)
        K = tf.compat.v1.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, B, T_k, C)
        V = tf.compat.v1.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, B, T_k, C)
        #Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
        #K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        #V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=3), axis=0) # (h*N, B, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=3), axis=0) # (h*N, B, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=3), axis=0) # (h*N, B, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(a=K_, perm=[0, 1, 3, 2])) # (h*N, T_q, B, T_k)
        #outputs= tf.tensordot(Q_, K_, axes=3)
        #outputs = Q_ * K_
        #print('Now The outputs are ======>', outputs)
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(input_tensor=tf.abs(keys), axis=-1)) # (N, B, T_k)
        #print('Now key_masks ======>', key_masks)
        key_masks = tf.tile(key_masks, [num_heads,1 ,1]) # (h*N, B, T_k) repeat on the second axis instead of the first
        #print('Now key_masks 2 ======>', key_masks)
        key_masks = tf.tile(tf.expand_dims(key_masks, -1), [1, 1, 1,tf.shape(input=queries)[2]]) # (h*N, B, T_q, T_k)
        #print('KEY MASKS.............................', key_masks)
        paddings = tf.ones_like(outputs)*(-2**32+1)
        #print('PADDING...', paddings)
        #print('outputs...', outputs)
        outputs = tf.compat.v1.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(input=outputs)[0], 1 ,1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.compat.v1.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(input_tensor=tf.abs(queries), axis=-1)) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, 1, tf.shape(input=keys)[2]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(value=is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=3 ) # (N, T_q, C)

        # Alpha Zero Residual connection
        if res:
          #outputs *= queries
          outputs = outputs * resweight_MultiHead
          outputs = outputs * queries 
        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)

    if with_qk: return Q,K
    else: return outputs



def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 4d tensor with the same shape and dtype as inputs
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Inner layer
        #print('INPUT for the feed forward------------------>>>>', inputs)
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": [1,1],
                  "activation": tf.nn.leaky_relu, "use_bias": True}
        outputs = tf.compat.v1.layers.conv2d(**params)
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(value=is_training))
        #print('OUTPUT of the first CNN----------------->>>>', outputs)
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": [1,1],
                  "activation": None, "use_bias": True}
        outputs = tf.compat.v1.layers.conv2d(**params)
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(value=is_training))
        #print('OUTPUT of the second CNN----------------->>>>', outputs)
        # Residual connection
        outputs += inputs

        # Normalize
        #outputs = normalize(outputs)

    return outputs

"""#Model"""

class Model():

    #def __init__(self, usernum, itemnum, args, ItemFeatures=None, UserFeatures=None, cxt_size=None, reuse=None , use_res=False):
    def __init__(self, itemnum, args, reuse=None , use_res=False):
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())

        self.input_Attributes = tf.compat.v1.placeholder(tf.int32, shape=( None, 1, args.num_Fields))

        self.Labels = tf.compat.v1.placeholder(tf.int32, shape=( None,1 ))
        self.beta = tf.Variable(0.5, trainable=True)
        self.resweight_DNN_1 = tf.Variable(0.0, trainable=True)
        self.resweight_DNN_2 = tf.Variable(0.0, trainable=True)
        self.resweight_DNN_3 = tf.Variable(0.0, trainable=True)
        self.resweight_DNN_4 = tf.Variable(0.0, trainable=True)
        self.resweight_DNN_5 = tf.Variable(0.0, trainable=True)

        mask = tf.expand_dims(tf.cast(tf.not_equal(self.input_Attributes, 0), dtype=tf.float32), -1)

        #with tf.variable_scope("SASRec", reuse=reuse):
        # sequence embedding, item embedding table
        self.input_Attributes_in, Attr_emb_table = embedding(self.input_Attributes,
                                              vocab_size= itemnum ,
                                              Name='att',
                                              num_units=args.hidden_units,
                                              zero_pad=True,
                                              scale=True,
                                              l2_reg=args.l2_emb,
                                              scope="att_input_embeddings",
                                              with_t=True,
                                              reuse=reuse
                                              )
        
        t1=tf.expand_dims(tf.range(tf.shape(self.input_Attributes_in)[2]), 0) # -> (1,Max_instance_Length)
        t2=tf.expand_dims(t1, 0) # -> (1,1,Max_instance_Length)
        posencoding = tf.tile(t2, [tf.shape(self.input_Attributes_in)[0],tf.shape(self.input_Attributes_in)[1], 1])
        t_pos, pos_emb_table1 = embedding(
            posencoding,
            Name= 'positionalenc',
            vocab_size= ( args.num_Fields ),
            num_units=args.hidden_units,
            zero_pad=False,
            scale=False,
            l2_reg=args.l2_emb,
            scope="dec_pos",
            #reuse=True,
            with_t=True
        )
        
        self.input_Attributes_in = self.input_Attributes_in+t_pos # adding positional encoding
        
        #print('embedding table .....', users_emb_table, 'The input use is .....', self.user_in)
        print('input_Bundles_in....................', self.input_Attributes_in)
        #self.user_in = tf.nn.embedding_lookup( users_emb_table, self.user_in) #128 x 200 x h
        # second branch   Flatten
        self.input_Attributes_rep= tf.reshape(self.input_Attributes_in, [(tf.shape(self.input_Attributes_in)[0]), 1, (args.hidden_units*args.num_Fields)])

        # First Layer for embedding
        self.input_Attributes_rep_in = tf.compat.v1.layers.dense(inputs= self.input_Attributes_rep, units= args.hidden_units, kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                                       activity_regularizer = tf.keras.regularizers.l2(0.5 * (args.l2_emb)), activation=  tf.nn.relu)
        self.input_Attributes_rep_in = tf.compat.v1.layers.dropout(self.input_Attributes_rep_in,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))    

        # ============================  Second Layer  =========================================
        self.input_Attributes_rep2 = tf.compat.v1.layers.dense(inputs= self.input_Attributes_rep_in, units= args.hidden_units, kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                                       activity_regularizer = tf.keras.regularizers.l2(0.5 * (args.l2_emb)), activation=  tf.nn.relu)
        
        self.input_Attributes_rep2 = self.input_Attributes_rep2 * self.resweight_DNN_1     # Alpha Residual 
        self.input_Attributes_rep2 = tf.compat.v1.layers.dropout(self.input_Attributes_rep2,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
        
        self.input_Attributes_rep2 = self.input_Attributes_rep2 + self.input_Attributes_rep_in

        # ===========================  Third Layer  =================================================
        
        self.input_Attributes_rep3 = tf.compat.v1.layers.dense(inputs= self.input_Attributes_rep2, units= args.hidden_units, kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                                      activity_regularizer = tf.keras.regularizers.l2(0.5 * (args.l2_emb)), activation=  tf.nn.relu)
        
        self.input_Attributes_rep3 = self.input_Attributes_rep3 * self.resweight_DNN_2     # Alpha Residual 
        self.input_Attributes_rep3 = tf.compat.v1.layers.dropout(self.input_Attributes_rep3,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
        
        self.input_Attributes_rep3 = self.input_Attributes_rep3 + self.input_Attributes_rep2
        '''
        # ===========================  Fourth Layer  =================================================
        
        
        self.input_Attributes_rep4 = tf.compat.v1.layers.dense(inputs= self.input_Attributes_rep3, units= args.hidden_units, kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                                       activity_regularizer = tf.keras.regularizers.l2(0.5 * (args.l2_emb)), activation=  tf.nn.relu)
        
        self.input_Attributes_rep4 = self.input_Attributes_rep4 * self.resweight_DNN_3     # Alpha Residual 
        self.input_Attributes_rep4 = tf.compat.v1.layers.dropout(self.input_Attributes_rep4,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
        
        self.input_Attributes_rep4 = self.input_Attributes_rep4 + self.input_Attributes_rep3
        # ===========================  Fifth Layer  =================================================
        
        self.input_Attributes_rep5 = tf.compat.v1.layers.dense(inputs= self.input_Attributes_rep4, units= args.hidden_units, kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                                       activity_regularizer = tf.keras.regularizers.l2(0.5 * (args.l2_emb)), activation=  tf.nn.relu)
        
        self.input_Attributes_rep5 = self.input_Attributes_rep5 * self.resweight_DNN_4     # Alpha Residual 
        self.input_Attributes_rep5 = tf.compat.v1.layers.dropout(self.input_Attributes_rep5,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
        
        self.input_Attributes_rep5 = self.input_Attributes_rep5 + self.input_Attributes_rep4
        
        # ===========================  Sixth Layer  =================================================
        self.input_Attributes_rep6 = tf.compat.v1.layers.dense(inputs= self.input_Attributes_rep5, units= args.hidden_units, kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                                       activity_regularizer = tf.keras.regularizers.l2(0.5 * (args.l2_emb)), activation=  tf.nn.relu)
        
        self.input_Attributes_rep6 = self.input_Attributes_rep6 * self.resweight_DNN_5     # Alpha Residual 
        self.input_Attributes_rep6 = tf.compat.v1.layers.dropout(self.input_Attributes_rep6,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
        
        self.input_Attributes_rep6 = self.input_Attributes_rep6 + self.input_Attributes_rep5
        '''
        #----------------------------------------------------------------------------------------------
        
        
        


        #self.seq_input_concat= tf.concat([self.input_Bundles_in, self.user_in], axis= 3)

        # Build blocks

        for i in range(args.num_blocks_items):
            with tf.compat.v1.variable_scope("num_blocks_%d" % i):
                self.resweight_MultiHead = tf.Variable(0.0, trainable=True, name = 'Res_MultiHead'+str(i))
                self.resweight_MultiHead_feed = tf.Variable(0.0, trainable=True, name = 'Res_Feed'+str(i))
                self.resweight_MultiHead_feed2 = tf.Variable(0.0, trainable=True, name = 'Res_Feed'+str(i))

                print('Entering the self attention')
                # Self-attention
                self.input_Attributes_in = multihead_attention_Item(queries=normalize(self.input_Attributes_in),
                                                keys=self.input_Attributes_in,
                                                resweight_MultiHead = self.resweight_MultiHead,
                                                num_units=args.hidden_units,
                                                num_heads=args.num_heads,
                                                dropout_rate=args.dropout_rate,
                                                is_training=self.is_training,
                                                causality=True,
                                                res= args.use_res,
                                                scope="self_attention")

                # Feed forward
                #self.input_Attributes_feed = self.input_Attributes_in
                self.input_Attributes_feed = feedforward(normalize(self.input_Attributes_in), num_units=[args.hidden_units, args.hidden_units],
                                        dropout_rate=args.dropout_rate, is_training=self.is_training)
                #print('OUPUT of the Feedforward==================>>>>', self.seq)
                self.input_Attributes_in = self.input_Attributes_feed
                #self.input_Attributes_in *= mask
                                


        #print('Before Normalize ==================>>>>', self.seq)
        self.seq_input_concat = normalize(self.input_Attributes_in)
        #print('After Normalize ==================>>>>', self.seq)
        self.seq_input_concat= tf.math.reduce_sum(input_tensor=self.seq_input_concat, axis= 2) # reduce 4d to be 3d
        #print('After REDUCTION *********** ==================>>>>', self.seq)
               
        print('Shape of the seq_input_concat.....', self.seq_input_concat)
        # concat the user embedding ######
        #self.seq_target_concat = tf.concat([self.seq_target_concat, self.user_in], axis= 2)
        self.seq_input_concat = tf.math.reduce_sum(input_tensor=self.seq_input_concat, axis= 1)
        print('Shape of the seq_input_concat.....', self.seq_input_concat)
        self.input_Attributes_rep3 = tf.math.reduce_sum(input_tensor=self.input_Attributes_rep3, axis= 1)
        self.alpha= args.alpha
        self.seq_input_concat = (self.alpha*self.seq_input_concat)+(1-self.alpha)*self.input_Attributes_rep3
        #self.test_logits = tf.nn.softmax(self.seq_input_concat)
        self.test_logits = tf.compat.v1.math.reduce_sum(input_tensor=self.seq_input_concat, axis= 1, keep_dims=True) # reduce 4d to be 3d
        print('Test logits..........................', self.test_logits,'LAbels........', self.Labels)

        self.loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.Labels, tf.float32), logits = self.test_logits))
        #self.loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(self.Labels, tf.float32), logits = self.test_logits))
        
        print('LOSS...........', self.loss)
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #self.loss += sum(reg_losses)

        tf.compat.v1.summary.scalar('loss', self.loss)
      
        if reuse is None:
            #tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)


            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            #self.optimizer = AdamW( weight_decay= 0.3, learning_rate= args.lr)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        #else:
        #    tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.compat.v1.summary.merge_all()

    def predict(self, sess, input_Attributes, target_labels):
        return sess.run([ self.test_logits, self.Labels, self.loss ],
                                        { model.input_Attributes: input_Attributes,
                                          model.Labels: target_labels, model.is_training: False})

# Checking the package Data==========
import  tensorflow as tf
import pickle

def map_features(trainfile, validationfile, testfile): # map the feature entries in all files, kept in self.features dictionary
    features = {}
    features1= read_features(trainfile, features)
    features2= read_features(testfile, features1)
    features_final= read_features(validationfile, features2)
    # print("features_M:", len(self.features))
    return  len(features_final), features_final

def read_features(file, features): # read a feature file
    f = open( file )
    line = f.readline()
    i = len(features)
    while line:
        items = line.strip().split(' ')
        for item in items[1:]:
            if item not in features:
                features[ item ] = i
                i = i + 1
        line = f.readline()
    f.close()
    return features

def read_data(file, features_final):
    # read a data file. For a row, the first column goes into Y_;
    # the other columns become a row in X_ and entries are maped to indexs in self.features
    f = open( file )
    X_ = []
    Y_ = []
    Y_for_logloss = []
    line = f.readline()
    while line:
        items = line.strip().split(' ')
        Y_.append( 1.0*float(items[0]) )
        #print(1.0*float(items[0]))
        if float(items[0]) > 0:# > 0 as 1; others as 0
            v = 1.0
        else:
            v = 0.0
        Y_for_logloss.append( v )
        
        # prepare attributes for each instance.... Number of attributes * 1
        items_accumulated= []
        for item in items[1:]:
          items_accumulated.append(features_final[item] )

        X_.append( items_accumulated )
        line = f.readline()
    f.close()
    return X_, Y_for_logloss






def map_features_lastfm(trainfile, validationfile, testfile): # map the feature entries in all files, kept in self.features dictionary
    features = {}
    features1= read_features_lastfm(trainfile, features)
    features2= read_features_lastfm(testfile, features1)
    features_final= read_features_lastfm(validationfile, features2)
    # print("features_M:", len(self.features))
    return  len(features_final), features_final

def read_features_lastfm(file_name, features): # read a feature file
    f = pd.read_csv(file_name)
    f = f.drop('label',axis=1)
    i = len(features)
    for k in range(0, len(f)):
        items = f.iloc[k].values.tolist()
        for item in items:
            if item not in features:
                features[ item ] = i
                i = i + 1
    return features

def read_data_lastfm(file_name, features_final):
    f = pd.read_csv(file_name)
    X_ = []
    
    Y_for_logloss= f.label.tolist()
    f = f.drop('label',axis=1)
    for k in range(0, len(f)):
        items = f.iloc[k].values.tolist()
        # prepare attributes for each instance.... Number of attributes * 1
        items_accumulated= []
        for item in items:
          items_accumulated.append(features_final[item] )

        X_.append( items_accumulated )
        
    return X_, Y_for_logloss

"""#Main"""

import os
import time
import argparse
import tensorflow as tf
from sklearn import preprocessing
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
#np.random.seed(0)
tf.compat.v1.set_random_seed(0)
def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

'''
class Args:

  #dataset = 'ml-1m_pre'
  #dataset = 'ml-100k'
  #dataset = 'VWFS'
  dataset='frappe'
  train_dir = 'default'
  #lr = 0.000009
  lr= 0.00009 # was 0.0000001
  #maxlen = 50
  hidden_units = 512
  num_blocks_items = 1
  num_epochs = 100
  num_heads = 8 #
  dropout_rate = 0.35 # was 0.3
  l2_emb = 0.01
  num_Fields = 10    # 3 for movielen and 10 for frappe
  use_res = True
  batch_size = 512
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='frappe')
parser.add_argument('--train_dir', default='default')
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--hidden_units', default=512, type=int)
parser.add_argument('--num_epochs', default=70, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--dropout_rate', default=0.45, type=float)
parser.add_argument('--l2_emb', default=0.01, type=float)
parser.add_argument('--use_res', default=True)
parser.add_argument('--num_blocks_items', default=2, type=int)
parser.add_argument('--num_Fields', default=10, type=int)
parser.add_argument('--alpha', default=0.6, type=float)

args = parser.parse_args()

print(args)
print('...................... SEED 42  ..............')

if args.dataset == 'MovieLen' :
  trainfile = '/home/elsayed/CTR/Data/DeepMR/movielen/ml-tag.train.libfm.txt'
  validationfile = '/home/elsayed/CTR/Data/DeepMR/movielen/ml-tag.valid.libfm.txt'
  testfile = '/home/elsayed/CTR/Data/DeepMR/movielen/ml-tag.test.libfm.txt'

if args.dataset == 'frappe' :

  trainfile = '/home/elsayed/CTR/Data/DeepMR/frappe/frappe.train.libfm.txt'
  validationfile = '/home/elsayed/CTR/Data/DeepMR/frappe/frappe.valid.libfm.txt'
  testfile = '/home/elsayed/CTR/Data/DeepMR/frappe/frappe.test.libfm.txt' 

if args.dataset == 'lastfm':
    trainfile = '/home/elsayed/CTR/Data/DeepMR/lastfm/train.csv'
    validationfile = '/home/elsayed/CTR/Data/DeepMR/lastfm/valid.csv'
    testfile = '/home/elsayed/CTR/Data/DeepMR/lastfm/test.csv' 

n_items =0
if args.dataset =='frappe' or args.dataset=='MovieLen':   
  n_items, features_final= map_features(trainfile, validationfile, testfile) 
  train_dataset_X, train_dataset_Y = read_data(trainfile, features_final)
  valid_dataset_X, valid_dataset_Y = read_data(validationfile, features_final)
  test_dataset_X, test_dataset_Y = read_data(testfile, features_final)

else:
  n_items, features_final= map_features_lastfm(trainfile, validationfile, testfile) 
  train_dataset_X, train_dataset_Y = read_data_lastfm(trainfile, features_final)
  valid_dataset_X, valid_dataset_Y = read_data_lastfm(validationfile, features_final)
  test_dataset_X, test_dataset_Y = read_data_lastfm(testfile, features_final)

n_train = len(train_dataset_X)
num_batch = int( n_train / args.batch_size)
model = Model( n_items, args, reuse=None , use_res=True)
print('Len of Training ....',len(train_dataset_X))
print('Len of validation ....',len(valid_dataset_X), 'shape of y ',len(valid_dataset_Y) )
print('Len of Testing ....',len(test_dataset_X))

print('Number of batches.....', num_batch)
#for epoch in range(1, args.num_epochs + 1):
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.initialize_all_variables())
sampler = WarpSampler( train_dataset_X, train_dataset_Y, args)
validation_curve=[]
testing_curve = []
Beta_per_epochs = []
for epoch in range(1, args.num_epochs):
#for epoch in range(1, 2):
    auc_all=0
    loss_all=0
    #for step in tqdm(range(int(num_batch)), total=int(num_batch), ncols=35, leave=False, unit='b'):
    print('Epoch==> ',epoch)
    #for step in range(int(num_batch)):
    step=0

    for i in range(0, num_batch):
    #for i in range(0, 2):
        step+=1
        # if step% 100 ==0:
        #   print('100 finished!')
        #input_user, input_articles, input_Media, input_Friends, target_article, target_media, target_friends ,target_labels = Prepare_Packages(train_data, args)
        input_Attributes, target_labels = sampler.next_batch()

        loss, beta, _ = sess.run([model.loss, model.beta, model.train_op],
                                      {model.input_Attributes: input_Attributes,
                                       model.Labels: target_labels, model.is_training: True})

        #auc_all= auc_all+ auc
        loss_all= loss_all + loss

    print('LOSS: ', loss_all/int(num_batch))
    Beta_per_epochs.append(beta)
    
    if epoch %1 ==0:
    #print('Test Length****', len(Input_Seq_Test)) , ,
         valid_auc, Valid_loss= evaluate(model, valid_dataset_X, valid_dataset_Y, args, sess)
         validation_curve.append(valid_auc)
         print(' Valid AUC =', valid_auc,'log Valid loss===',Valid_loss)
         
print("***Testing***")
test_auc, test_loss=evaluate(model, test_dataset_X, test_dataset_Y, args, sess)
print(' Testing AUC=====', test_auc,'log Test Loss====', test_loss)
testing_curve.append(test_auc)
#print('Epochs Beta ...', Beta_per_epochs)


#import matplotlib.pyplot as plt


#x= np.arange(1,len(testing_curve)+1,1)
#plt.plot(x, validation_curve )  # Matplotlib plot.
#plt.plot(x, testing_curve )  # Matplotlib plot.

