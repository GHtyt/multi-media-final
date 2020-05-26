# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from gensim.models import word2vec
import sys
sys.path.append("..")
import VGG19.Vgg19 as vgg19 
from VGG19.utils import *


#tf.enable_eager_execution()
def connect(layer, input_size, output_size):    
    w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[output_size]))
    y = tf.matmul(layer, w) + b
    #return tf.nn.relu(y)
    return tf.maximum(0.01*y, y)

def _conv_relu(input_layer):
    conv = tf.nn.conv2d(input_layer, strides=[1, 1, 1, 1], padding='SAME')
    b = tf.Variable(tf.constant(0.1, shape=input_layer.shape))
    relu = tf.nn.relu(conv + wb[1])
    return relu

def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length


def get_embedding(embedding_size = 128):
    model = word2vec.Word2Vec.load('model/w2v_test_%d.model'%(embedding_size))   
    vects = model.wv.vocab
    vocabs = list(vects.keys())
    vocab_size = len(vocabs)
    vec = []
    for i in vocabs:
        vec.append(model[i])
    vec = np.array(vec)
    #print(vec.shape)
    init = tf.constant_initializer(vec.astype(np.float32))
    embeddings = tf.get_variable("word_embeddings", shape = [vocab_size, embedding_size], initializer=init)

    return embeddings

def attgru1(input_x, embeddings, hidden_units = 64, mid_unit = 64):
        
    #input_x = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_x')

    #embeddings = get_embedding(embedding_size)

    inputs_embedded = tf.nn.embedding_lookup(embeddings, input_x)

    fw_cell = tf.contrib.rnn.GRUCell(hidden_units)
    bw_cell = tf.contrib.rnn.GRUCell(hidden_units)

    out, final_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell, bw_cell, inputs_embedded,
        dtype=tf.float32,
    )
  
    fs = tf.concat(out, axis=2)
    logits = connect(fs, 2*hidden_units, mid_unit)
    att = tf.get_variable(name='attention', shape=[mid_unit], dtype=tf.float32)
    logits = tf.multiply(logits, att)
    logits = tf.reduce_max(logits, axis=1)
    return logits

def attgru2(input_x, embeddings, hidden_units = 64, mid_unit = 64):

    #embeddings = get_embedding(embedding_size)
    inputs_embedded = tf.nn.embedding_lookup(embeddings, input_x)

    fw_cell = tf.contrib.rnn.GRUCell(hidden_units)
    bw_cell = tf.contrib.rnn.GRUCell(hidden_units)

    out, final_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell, bw_cell, inputs_embedded,
        dtype=tf.float32,
    )
    fs = tf.concat(final_state, axis=1)
    logits = connect(fs, 2*hidden_units, mid_unit)
    att = tf.get_variable(name='attention', shape=[mid_unit], dtype=tf.float32)
    logits = tf.multiply(logits, att)
    #print("here\n\n", fs.shape, logits.shape)
    return logits

def vgggru(vgg, hidden_units = 64):
    pool1 = vgg.pool1 #112 32
    pool2 = vgg.pool3 #56 64
    pool3 = vgg.pool4 #28 128
    pool4 = vgg.pool4 #14 256
    pool5 = vgg.pool5 #7 512

    fc1 = tf.expand_dims(connect(pool1, 112*112*32, 64), 0)
    fc2 = tf.expand_dims(connect(pool1, 56*56*64, 64), 0)
    fc3 = tf.expand_dims(connect(pool1, 28*28*128, 64), 0)
    fc4 = tf.expand_dims(connect(pool1, 14*14*256, 64), 0)
    fc5 = tf.expand_dims(connect(pool1, 7*7*512, 64), 0)

    input_fc = tf.concat((fc1, fc2, fc3, fc4, fc5), axis = 0)

    cell = tf.contrib.rnn.GRUCell(hidden_units)

    out, final_state = tf.nn.dynamic_rnn(
        cell, input_fc, 
        dtype=tf.float32, 
    )

    return final_state

