import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import VGG19.Vgg19 as Vgg19
from VGG19.utils import *

def connect(layer, input_size, output_size):    
    w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[output_size]))
    y = tf.matmul(layer, w) + b
    return tf.maximum(0.01*y, y)
    #return tf.nn.relu(y)

pic_len = 224
p5_len = 7
num_labels = 2

is_training = True
is_testing = True

learning_rate = 0.001
batch_size = 10
num_iter = 5000
num_labels = 2


input_x = tf.placeholder(tf.float32, [None, pic_len, pic_len, 3])
input_y = tf.placeholder(tf.int32)

dropout_rate=tf.placeholder(dtype=tf.float32,name="dropout")

output = Vgg19.Model(input_x, 'model/vgg19/Vgg19.model').output

h_flat1 = tf.reshape(output, [-1, p5_len * p5_len * 512])

mid_unit = 1024
logits = connect(h_flat1, p5_len * p5_len * 512, mid_unit)
logits = tf.nn.dropout(connect(logits, mid_unit, num_labels), keep_prob=dropout_rate) # (20, 1632, 2)

#logits = connect(h_flat1, p5_len * p5_len * 512, num_labels)

log_probs = tf.nn.log_softmax(logits, axis=-1)
one_hot_labels = tf.one_hot(input_y, depth=num_labels, dtype=tf.float32)
loss = tf.reduce_mean(-tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
predict = tf.argmax(tf.nn.softmax(logits), axis=1, name="predictions")
acc = tf.reduce_mean(tf.cast(tf.equal(input_y, tf.cast(predict, dtype=tf.int32)), "float"), name="accuracy")

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def load_data():
    x_truth = get_images(image_path = "data/train/truth_pic")
    x_rumor = get_images(image_path = "data/train/rumor_pic")
    print(x_truth.shape)
    print(x_rumor.shape)
    x = np.concatenate((x_truth, x_rumor), axis=0)
    y = [1]*len(x_truth) + [0] * len(x_rumor)
    return x, np.array(y)

def load_path():
    x_truth = get_paths(image_path = "data/train/truth_pic")
    x_rumor = get_paths(image_path = "data/train/rumor_pic")
    #x_truth = get_paths(image_path = "data/train/tp")
    #x_rumor = get_paths(image_path = "data/train/rp")
    '''x = x_truth+x_rumor
    y = [1]*len(x_truth) + [0] * len(x_rumor)
    return np.array(x), np.array(y)'''
    return np.array(x_truth), np.array(x_rumor), np.array([1]*len(x_truth)), np.array([0]*len(x_rumor))


x_1, x_0, y_1, y_0 = load_path()
#x,labels = load_path()
#print(x.shape)
#print(labels.shape)

if is_training:
    valid = {}
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for current_step in range(num_iter):
            shuff1 = np.random.permutation(range(y_1.shape[0]))[:batch_size]
            shuff0 = np.random.permutation(range(y_0.shape[0]))[:batch_size]
            batch_input = np.concatenate((x_1[shuff1], x_0[shuff0]), axis=0)
            batch_input = get_path_images(batch_input)
            batch_labels = np.concatenate((y_1[shuff1], y_0[shuff0]), axis=0)
            '''batch_input = np.concatenate((x_1, x_0), axis=0)
            #print(batch_input[0])
            batch_input = get_path_images(batch_input)
            batch_labels = np.concatenate((y_1, y_0), axis=0)'''
            '''
            for i in range(batch_size):
                if x[shuffIndex[i]] in valid.keys():
                    if labels[shuffIndex[i]] != valid[x[shuffIndex[i]]]:
                        print(x[shuffIndex[i]], labels[shuffIndex[i]], valid[x[shuffIndex[i]]])
                else:
                    valid[x[shuffIndex[i]]]=labels[shuffIndex[i]]'''
            '''l, l2=sess.run([logits, final_state],feed_dict={
                input_x:batch_input,
                input_y:batch_labels,
                dropout_rate:0.9
            })
            print(np.array(l).shape, np.array(l2).shape)'''
            l, a, ac, _=sess.run([loss, acc, logits, train_op],feed_dict={
                input_x:batch_input,
                input_y:batch_labels,
                dropout_rate:1
            })
            #print("{}, acc:{}, loss:{}".format(current_step, a, l))
            print("{}, acc:{}, loss:{} {}".format(current_step,a,l, ac[0]))
            
            if (current_step + 1) % 1000 == 0:
                path = saver.save(sess, "./model/vgg/vgg_model", global_step=current_step+1)
                print("Saved model checkpoint to {}\n".format(path))