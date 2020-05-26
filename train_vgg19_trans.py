import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import VGG19.Vgg19 as Vgg19
from VGG19.utils import *
from load_data import load_data, batch_iter

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

def load_pic():
    x_train, label_train, vocab_size, events_train, pic_train, x_test, label_test, events_test, pic_test = load_data()

    ptrain = []
    ltrain =[]
    ptest = []
    ltest =[]
    for i,p in enumerate(pic_train):
        #print(p)
        if len(p)>0:
            #print(label_train[i])
            ptrain.append(p)
            ltrain.append(label_train[i])
    for i,p in enumerate(pic_test):
        if len(p)>0:
            ptest.append(p)
            ltest.append(label_train[i])
    return np.array(ptrain), np.array(ltrain), np.array(ptest), np.array(ltest)

x_train, y_train, x_test, y_test = load_pic()
print(len(y_train))
print(len(y_test))
print(np.sum(y_train))
print(np.sum(y_test))

acc_all = []
loss_all = []

if is_training:
    valid = {}
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for current_step in range(num_iter):
            shuff1 = np.random.permutation(range(len(y_train)))[:batch_size]
            batch_input = x_train[shuff1]
            batch_input = get_path_images(batch_input)
            #print(batch_input[0])
            batch_labels = y_train[shuff1]
            '''l, l2=sess.run([logits, log_probs],feed_dict={
                input_x:batch_input,
                input_y:batch_labels,
                dropout_rate:0.9
            })
            print(l[0], np.array(l2).shape, l2[0])'''
            l, a, ac, _=sess.run([loss, acc, logits, train_op],feed_dict={
                input_x:batch_input,
                input_y:batch_labels,
                dropout_rate:1
            })
            #print("{}, acc:{}, loss:{}".format(current_step, a, l))
            print("{}, acc:{}, loss:{} {}".format(current_step,a,l, ac[0]))

            if (current_step + 1) % 100 == 0:
                print("testing")
                data_size = len(y_test)
                print(data_size)
                epoch = int((data_size-1)/batch_size) + 1
                li = 0
                ai = 0
                for cs in range(epoch):
                    si = cs * batch_size
                    ei = min((cs + 1) * batch_size, data_size)
                    batch_input = x_test[si:ei]
                    batch_input = get_path_images(batch_input)
                    batch_labels = y_test[si:ei]
                    l, a=sess.run([loss, acc],feed_dict={
                        input_x:batch_input,
                        input_y:batch_labels,
                        dropout_rate:1
                    })
                    li += l
                    ai += a
                    print("{}, acc:{}, loss:{}".format(cs, a, l))
                acc_all.append(ai/epoch)
                loss_all.append(li/epoch)
                print("acc:{}, loss:{}".format(ai/epoch, li/epoch))
                print("test_end")
            if (current_step + 1) % 1000 == 0:
                path = saver.save(sess, "./model/vgg/vgg_model", global_step=current_step+1)
                print("Saved model checkpoint to {}\n".format(path))
print(acc_all)
print(loss_all)