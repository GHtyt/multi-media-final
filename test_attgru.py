# -*- coding:utf-8 -*-
import tensorflow as tf
from utils.vocab_trans import *
from utils.data_slice import *


#tf.enable_eager_execution()
def connect(layer, input_size, output_size):    
    w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[output_size]))
    y = tf.matmul(layer, w) + b
    return tf.nn.relu(y)

def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

is_training = True
is_testing = True


learning_rate = 0.001

embedding_size = 128
#vocab_size = 10
batch_size = 200
num_iter = 1000
hidden_units = 64
num_labels = 2
max_seq_length = 100 #mean len :68.23612591302539
#max_seq_length = 1632

_, labels = load_texts_and_labels()
vocab_size, vocabs, x = get_vocab_trans(vocabpath='utils/vocab.txt', transpath='utils/trans.txt')
#vocab_size, vocabs, x = get_vocab_trans(vocabpath='utils/vocab_%d.txt'%(embedding_size), transpath='utils/trans_%d.txt'%(embedding_size))
x = x[:,:max_seq_length]
#max_seq_length = x.shape[1]
#print(x.shape[1])

#x = np.array(x)
labels = np.array(labels)
#print(x.shape)
#print(labels.shape)

input_x = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_x')
input_y = tf.placeholder(shape=(None), dtype=tf.int32, name='input_y')

dropout_rate=tf.placeholder (dtype=tf.float32,name="dropout")


model = word2vec.Word2Vec.load('model/w2v_test_%d.model'%(embedding_size))   
vects = model.wv.vocab
vocabs = list(vects.keys())
vec = []
for i in vocabs:
    vec.append(model[i])
vec = np.array(vec)
#print(vec.shape)
init = tf.constant_initializer(vec.astype(np.float32))

embeddings = tf.get_variable("word_embeddings", shape = [vocab_size, embedding_size], initializer=init)


#embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)

inputs_embedded = tf.nn.embedding_lookup(embeddings, input_x)
#print("start")
#print(inputs_embedded)

fw_cell = tf.contrib.rnn.GRUCell(hidden_units)
bw_cell = tf.contrib.rnn.GRUCell(hidden_units)
#zero_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

out, final_state = tf.nn.bidirectional_dynamic_rnn(
    fw_cell, bw_cell, inputs_embedded, #initial_state=zero_state,
    dtype=tf.float32, #time_major=True,
    #sequence_length=length(inputs_embedded),
)# out (200, 100, 128) final_state (2, 200, 64)


#logits = tf.nn.dropout(connect(final_state, hidden_units, num_labels), keep_prob=dropout_rate)

#final_state = tf.concat(final_state, axis=0)
#final_state = tf.reshape(final_state, (-1, 2*hidden_units))
#print(final_state)

flag = 1
if flag :   
    mid_unit = 64 
    fs = tf.concat(out, axis=2)
    logits = connect(fs, 2*hidden_units, mid_unit)
    att = tf.get_variable(name='attention', shape=[mid_unit], dtype=tf.float32)
    #att = tf.get_variable(tf.random_normal([mid_unit], stddev=0.1), shape=[mid_unit], dtype=tf.float32)
    logits = tf.multiply(logits, att)
    logits = tf.reduce_max(logits, axis=1)
    logits = tf.nn.dropout(connect(logits, mid_unit, num_labels), keep_prob=dropout_rate) # (20, 1632, 2)

else:
    mid_unit = 64
    fs = tf.concat(final_state, axis=1)
    logits = connect(fs, 2*hidden_units, mid_unit)
    att = tf.get_variable(name='attention', shape=[mid_unit], dtype=tf.float32)
    #att = tf.get_variable(tf.random_normal([mid_unit], stddev=0.1), shape=[mid_unit], dtype=tf.float32)
    logits = tf.multiply(logits, att)
    logits = tf.nn.dropout(connect(logits, mid_unit, num_labels), keep_prob=dropout_rate) # (20, 1632, 2)


log_probs = tf.nn.log_softmax(logits, axis=-1)
one_hot_labels = tf.one_hot(input_y, depth=num_labels, dtype=tf.float32)
loss = tf.reduce_mean(-tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
predict = tf.argmax(tf.nn.softmax(logits), axis=1, name="predictions")
acc = tf.reduce_mean(tf.cast(tf.equal(input_y, tf.cast(predict, dtype=tf.int32)), "float"), name="accuracy")

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

train_index = get_train_index()
test_index = get_test_index()
x_train=x[train_index]
labels_train = labels[train_index]
x_test=x[test_index]
labels_test = labels[test_index]

if is_training:
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for current_step in range(num_iter):
            shuffIndex = np.random.permutation(range(labels_train.shape[0]))[:batch_size]
            batch_input = x_train[shuffIndex]
            batch_labels = labels_train[shuffIndex]
            '''l, l2, l3=sess.run([out, logits, fs],feed_dict={
                input_x:batch_input,
                input_y:batch_labels,
                dropout_rate:0.9
            })
            print(np.array(l).shape, np.array(l2).shape, np.array(l3).shape)'''
            l,a,_=sess.run([loss,acc,train_op],feed_dict={
                input_x:batch_input,
                input_y:batch_labels,
                dropout_rate:1
            })
            print("{}, acc:{}, loss:{}".format(current_step,a,l))
            
            if (current_step + 1) % 1000 == 0:
                path = saver.save(sess, "./model/attgru/attgru_model", global_step=current_step+1)
                print("Saved model checkpoint to {}\n".format(path))

if is_testing:
    test_iter = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph('./model/attgru/attgru_model-1000.meta')
        #saver.restore(sess, 'model/bert-model-900.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model/attgru/'))
        '''li = 0
        ai = 0
        for current_step in range(test_iter):
            shuffIndex = np.random.permutation(range(labels_test.shape[0]))[:batch_size]
            batch_input = x_test[shuffIndex]
            batch_labels = labels_test[shuffIndex]
            l,a=sess.run([loss,acc],feed_dict={
                input_x:batch_input,
                input_y:batch_labels,
                dropout_rate:1
            })
            li += l
            ai += a
            print("{}, acc:{}, loss:{}".format(current_step,a,l))'''
        l,a=sess.run([loss,acc],feed_dict={
            input_x:x_test,
            input_y:labels_test,
            dropout_rate:1
        })
        print("acc:{}, loss:{}".format(a,l))
        #print("mean: acc:{}, loss:{}".format(a/(test_iter),l/(test_iter)))