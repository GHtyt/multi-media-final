# -*- coding:utf-8 -*-
import tensorflow as tf
from utils.vocab_trans import *
from utils.data_slice import *
from load_data import load_data, batch_iter


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

is_training = False
is_testing = True


learning_rate = 0.001

embedding_size = 128
#vocab_size = 10
batch_size = 100
num_iter = 2000
hidden_units = 64
num_labels = 2
max_seq_length = 100 #mean len :68.23612591302539
#max_seq_length = 1632

x_train, label_train, vocab_size, events_train, pic_train, x_test, label_test, events_test, pic_test = load_data()

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

cell = tf.contrib.rnn.GRUCell(hidden_units)
#zero_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

out, final_state = tf.nn.dynamic_rnn(
    cell, inputs_embedded, #initial_state=zero_state,
    dtype=tf.float32, #time_major=True,
    #sequence_length=length(inputs_embedded),
)

#print(final_state)
mid_unit = 32
logits = connect(final_state, hidden_units, mid_unit)
logits = tf.nn.dropout(connect(logits, mid_unit, num_labels), keep_prob=dropout_rate) # (20, 1632, 2)
log_probs = tf.nn.log_softmax(logits, axis=-1)
one_hot_labels = tf.one_hot(input_y, depth=num_labels, dtype=tf.float32)
loss = tf.reduce_mean(-tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
predict = tf.argmax(tf.nn.softmax(logits), axis=1, name="predictions")
acc = tf.reduce_mean(tf.cast(tf.equal(input_y, tf.cast(predict, dtype=tf.int32)), "float"), name="accuracy")

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

if is_training:
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for current_step in range(num_iter):
            shuffIndex = np.random.permutation(range(label_train.shape[0]))[:batch_size]
            batch_input = x_train[shuffIndex]
            batch_labels = label_train[shuffIndex]
            #print(batch_input.shape, batch_labels.shape)
            '''l, l2=sess.run([logits, final_state],feed_dict={
                input_x:batch_input,
                input_y:batch_labels,
                dropout_rate:0.9
            })
            print(np.array(l).shape, np.array(l2).shape)'''
            l,a,_=sess.run([loss,acc,train_op],feed_dict={
                input_x:batch_input,
                input_y:batch_labels,
                dropout_rate:1
            })
            print("{}, acc:{}, loss:{}".format(current_step,a,l))

            if (current_step + 1) % 20 == 0:
                print("testing")
                data_size = len(label_test)
                print(data_size)
                epoch = int((data_size-1)/batch_size) + 1
                li = 0
                ai = 0
                for cs in range(epoch):
                    si = cs * batch_size
                    ei = min((cs + 1) * batch_size, data_size)
                    batch_input = x_test[si:ei]
                    batch_labels = label_test[si:ei]
                    batch_pic = pic_test[si:ei]
                    l, a=sess.run([loss, acc],feed_dict={
                        input_x:batch_input,
                        input_y:batch_labels,
                        dropout_rate:1
                    })
                    li += l
                    ai += a
                    print("{}, acc:{}, loss:{}".format(cs, a, l))
                print("acc:{}, loss:{}".format(ai/epoch, li/epoch))
                print("test_end")
            if (current_step + 1) % 1000 == 0:
                path = saver.save(sess, "./model/gru/gru_model", global_step=current_step+1)
                print("Saved model checkpoint to {}\n".format(path))

if is_testing:
    test_iter = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph('./model/gru/gru_model-1000.meta')
        #saver.restore(sess, 'model/bert-model-900.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model/gru/'))
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
        print(len(x_test), len(label_test))
        l,a=sess.run([loss,acc],feed_dict={
            input_x:x_test,
            input_y:label_test,
            dropout_rate:1
        })
        print("acc:{}, loss:{}".format(a,l))
        #print("mean: acc:{}, loss:{}".format(a/(test_iter),l/(test_iter)))