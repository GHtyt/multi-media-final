import tensorflow as tf
from bert import modeling
import os
import create_input
import tokenization
import numpy as np
import csv
from utils.data_slice import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

is_training = False
is_testing = True

bert_config = modeling.BertConfig.from_json_file("test_model/bert_config.json")
vocab_file="test_model/vocab.txt"
batch_size=20
num_labels=2
max_seq_length=128
iter_num=1001
lr=0.00005
if max_seq_length > bert_config.max_position_embeddings: 
    raise ValueError("超出模型最大长度")

texts, labels = load_texts_and_labels()

'''with open('data/train.csv', encoding="utf-8")as f:
    data = csv.reader(f)
    for line in data:
        if (len(line[-1]) < 2) and (int(line[-1]) < 2):# 这里演示一个二分类问题，但训练样本并没有认真处理过，所以去掉label大于1的。
            texts.append(line[1])
            labels.append(line[-1])'''


    

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file) # token 处理器，主要作用就是 分字，将字转换成ID。vocab_file 字典文件路径
input_idsList=[]
input_masksList=[]
segment_idsList=[]
for t in texts:
    single_input_id, single_input_mask, single_segment_id=create_input.convert_single_example(max_seq_length,tokenizer,t)
    '''print(t)
    print(len(single_input_id))
    print(single_input_mask)
    print(single_segment_id)
    break'''
    input_idsList.append(single_input_id)
    input_masksList.append(single_input_mask)
    segment_idsList.append(single_segment_id)
    
input_idsList=np.asarray(input_idsList,dtype=np.int32)
input_masksList=np.asarray(input_masksList,dtype=np.int32)
segment_idsList=np.asarray(segment_idsList,dtype=np.int32)
labels=np.asarray(labels,dtype=np.int32)


sIndex = np.random.permutation(np.arange(len(texts)))
input_idsList=input_idsList[sIndex]
input_masksList=input_masksList[sIndex]
segment_idsList=segment_idsList[sIndex]
labels = labels[sIndex]

'''input_ids=tf.placeholder (shape=[batch_size,max_seq_length],dtype=tf.int32,name="input_ids")
input_mask=tf.placeholder (shape=[batch_size,max_seq_length],dtype=tf.int32,name="input_mask")
segment_ids=tf.placeholder (shape=[batch_size,max_seq_length],dtype=tf.int32,name="segment_ids")
input_labels=tf.placeholder (shape=batch_size,dtype=tf.int32,name="input_ids")'''
input_ids=tf.placeholder (shape=[None,max_seq_length],dtype=tf.int32,name="input_ids")
input_mask=tf.placeholder (shape=[None,max_seq_length],dtype=tf.int32,name="input_mask")
segment_ids=tf.placeholder (shape=[None,max_seq_length],dtype=tf.int32,name="segment_ids")
input_labels=tf.placeholder (shape=None,dtype=tf.int32,name="input_ids")
dropout_rate=tf.placeholder (dtype=tf.float32,name="dropout")

model = modeling.BertModel(
    config=bert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False
)

output_layer = model.get_pooled_output()
hidden_size = output_layer.shape[-1].value 


output_weights = tf.get_variable(
    "output_weights", [num_labels, hidden_size],
    initializer=tf.truncated_normal_initializer(stddev=0.02))
output_bias = tf.get_variable(
    "output_bias", [num_labels], initializer=tf.zeros_initializer())
with tf.variable_scope("loss"):
    output_layer = tf.nn.dropout(output_layer, keep_prob=dropout_rate)
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(input_labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    predict = tf.argmax(tf.nn.softmax(logits), axis=1, name="predictions")
    acc = tf.reduce_mean(tf.cast(tf.equal(input_labels, tf.cast(predict, dtype=tf.int32)), "float"), name="accuracy")

train_op = tf.train.AdamOptimizer(lr).minimize(loss)


if is_training:
    init_checkpoint = "test_model/bert_model.ckpt"
    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                        init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"

    #print(len(texts))    #38471
    train_index = get_train_index()
    train_labels = labels[train_index]
    train_input_idsList=input_idsList[train_index]
    train_input_masksList=input_masksList[train_index]
    train_segment_idsList=segment_idsList[train_index]


    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for current_step in range(iter_num):
            shuffIndex = np.random.permutation(range(train_len))[:batch_size]
            batch_labels = train_labels[shuffIndex]
            batch_input_idsList=train_input_idsList[shuffIndex]
            batch_input_masksList=train_input_masksList[shuffIndex]
            batch_segment_idsList=train_segment_idsList[shuffIndex]
            l,a,_=sess.run([loss,acc,train_op],feed_dict={
                input_ids:batch_input_idsList,input_mask:batch_input_masksList,
                segment_ids:batch_segment_idsList,input_labels:batch_labels,
                dropout_rate:0.9
            })
            print("{}, acc:{}, loss:{}".format(current_step,a,l))
            
            if current_step % 100 == 0:
                path = saver.save(sess, "./model/bert_model", global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


test_iter = 10
if is_testing:

    test_index = get_test_index()
    test_labels = labels[test_index]
    test_input_idsList=input_idsList[test_index]
    test_input_masksList=input_masksList[test_index]
    test_segment_idsList=segment_idsList[test_index]


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph('./model/bert_model-900.meta')
        #saver.restore(sess, 'model/bert-model-900.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        li = 0
        ai = 0
        for current_step in range(test_iter):
            shuffIndex = np.random.permutation(range(test_len))[:5*batch_size]
            l,a=sess.run([loss,acc],feed_dict={
                input_ids:test_input_idsList[shuffIndex],
                input_mask:test_input_masksList[shuffIndex],
                segment_ids:test_segment_idsList[shuffIndex],
                input_labels:test_labels[shuffIndex],
                dropout_rate:1.0
            })
            li += l
            ai += a
            print("{}, acc:{}, loss:{}".format(current_step,a,l))
        
        print("mean: acc:{}, loss:{}".format(ai/(test_iter),li/(test_iter)))