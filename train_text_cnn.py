#! /usr/bin/env python
#coding=utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import textcnn.data_input_helper as data_helpers
from textcnn.text_cnn import TextCNN
import math
from tensorflow.contrib import learn
from gensim.models import word2vec
from utils.vocab_trans import get_vocab_trans
from utils.data_slice import *
from load_data import load_data

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("train_data_file", "/var/proj/sentiment_analysis/data/cutclean_tiny_stopword_corpus10000.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("train_data_file", "data/train.csv", "Data source for the positive data.")
tf.flags.DEFINE_string("train_label_data_file", "", "Data source for the label data.")
tf.flags.DEFINE_string("w2v_file", "model/w2v_test_128.model", "w2v_file path")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

learning_rate = 0.001




def load_data(w2v_model):
    """Loads starter word-vectors and train/dev/test data."""
    # Load the starter word vectors
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.train_data_file)
    _, _, events, _, train_index, test_index = data_helpers.load_all_info("data/train.csv")
    #print(x_text)
    #print("y", y)
    #input()
    # for x in x_text:
    #     l = len(x.split(" "))
    #     break
    #print(x_text.shape)

    #max_document_length = max([len(x.split(" ")) for x in x_text])
    max_document_length = 100
    print ('len(x) = ',len(x_text),' ',len(y))
    print(' max_document_length = ' , max_document_length)

    x = []
    vocab_size = 0
    if(w2v_model is None):
        '''vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        vocab_size = len(vocab_processor.vocabulary_)

        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", str(int(time.time()))))
        vocab_processor.save("vocab.txt")
        print( 'save vocab.txt')'''
        
        '''model = word2vec.Word2Vec.load(FLAGS.w2v_file)   
        vects = model.wv.vocab
        vocabs = list(vects.keys())
        vec = []
        for i in vocabs:
            vec.append(model[i])
        #print(x_text[0])
        #print(len(x_text[0]))
        vocab_size = len(vocabs)
        x = [[0] * (max_document_length+1) for i in range(len(x_text))]
        #print(x[7851][1])
        for i,li in enumerate(x_text):
            line = li.split(" ")
            print(i)
            for j in range(len(line)):
                #print(i,j)
                #print(x[i][j])
                #print(vocabs.index(line[j]))
                #print(line[j])
                #print(model["标准间"])
                #print(vocabs.index(line[j]))
                #print(model[line[j]])
                if line[j] in vocabs:
                    x[i][j] = vocabs.index(line[j])
        x = np.array(x)
        print("x done")'''
        vocab_size, _, x = get_vocab_trans(vocabpath='utils/vocab.txt', transpath='utils/trans.txt')
        x = x[:, :max_document_length]
        print(vocab_size)
        #input()
    else:
        x = data_helpers.get_text_idx(x_text,w2v_model.vocab_hash,max_document_length)
        vocab_size = len(w2v_model.vocab_hash)
        print('use w2v .bin')

    '''np.random.seed(10)
    train_index = get_train_index()'''
    x_train = x[train_index]
    y_train = y[train_index]
    x_dev = x[test_index]
    y_dev = y[test_index]
    '''shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]'''

    return x_train,x_dev,y_train,y_dev,vocab_size

def train(w2v_model):
    # Training
    # ==================================================
    x_train, x_dev, y_train, y_dev ,vocab_size= load_data(w2v_model)
    #y_train = tf.one_hot(y_train, depth=2, dtype=tf.float32)
    #y_dev = tf.one_hot(y_dev, depth=2, dtype=tf.float32)
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                w2v_model,
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=vocab_size,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                # _, step, summaries, loss, accuracy,(w,idx) = sess.run(
                #     [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,cnn.get_w2v_W()],
                #     feed_dict)
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # print w[:2],idx[:2]
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                #print("testing  {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return step, accuracy, loss

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)


            def dev_test():
                batches_dev = data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                ai = 0
                li = 0
                #lent = len(list(batches_dev))
                #print(lent)
                i = 0
                for batch_dev in batches_dev:
                    x_batch_dev, y_batch_dev = zip(*batch_dev)
                    step, a, l = dev_step(x_batch_dev, y_batch_dev, writer=dev_summary_writer)
                    ai += a
                    li += l
                    i += 1
                    #print(a,l,i)
                print("testing : setp:{} acc {}, loss {}".format(step, ai/i,li/i))
                

            # Training loop. For each batch...
            for i in range(2):
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    '''for i in x_batch:
                        for j in i:
                            if j > 7851:
                                print(j)'''
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    # Training loop. For each batch...
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_test()


                if current_step % FLAGS.checkpoint_every == 0:
                    #path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    path = saver.save(sess, "./model/textcnn/textcnn_model", global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":  
    #w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)
    train(None)