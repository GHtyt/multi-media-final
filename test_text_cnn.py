#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import textcnn.data_input_helper as data_helpers
from textcnn.text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from gensim.models import word2vec
from utils.vocab_trans import get_vocab_trans


# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("valid_data_file", "./data/train.csv", "Data source for the positive data.")
tf.flags.DEFINE_string("w2v_file", "../data/vectors.bin", "w2v_file path")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1501842714/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



def load_data(w2v_model,max_document_length = 1290):
    """Loads starter word-vectors and train/dev/test data."""
    # Load the starter word vectors
    print("Loading data...")
    '''x_text, y_test = data_helpers.load_data_and_labels(FLAGS.valid_data_file)
    y_test = np.argmax(y_test, axis=1)

    if(max_document_length == 0) :
        max_document_length = max([len(x.split(" ")) for x in x_text])

    print ('max_document_length = ' , max_document_length)

    x = data_helpers.get_text_idx(x_text,w2v_model.vocab_hash,max_document_length)'''

    x_text, y = data_helpers.load_data_and_labels(FLAGS.valid_data_file)

    #max_document_length = max([len(x.split(" ")) for x in x_text])
    max_document_length = 100
    print ('len(x) = ',len(x_text),' ',len(y))
    print(' max_document_length = ' , max_document_length)

    x = []
    vocab_size = 0
    if(w2v_model is None):
        vocab_size, _, x = get_vocab_trans(vocabpath='utils/vocab.txt', transpath='utils/trans.txt')
        x = x[:, :max_document_length]
        print(vocab_size)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    dev_sample_index = -1 * int(1 * float(len(y)))
    x = x_shuffled[dev_sample_index:]
    y_test = y_shuffled[dev_sample_index:]


    return x,y_test

def eval(w2v_model):
    # Evaluation
    # ==================================================
    #checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            #saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            
            saver = tf.train.import_meta_graph('./model/textcnn/textcnn_model-9700.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./model/textcnn/'))

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
          
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            x_test, y_test = load_data(w2v_model,1290)
            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
            #y_test = np.array(y_test)

    # Print accuracy if y_test is defined
    #y_test = np.array(y_test)
    y_test = y_test[:,1:]
    #print(all_predictions[0:10])
    #print(y_test[0:10])
    k=0
    for i in range(38471):
        if (y_test[i] == all_predictions[i]):
            k += 1
    print(k,k/38471)
    if y_test is not None:
        correct_predictions = float(np.sum(all_predictions == y_test))
        print(correct_predictions)
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

    # Save the evaluation to a csv
    #predictions_human_readable = np.column_stack(all_predictions)
    #out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    #print("Saving evaluation to {0}".format(out_path))
    #with open(out_path, 'w') as f:
        #csv.writer(f).writerows(predictions_human_readable)



if __name__ == "__main__":
    #w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)
    eval(None)
