from gru.gru import *
from textcnn.text_cnn2 import *
import VGG19.Vgg19 as vgg19 
from VGG19.utils import *
import textcnn.data_input_helper as data_helpers
from utils.vocab_trans import get_vocab_trans
from utils.data_slice import *
from utils.GRL import *
from load_data import load_data, batch_iter

is_training = True
is_testing = True

batch_size = 10
max_seq_length = 100
num_labels = 2
event_labels = 8
num_iter = 5000
learning_rate = 0.001



x_train, label_train, vocab_size, events_train, pic_train, x_test, label_test, events_test, pic_test = load_data()


input_x = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_x')
input_pic = tf.placeholder(tf.float32, [None, 224, 224, 3])
input_y = tf.placeholder(shape=(None), dtype=tf.int32, name='input_y')
input_dis = tf.placeholder(shape=(None), dtype=tf.int32, name='input_discriminator')

dropout_rate=tf.placeholder(dtype=tf.float32,name="dropout")

embeddings = get_embedding(128)

out_gru = attgru2(input_x, embeddings, 64, 64)
cnn = TextCNN(
    None,
    sequence_length=100,
    num_classes=64,
    vocab_size=vocab_size,
    input_x=input_x,
    embeddings=embeddings,
    embedding_size=128,
    filter_sizes=[2, 3, 4],
    num_filters=128,
    l2_reg_lambda=0)
out_textcnn = connect(cnn.h_pool_flat, 384, 64)

mid_vgg = 256
out_vgg_flat1 = tf.reshape(vgg19.Model(input_pic, 'model/vgg19/Vgg19.model').output, [-1, 7 * 7 * 512])
out_vgg = connect(out_vgg_flat1, 7*7*512, mid_vgg)
out_vgg = connect(out_vgg, mid_vgg, 128)

out_all = tf.concat((out_gru, out_textcnn, out_vgg), axis=1)

mid_unit = 64
logits1 = connect(out_all, 256, mid_unit)
logits1 = tf.nn.dropout(connect(logits1, mid_unit, num_labels), keep_prob=dropout_rate)
log_probs1 = tf.nn.log_softmax(logits1, axis=-1)
one_hot_labels1 = tf.one_hot(input_y, depth=num_labels, dtype=tf.float32)
loss1 = tf.reduce_mean(-tf.reduce_sum(one_hot_labels1 * log_probs1, axis=-1))
predict = tf.argmax(tf.nn.softmax(logits1), axis=1, name="predictions")
acc = tf.reduce_mean(tf.cast(tf.equal(input_y, tf.cast(predict, dtype=tf.int32)), "float"), name="accuracy")


gr = GradientReversal()
out_gr = gr(out_all, 1)
logits2 = connect(out_gr, 256, mid_unit)
logits2 = tf.nn.dropout(connect(logits2, mid_unit, event_labels), keep_prob=dropout_rate)
log_probs2 = tf.nn.log_softmax(logits2, axis=-1)
one_hot_labels2 = tf.one_hot(input_dis, depth=event_labels, dtype=tf.float32)
loss2 = tf.reduce_mean(-tf.reduce_sum(one_hot_labels2 * log_probs2, axis=-1))

loss = tf.add(loss1, loss2)

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


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
            shuff = np.random.permutation(range(label_train.shape[0]))[:batch_size]
            batch_input = x_train[shuff]
            batch_labels = label_train[shuff]
            batch_event = events_train[shuff]
            batch_pic = pic_train[shuff]
            #print(batch_labels)
            #print(batch_event)
            #print(batch_pic)
            batch_pic = get_path_images(batch_pic)

            l, a, _=sess.run([loss, acc,  train_op],feed_dict={
                input_x:batch_input,
                input_y:batch_labels,
                input_dis:batch_event,
                input_pic:batch_pic,
                dropout_rate:1
            })
            '''l=sess.run([loss],feed_dict={
                input_x:batch_input,
                input_y:batch_labels,
                input_dis:batch_event,
                input_pic:batch_pic,
                dropout_rate:1
            })'''
            #print("{} {}".format(l[0].shape, l))
            print("{}, acc:{}, loss:{}".format(current_step, a, l))

            if (current_step + 1) % 100 == 0:
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
                    #print(batch_labels)
                    #print(batch_event)
                    #print(batch_pic)
                    batch_pic = get_path_images(batch_pic)
                    l, a=sess.run([loss1, acc],feed_dict={
                        input_x:batch_input,
                        input_y:batch_labels,
                        input_pic:batch_pic,
                        dropout_rate:1
                    })
                    li += l
                    ai += a
                    print("{}, acc:{}, loss:{}".format(cs, a, l))
                acc_all.append(ai/epoch)
                loss_all.append(li/epoch)
                print("acc:{}, loss:{}".format(ai/epoch, li/epoch))
            if (current_step + 1) % 1000 == 0:
                path = saver.save(sess, "./model/mymodel/mymodel", global_step=current_step+1)
                print("Saved model checkpoint to {}\n".format(path))

                
print(acc_all)
print(loss_all)