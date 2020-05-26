import textcnn.data_input_helper as data_helpers
from utils.vocab_trans import get_vocab_trans
from utils.data_slice import *

max_seq_length = 100
def load_data():
    _, y, events, pic, train_index, test_index = data_helpers.load_all_info("data/train.csv")
    #print(x_text.shape)

    vocab_size, _, x = get_vocab_trans(vocabpath='utils/vocab.txt', transpath='utils/trans.txt')
    x = x[:, :max_seq_length ]
    print(x.shape)

    print(vocab_size)
    #pics = get_path_images(pic)
    #print(pics[:, 0:1, 0:1, :])

    '''np.random.seed(10)
    train_index = get_train_index()'''
    x_train = x[train_index]
    y_train = y[train_index]
    events_train = events[train_index]
    pic_train = pic[train_index]

    x_test = x[test_index]
    y_test = y[test_index]
    events_test = events[test_index]
    pic_test = pic[test_index]
    #print(len(events_train))
    #print(len(events_test))
    #print(x_train[0], y_train[0], events_train[0], pic_train[0])
    #return x,y,vocab_size,events,pic
    return x_train, y_train, vocab_size, events_train, pic_train, x_test, y_test, events_test, pic_test

def batch_iter(batch_size, data_size):
    epoch = int((data_size-1)/batch_size) + 1
    for i in range(epoch):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_size)
        yield [start_index, end_index]