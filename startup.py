from model.featureExtraction import tfidf_word_list

import gensim, smart_open
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import numpy as np
from sklearn.externals import joblib
from gensim.models import word2vec

import data.atis_load as load

from gensim import corpora, models, similarities
import gensim

import pickle
from bilstm_crf import LSTM_CRF as used_model
import tensorflow as tf

from urllib.request import urlretrieve


def mysen(neg_tf_idf1):
    yield neg_tf_idf1


if __name__ == '__main__':
    neg_file_path = './data/neg.txt'
    pos_file_path = './data/pos.txt'

    '''line_len = 400
    best_word_num = 300'atis.pkl'
    sample_num = 3000
    train_num = int(sample_num * 0.8)

    # term_frequency(neg_file_path, pos_file_path, best_word_num, sample_num, train_num)

    neg_tf_idf = tfidf_word_list(neg_file_path, 2, line_len)
    print(neg_tf_idf)

    pos_tf_idf = tfidf_word_list(pos_file_path, 2, line_len)
    print(pos_tf_idf)

    # sentences = mysen(neg_tf_idf[0])
    model = word2vec.Word2Vec(size=100, min_count=1)
    model.build_vocab(neg_tf_idf)
    model.train(neg_tf_idf, total_examples=model.corpus_count, epochs=model.epochs)

    print(model['标准间'])
    print(model['陈旧'])

    print(model['标准间'] + model['陈旧'])
    '''

    f = open('./data/atis.pkl', 'rb')

    try:
        train_set, test_set, dicts = pickle.load(f, encoding='utf8')
    except UnicodeDecodeError:
        print("exception when get atis data!!!")
        train_set, test_set, dicts = pickle.load(f, encoding='latin1')

    # train_set, valid_set, dicts = load.atisfull()
    w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']

    # print(w2idx)
    print(len(w2idx))

    train_x, _, train_label = train_set
    val_x, _, val_label = test_set

    # Create index to word/label dicts
    idx2w = {w2idx[k]: k for k in w2idx}
    idx2ne = {ne2idx[k]: k for k in ne2idx}
    idx2la = {labels2idx[k]: k for k in labels2idx}

    # For conlleval script
    words_train = [list(map(lambda x: idx2w[x], w)) for w in train_x]
    labels_train = [list(map(lambda x: idx2la[x], y)) for y in train_label]
    words_val = [list(map(lambda x: idx2w[x], w)) for w in val_x]
    labels_val = [list(map(lambda x: idx2la[x], y)) for y in val_label]

    # data = zip(train_x, train_label)

    print(train_x[0])
    print(train_label[0])

    print("-------")
    print(words_val[0])
    print(labels_val[0])

    print("--------")
    print(val_x[0])
    print(val_label[0])

    # print("Example sentence : {}".format(words_train[0]))
    print("train data Encoded form: {}".format(train_x[0]))
    print("label data Encoded form: {}".format(train_label[0]))

    print()
    print("It's label : {}".format(labels_train[0]))

    num_words = len(w2idx)
    embedding_dim = 100

    embedding_mat = np.random.uniform(-0.25, 0.25, (num_words, embedding_dim))
    embedding_mat = np.float32(embedding_mat)

    path_dic = {}
    path_dic['summary_path'] = './saver/summary'
    path_dic['model_path'] = './saver/model'
    path_dic['result_path'] = './saver/model'

    batch_size = 128
    hidden_dim = 100
    embedding_dim_update = embedding_mat
    tag2label = labels2idx
    vocab = w2idx
    paths = path_dic
    epoch_num = 20
    lr = 0.01

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2

    # init model
    model = used_model(num_words, batch_size, epoch_num, hidden_dim, embedding_dim, embedding_dim_update, tag2label, vocab, paths, config, lr)

    # model generator
    model.get_graph()

    print("train data: {}".format(len(train_x)))
    print("valid data: {}".format(len(val_x)))
    model.train(train_x, train_label, val_x, val_label)

