import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import progressbar

# from eval import conlleval

import data.atis_load as load

# from metrics.accuracy import conlleval

from sklearn.metrics import classification_report

from tensorflow.contrib.rnn import LSTMCell

class LSTM_CRF(object):
    def __init__(self, num_words, batch_size, epoch_num, hidden_dim, embedding_dim, embedding_dim_update, tag2label, vocab, paths, config, lr):
        self.epoch_num = epoch_num
        self.num_words = num_words
        self.embedding_dim = embedding_dim
        self.updatedembedding = embedding_dim_update
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        # self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config
        self.n_hidden = hidden_dim
        self.batch_size = batch_size
        self.lr = lr
        self.dropout_r = 0.3
        self.num_layers = 1

    def get_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

    def model_generator(self):

        # fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        # bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        fw_cell = LSTMCell(self.n_hidden)
        bw_cell = LSTMCell(self.n_hidden)

        # if is_training:
        #    fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=(1 - dropout_rate))
        #    bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=(1 - dropout_rate))

        # fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * self.num_layers)
        # bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * self.num_layers)

        # (output_fw_seq, output_bw_seq), fw_state, bw_state = \
        #    tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, self.word_embedding, dtype=tf.float32,
        #                                            sequence_length=self.sentence_len)

        (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                            cell_bw=bw_cell,
                                                                            inputs=self.word_embedding,
                                                                            sequence_length=self.sentence_len,
                                                                            dtype=tf.float32)

        output_lstm = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        matric_output = tf.reshape(output_lstm, [-1, 2 * self.n_hidden])

        weight = tf.Variable(tf.random_normal([2 * self.n_hidden, self.num_tags]))
        bias = tf.Variable(tf.random_normal([self.num_tags]))

        matricized_unary_scores = tf.matmul(matric_output, weight) + bias

        ntime_steps = tf.shape(output_lstm)[1]
        self.logits = tf.reshape(matricized_unary_scores, [-1, ntime_steps, self.num_tags])

    def cost_function(self):
        # cost函数
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.y, self.sentence_len)

        self.transition_params = transition_params
        self.cost = tf.reduce_mean(-log_likelihood)

        self.decode_tags, self.best_score = tf.contrib.crf.crf_decode(self.logits, transition_params, self.sentence_len)

    def get_embedding(self):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            w_embedding = tf.Variable(tf.random_uniform([self.num_words, self.embedding_dim], -1.0, 1.0),
                                      name="W_embedding")
            self.word_embedding = tf.nn.embedding_lookup(w_embedding, self.x, name="word_embedding")
            # self.word_embedding = tf.expand_dims(embedded_chars, -1)

    def get_placeholder(self):
        self.x = tf.placeholder(tf.int32, shape=[None, None], name="x")
        self.y = tf.placeholder(tf.int32, shape=[None, None], name="y")
        self.sentence_len = tf.placeholder(tf.int32, shape=[None], name="sentence_len")

        self.dropout_rate = tf.placeholder(tf.float32, shape=[], name="dropout_rate")
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")

    def get_graph(self):
        # get placeholder
        self.get_placeholder()

        # 获取词向量
        self.get_embedding()

        # 生成深度学习模型
        self.model_generator()

        # 损失函数
        self.cost_function()

        # optimizer
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.init_op = tf.global_variables_initializer()

        return

    def run_one_epoch(self, sess, train_x, train_label, epoch, saver):

        batches = self.batch_yield(train_x, train_label, self.batch_size)
        num_batches = (len(train_x) + self.batch_size - 1) // self.batch_size
        for step, (seqs, labels) in enumerate(batches):
            # time used for log
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_r)

            _, loss_train = sess.run([self.train_op, self.cost], feed_dict=feed_dict)

            if step + 1 == 1 or (step + 1) % 20 == 0 or step + 1 == num_batches:
                print('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                  loss_train, step_num))
            # self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                print("~~ save model file ~~" + self.model_path)
                saver.save(sess, self.model_path, global_step=step_num)

    def train(self, train_x, train_label, val_x, val_y):
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            # self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train_x, train_label, epoch, saver)

            print('===========validation / test===========')
            print()

            label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, val_x, val_y)

            # print(tf.shape(label_list_dev))
            print(label_list_dev[0])

            # data, label = dev
            self.evaluate(label_list_dev, val_x, val_y, epoch)

    def batch_yield(self, x, y, batch_size):
        seqs, labels = [], []
        for (sent_, tag_) in zip(x, y):

            if len(seqs) == batch_size:
                yield seqs, labels
                seqs, labels = [], []

            seqs.append(sent_)
            labels.append(tag_)

        if len(seqs) != 0:
            yield seqs, labels

        return

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        # word_ids is padding one
        # len_list is the actual length
        word_ids, seq_len_list = self.pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.x: word_ids,
                     self.sentence_len: seq_len_list}

        if labels is not None:
            labels_, _ = self.pad_sequences(labels, pad_mark=0)
            feed_dict[self.y] = labels_

        if lr is not None:
            feed_dict[self.learning_rate] = lr

        if dropout is not None:
            feed_dict[self.dropout_rate] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, val_x, val_y):

        label_list, seq_len_list = [], []
        x = val_x
        y = val_y

        for seqs, labels in self.batch_yield(x, y, self.batch_size):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):

        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        # viterbi_seq, seq_len_list = sess.run([self.decode_tags, self.best_score], feed_dict=feed_dict)

        # get tag scores and transition params of CRF
        logits, trans_params = sess.run([self.logits, self.transition_params], feed_dict=feed_dict)

        # iterate over the sentences because no batching in vitervi_decode
        viterbi_sequences = []
        for logit, sequence_length in zip(logits, seq_len_list):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, seq_len_list

        # return viterbi_seq, seq_len_list

    def pad_sequences(self, input_data, pad_mark=0):

        # 返回一个序列中长度最长的那条样本的长度
        max_len = max(map(lambda x: len(x), input_data))
        seq_list, seq_len_list = [], []
        for seq in input_data:
            # 由元组格式()转化为列表格式[]
            seq = list(seq)

            # 不够最大长度的样本用0补上放到列表seq_list
            seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
            seq_list.append(seq_)

            # seq_len_list用来统计每个样本的真实长度
            seq_len_list.append(min(len(seq), max_len))

        return seq_list, seq_len_list

    def evaluate(self, label_list, val_x, val_y, epoch=None):

            index2tag = {self.tag2label[k]: k for k in self.tag2label}
            idx2w = {self.vocab[k]: k for k in self.vocab}

            data = zip(val_x, val_y)
            model_predict = []

            sent_la = []
            sent_pr = []
            for label_, (sent, tag) in zip(label_list, data):
                tag_ = [index2tag[label__] for label__ in label_]
                tag_r = [index2tag[tag__] for tag__ in tag]
                sent_ = [idx2w[sent__] for sent__ in sent]

                sent_res = []
                for i in range(len(sent)):
                    sent_res.append([sent_[i], tag_r[i], tag_[i]])
                    sent_la.append(label_[i])
                    sent_pr.append(tag[i])

                model_predict.append(sent_res)

            from sklearn.metrics import f1_score
            from sklearn.metrics import precision_score
            from sklearn.metrics import recall_score

            print("f1 score is " + str(f1_score(sent_la, sent_pr, average='weighted')))
            # print("precision is " + str(precision_score(sent_la, sent_pr, average=None)))
            print("precision is " + str(precision_score(sent_la, sent_pr, average='micro')))
            print("recall is " + str(recall_score(sent_la, sent_pr, average='weighted')))

            # print(classification_report(sent_la, sent_pr, target_names="test"))

            from sklearn import metrics
            confu_mat = metrics.confusion_matrix(sent_la, sent_pr)

            # import numpy as np
            # np.set_printoptions(threshold='nan')
            # print(np.array(confu_mat))

            epoch_num = str(epoch + 1)

            label_path = self.result_path + '_label_' + str(epoch_num)
            metric_path = self.result_path + '_result_metric_' + str(epoch_num)

            self.conlleval(model_predict, label_path, metric_path)

    def conlleval(self, label_predict, label_path, metric_path):

        # eval_perl = "./saver/bilstm_crf.txt"
        with open(label_path, "w") as fw:
            line = []
            for sent_result in label_predict:
                for char, tag, tag_ in sent_result:
                    # tag = tag.encode("utf-8")
                    # char = char.encode("utf-8")
                    line.append("{} {} {}\n".format(char, tag, tag_))
                line.append("\n")
            print(line)
            fw.writelines(line)
            fw.flush()
