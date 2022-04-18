<<<<<<< HEAD
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "8"

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from bert import modeling

from matric import report_performance, report_loss
from parameter_attention_weibo import *
from load_input_att import *


# 将激活函数更换为sigmoid函数

class TweetEncoder(object):
    def __init__(self):

        self.input_ids = tf.placeholder(shape=[batch_size, (1 + max_history_length), max_seq_length], dtype=tf.int32,
                                        name="input_ids")
        self.input_mask = tf.placeholder(shape=[batch_size, (1 + max_history_length), max_seq_length], dtype=tf.int32,
                                         name="input_mask")
        self.segment_ids = tf.placeholder(shape=[batch_size, (1 + max_history_length), max_seq_length], dtype=tf.int32,
                                          name="segment_ids")
        self.input_labels = tf.placeholder(shape=[batch_size, (1 + max_history_length), num_labels], dtype=tf.float32,
                                           name="input_labels")

        self.compress_dim()

        #  创建bert的输入
        self.model = self.bert_model()
        self.bert_output = self.get_bert_output()

        # 将当前tweet和history分离
        # self.tweet_emb: [batch_size, 1, hidden_size]
        # self.history_emb: [batch_size, max_history_length, hidden_size]
        self.tweet_emb, self.history_emb, self.tweet_label, self.history_label = self.expand_dim()

        # [batch_size, 1, hidden_size]
        self.attention_output = self.multihead_attention(self.tweet_emb, self.history_emb, self.history_label)

        # [batch_size, 1, hidden_size *2]
        self.output = tf.concat([self.tweet_emb, self.attention_output], axis=2)
        # [batch_size, hidden_size *2]
        self.output = tf.squeeze(self.output)

        self.predict = self.predict_score()
        self.loss = self.loss_function()
        self.train_op = self.define_train_op()
        self.load_bert_checkpoint()

    def compress_dim(self):
        self.input_ids_compress = tf.reshape(self.input_ids, [-1, max_seq_length])
        self.input_mask_compress = tf.reshape(self.input_mask, [-1, max_seq_length])
        self.segment_ids_compress = tf.reshape(self.segment_ids, [-1, max_seq_length])

    def bert_model(self):
        # 创建bert模型
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=self.input_ids_compress,
            input_mask=self.input_mask_compress,
            token_type_ids=self.segment_ids_compress,
            use_one_hot_embeddings=False  # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
        )
        print("init BERT parameters")
        return model

    def get_bert_output(self):
        return self.model.get_pooled_output()  # 这个获取句子的output

    def expand_dim(self):
        reshape_output = tf.reshape(self.bert_output, [batch_size, (1 + max_history_length), -1])
        # 这里 tweet_emb [batch_size, 1, num_labels], 不能使用 [:, 0, :]的切片方法
        tweet_emb = reshape_output[:, :1, :]
        history_emb = reshape_output[:, 1:, :]

        # 这里 tweet_label的维度是 [batch_size, num_labels], 不需要expand dim
        tweet_label = self.input_labels[:, 0, :]
        history_label = self.input_labels[:, 1:, :]
        return tweet_emb, history_emb, tweet_label, history_label

    def multihead_attention(self, queries, keys, values, num_units=None, num_output_units=None,
                            activation_fn=None, num_heads=8, keep_prob=0.8, is_training=True,
                            scope="multihead_attention", reuse=None):
        '''Applies multihead attention.
        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k == max_history_length, C_k].
          values: A 3d tensor with shape of [N, T_k, C_v == num_labels].  history tweet label
        Returns
          A 3d tensor with shape of (N, T_q, C_v)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.shape.as_list()[-1]

            if num_output_units is None:
                num_output_units = queries.shape.as_list()[-1]

            # (N, T_q, C)
            Q = layers.fully_connected(queries, num_units, activation_fn=activation_fn, scope="Q")
            K = layers.fully_connected(keys, num_units, activation_fn=activation_fn, scope="K")  # (N, T_k, C)
            V = layers.fully_connected(values, num_output_units, activation_fn=activation_fn, scope="V")  # (N, T_k, C)

            def split_last_dimension_then_transpose(tensor, num_heads):
                t_shape = tensor.get_shape().as_list()
                tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, t_shape[-1] // num_heads])
                return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, t_shape[-1]]

            Q_ = split_last_dimension_then_transpose(Q, num_heads)  # (h*N, T_q, C/h)
            K_ = split_last_dimension_then_transpose(K, num_heads)  # (h*N, T_k, C/h)
            V_ = split_last_dimension_then_transpose(V, num_heads)  # (h*N, T_k, Cv/h)

            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # [batch_size, num_heads, query_len, key_len]

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
            # Attention vector    att_vec = outputs
            # Dropouts
            outputs = layers.dropout(outputs, keep_prob=keep_prob, is_training=is_training)
            # Weighted sum (N, h, T_q, T_k) * (N, h, T_k, Cv/h)
            outputs = tf.matmul(outputs, V_)  # (N, h, T_q, Cv/h)

            def transpose_then_concat_last_two_dimenstion(tensor):
                tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
                t_shape = tensor.get_shape().as_list()
                num_heads, dim = t_shape[-2:]
                return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

            outputs = transpose_then_concat_last_two_dimenstion(outputs)  # (N, T_q, Cv)
            return outputs

    def predict_score(self):
        # TODO: 此处的 dropout_output 是否有必要
        dropout_output = layers.dropout(inputs=self.output, keep_prob=0.9, is_training=is_training)
        predict = layers.fully_connected(dropout_output, num_labels, activation_fn=tf.sigmoid)
        return predict

    def loss_function(self):
        # 计算loss函数
        print("one_hot_labels shape:" + str(self.tweet_label.shape))
        return tf.keras.losses.binary_crossentropy(self.tweet_label, self.predict)

    def define_train_op(self):
        return tf.train.AdamOptimizer(lr).minimize(self.loss)

    def load_bert_checkpoint(self):
        # 获取模型中所有的训练参数。
        tvars = tf.trainable_variables()
        # 加载BERT模型
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        print("**** Trainable Variables ****")
        # 打印加载模型的参数
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            print("name = %s, shape = %s%s" % (var.name, var.shape, init_string))

    def run_train(self, epoch_i, session):
        print("epoch %d start train" % epoch_i)
        start = 0
        while start + batch_size < train_texts_len:
            shuffle_index = shuffle_index_array[start:start + batch_size]
            batch_labels = labels[shuffle_index]
            batch_input_idsList = input_idsList[shuffle_index]
            batch_input_masksList = input_masksList[shuffle_index]
            batch_segment_idsList = segment_idsList[shuffle_index]

            l, batch_pred, _ = session.run([self.loss, self.predict, self.train_op], feed_dict={
                self.input_ids: batch_input_idsList, self.input_mask: batch_input_masksList,
                self.segment_ids: batch_segment_idsList, self.input_labels: batch_labels})

            if start == 0:
                prediction = batch_pred
                train_y = batch_labels
                train_loss = l
            else:
                prediction = np.concatenate((prediction, batch_pred), axis=0)
                train_y = np.concatenate((train_y, batch_labels), axis=0)
                train_loss = np.concatenate((train_loss, l), axis=0)
            start = start + batch_size
        return train_y, prediction, train_loss

    def run_val(self, epoch_i, session, labels, input_idsList, input_masksList, segment_idsList, texts_len):
        start = 0
        print("epoch %d start test" % epoch_i)
        while start + batch_size < texts_len:
            batch_labels = labels[start: start + batch_size]
            batch_input_idsList = input_idsList[start: start + batch_size]
            batch_input_masksList = input_masksList[start: start + batch_size]
            batch_segment_idsList = segment_idsList[start: start + batch_size]

            l, batch_pred = session.run([self.loss, self.predict], feed_dict={
                self.input_ids: batch_input_idsList, self.input_mask: batch_input_masksList,
                self.segment_ids: batch_segment_idsList, self.input_labels: batch_labels})
            if start == 0:
                test_prediction = batch_pred
                test_y = batch_labels
                test_loss = l
            else:
                test_prediction = np.concatenate((test_prediction, batch_pred), axis=0)
                test_y = np.concatenate((test_y, batch_labels), axis=0)
                test_loss = np.concatenate((test_loss, l), axis=0)
            start = start + batch_size
        return test_y, test_prediction, test_loss


# TODO: 需要将history和 tweet 融合后输入给模型
# load data, training set

tweetEncoder = TweetEncoder()

ids = load_ids(train_data_type)
# 记录未融合history的数据，下文中构建 test 和 val +数据需要使用
origin_input_idsList, origin_input_masksList, origin_segment_idsList, origin_one_hot_labels, train_texts_len = \
    process_train_data(train_data_type)
input_idsList, input_masksList, segment_idsList, labels \
    = process_train_history_data(train_data_type, ids, origin_input_idsList, origin_input_masksList,
                                 origin_segment_idsList, origin_one_hot_labels)

# 由于 index==0 的位置存放的是补零的空sample
shuffle_index_array = np.random.permutation(np.arange(1, train_texts_len))

val_ids = load_ids(val_data_type)
# 训练集和验证集中， sequence的样本应当全部来源于训练集，因此这里的 val_history_list 中的index是训练集数据的 index
val_input_idsList, val_input_masksList, val_segment_idsList, val_labels, val_texts_len \
    = process_test_val_history_data(val_data_type, val_ids, ids, origin_input_idsList, origin_input_masksList,
                                    origin_segment_idsList, origin_one_hot_labels)

test_ids = load_ids(test_data_type)
test_input_idsList, test_input_masksList, test_segment_idsList, test_labels, test_texts_len \
    = process_test_val_history_data(test_data_type, test_ids, ids, origin_input_idsList,
                                    origin_input_masksList, origin_segment_idsList, origin_one_hot_labels)

last_iter_loss = float("inf")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iter_num):
        is_training = True
        train_y, prediction, train_loss = tweetEncoder.run_train(i, sess)
        train_loss = np.sum(train_loss)
        report_loss(i, train_loss)

        # 开始验证：
        is_training = False
        val_y, val_prediction, val_loss = \
            tweetEncoder.run_val(i, sess, val_labels, val_input_idsList, val_input_masksList, val_segment_idsList,
                                 val_texts_len)

        val_loss = np.sum(val_loss)
        report_performance(val_y[:, 0, :], val_prediction, val_loss, "val result")

        # 开始测试：
        test_y, test_prediction, test_loss = \
            tweetEncoder.run_val(i, sess, test_labels, test_input_idsList, test_input_masksList, test_segment_idsList,
                                 test_texts_len)
        test_loss = np.sum(test_loss)
        report_performance(test_y[:, 0, :], test_prediction, test_loss, "test result")

        save_path = "./checkpoint/weibo_history/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver = tf.train.Saver()
        saver.save(sess, save_path, global_step=i)
        print("save model in :" + save_path)

        # 训练终止条件
        if train_loss > last_iter_loss:
            break
        last_iter_loss = train_loss
=======
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "8"

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from bert import modeling

from matric import report_performance, report_loss
from parameter_attention_weibo import *
from load_input_att import *


# 将激活函数更换为sigmoid函数

class TweetEncoder(object):
    def __init__(self):

        self.input_ids = tf.placeholder(shape=[batch_size, (1 + max_history_length), max_seq_length], dtype=tf.int32,
                                        name="input_ids")
        self.input_mask = tf.placeholder(shape=[batch_size, (1 + max_history_length), max_seq_length], dtype=tf.int32,
                                         name="input_mask")
        self.segment_ids = tf.placeholder(shape=[batch_size, (1 + max_history_length), max_seq_length], dtype=tf.int32,
                                          name="segment_ids")
        self.input_labels = tf.placeholder(shape=[batch_size, (1 + max_history_length), num_labels], dtype=tf.float32,
                                           name="input_labels")

        self.compress_dim()

        #  创建bert的输入
        self.model = self.bert_model()
        self.bert_output = self.get_bert_output()

        # 将当前tweet和history分离
        # self.tweet_emb: [batch_size, 1, hidden_size]
        # self.history_emb: [batch_size, max_history_length, hidden_size]
        self.tweet_emb, self.history_emb, self.tweet_label, self.history_label = self.expand_dim()

        # [batch_size, 1, hidden_size]
        self.attention_output = self.multihead_attention(self.tweet_emb, self.history_emb, self.history_label)

        # [batch_size, 1, hidden_size *2]
        self.output = tf.concat([self.tweet_emb, self.attention_output], axis=2)
        # [batch_size, hidden_size *2]
        self.output = tf.squeeze(self.output)

        self.predict = self.predict_score()
        self.loss = self.loss_function()
        self.train_op = self.define_train_op()
        self.load_bert_checkpoint()

    def compress_dim(self):
        self.input_ids_compress = tf.reshape(self.input_ids, [-1, max_seq_length])
        self.input_mask_compress = tf.reshape(self.input_mask, [-1, max_seq_length])
        self.segment_ids_compress = tf.reshape(self.segment_ids, [-1, max_seq_length])

    def bert_model(self):
        # 创建bert模型
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=self.input_ids_compress,
            input_mask=self.input_mask_compress,
            token_type_ids=self.segment_ids_compress,
            use_one_hot_embeddings=False  # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
        )
        print("init BERT parameters")
        return model

    def get_bert_output(self):
        return self.model.get_pooled_output()  # 这个获取句子的output

    def expand_dim(self):
        reshape_output = tf.reshape(self.bert_output, [batch_size, (1 + max_history_length), -1])
        # 这里 tweet_emb [batch_size, 1, num_labels], 不能使用 [:, 0, :]的切片方法
        tweet_emb = reshape_output[:, :1, :]
        history_emb = reshape_output[:, 1:, :]

        # 这里 tweet_label的维度是 [batch_size, num_labels], 不需要expand dim
        tweet_label = self.input_labels[:, 0, :]
        history_label = self.input_labels[:, 1:, :]
        return tweet_emb, history_emb, tweet_label, history_label

    def multihead_attention(self, queries, keys, values, num_units=None, num_output_units=None,
                            activation_fn=None, num_heads=8, keep_prob=0.8, is_training=True,
                            scope="multihead_attention", reuse=None):
        '''Applies multihead attention.
        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k == max_history_length, C_k].
          values: A 3d tensor with shape of [N, T_k, C_v == num_labels].  history tweet label
        Returns
          A 3d tensor with shape of (N, T_q, C_v)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.shape.as_list()[-1]

            if num_output_units is None:
                num_output_units = queries.shape.as_list()[-1]

            # (N, T_q, C)
            Q = layers.fully_connected(queries, num_units, activation_fn=activation_fn, scope="Q")
            K = layers.fully_connected(keys, num_units, activation_fn=activation_fn, scope="K")  # (N, T_k, C)
            V = layers.fully_connected(values, num_output_units, activation_fn=activation_fn, scope="V")  # (N, T_k, C)

            def split_last_dimension_then_transpose(tensor, num_heads):
                t_shape = tensor.get_shape().as_list()
                tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, t_shape[-1] // num_heads])
                return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, t_shape[-1]]

            Q_ = split_last_dimension_then_transpose(Q, num_heads)  # (h*N, T_q, C/h)
            K_ = split_last_dimension_then_transpose(K, num_heads)  # (h*N, T_k, C/h)
            V_ = split_last_dimension_then_transpose(V, num_heads)  # (h*N, T_k, Cv/h)

            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # [batch_size, num_heads, query_len, key_len]

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
            # Attention vector    att_vec = outputs
            # Dropouts
            outputs = layers.dropout(outputs, keep_prob=keep_prob, is_training=is_training)
            # Weighted sum (N, h, T_q, T_k) * (N, h, T_k, Cv/h)
            outputs = tf.matmul(outputs, V_)  # (N, h, T_q, Cv/h)

            def transpose_then_concat_last_two_dimenstion(tensor):
                tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
                t_shape = tensor.get_shape().as_list()
                num_heads, dim = t_shape[-2:]
                return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

            outputs = transpose_then_concat_last_two_dimenstion(outputs)  # (N, T_q, Cv)
            return outputs

    def predict_score(self):
        # TODO: 此处的 dropout_output 是否有必要
        dropout_output = layers.dropout(inputs=self.output, keep_prob=0.9, is_training=is_training)
        predict = layers.fully_connected(dropout_output, num_labels, activation_fn=tf.sigmoid)
        return predict

    def loss_function(self):
        # 计算loss函数
        print("one_hot_labels shape:" + str(self.tweet_label.shape))
        return tf.keras.losses.binary_crossentropy(self.tweet_label, self.predict)

    def define_train_op(self):
        return tf.train.AdamOptimizer(lr).minimize(self.loss)

    def load_bert_checkpoint(self):
        # 获取模型中所有的训练参数。
        tvars = tf.trainable_variables()
        # 加载BERT模型
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        print("**** Trainable Variables ****")
        # 打印加载模型的参数
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            print("name = %s, shape = %s%s" % (var.name, var.shape, init_string))

    def run_train(self, epoch_i, session):
        print("epoch %d start train" % epoch_i)
        start = 0
        while start + batch_size < train_texts_len:
            shuffle_index = shuffle_index_array[start:start + batch_size]
            batch_labels = labels[shuffle_index]
            batch_input_idsList = input_idsList[shuffle_index]
            batch_input_masksList = input_masksList[shuffle_index]
            batch_segment_idsList = segment_idsList[shuffle_index]

            l, batch_pred, _ = session.run([self.loss, self.predict, self.train_op], feed_dict={
                self.input_ids: batch_input_idsList, self.input_mask: batch_input_masksList,
                self.segment_ids: batch_segment_idsList, self.input_labels: batch_labels})

            if start == 0:
                prediction = batch_pred
                train_y = batch_labels
                train_loss = l
            else:
                prediction = np.concatenate((prediction, batch_pred), axis=0)
                train_y = np.concatenate((train_y, batch_labels), axis=0)
                train_loss = np.concatenate((train_loss, l), axis=0)
            start = start + batch_size
        return train_y, prediction, train_loss

    def run_val(self, epoch_i, session, labels, input_idsList, input_masksList, segment_idsList, texts_len):
        start = 0
        print("epoch %d start test" % epoch_i)
        while start + batch_size < texts_len:
            batch_labels = labels[start: start + batch_size]
            batch_input_idsList = input_idsList[start: start + batch_size]
            batch_input_masksList = input_masksList[start: start + batch_size]
            batch_segment_idsList = segment_idsList[start: start + batch_size]

            l, batch_pred = session.run([self.loss, self.predict], feed_dict={
                self.input_ids: batch_input_idsList, self.input_mask: batch_input_masksList,
                self.segment_ids: batch_segment_idsList, self.input_labels: batch_labels})
            if start == 0:
                test_prediction = batch_pred
                test_y = batch_labels
                test_loss = l
            else:
                test_prediction = np.concatenate((test_prediction, batch_pred), axis=0)
                test_y = np.concatenate((test_y, batch_labels), axis=0)
                test_loss = np.concatenate((test_loss, l), axis=0)
            start = start + batch_size
        return test_y, test_prediction, test_loss


# TODO: 需要将history和 tweet 融合后输入给模型
# load data, training set

tweetEncoder = TweetEncoder()

ids = load_ids(train_data_type)
# 记录未融合history的数据，下文中构建 test 和 val +数据需要使用
origin_input_idsList, origin_input_masksList, origin_segment_idsList, origin_one_hot_labels, train_texts_len = \
    process_train_data(train_data_type)
input_idsList, input_masksList, segment_idsList, labels \
    = process_train_history_data(train_data_type, ids, origin_input_idsList, origin_input_masksList,
                                 origin_segment_idsList, origin_one_hot_labels)

# 由于 index==0 的位置存放的是补零的空sample
shuffle_index_array = np.random.permutation(np.arange(1, train_texts_len))

val_ids = load_ids(val_data_type)
# 训练集和验证集中， sequence的样本应当全部来源于训练集，因此这里的 val_history_list 中的index是训练集数据的 index
val_input_idsList, val_input_masksList, val_segment_idsList, val_labels, val_texts_len \
    = process_test_val_history_data(val_data_type, val_ids, ids, origin_input_idsList, origin_input_masksList,
                                    origin_segment_idsList, origin_one_hot_labels)

test_ids = load_ids(test_data_type)
test_input_idsList, test_input_masksList, test_segment_idsList, test_labels, test_texts_len \
    = process_test_val_history_data(test_data_type, test_ids, ids, origin_input_idsList,
                                    origin_input_masksList, origin_segment_idsList, origin_one_hot_labels)

last_iter_loss = float("inf")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iter_num):
        is_training = True
        train_y, prediction, train_loss = tweetEncoder.run_train(i, sess)
        train_loss = np.sum(train_loss)
        report_loss(i, train_loss)

        # 开始验证：
        is_training = False
        val_y, val_prediction, val_loss = \
            tweetEncoder.run_val(i, sess, val_labels, val_input_idsList, val_input_masksList, val_segment_idsList,
                                 val_texts_len)

        val_loss = np.sum(val_loss)
        report_performance(val_y[:, 0, :], val_prediction, val_loss, "val result")

        # 开始测试：
        test_y, test_prediction, test_loss = \
            tweetEncoder.run_val(i, sess, test_labels, test_input_idsList, test_input_masksList, test_segment_idsList,
                                 test_texts_len)
        test_loss = np.sum(test_loss)
        report_performance(test_y[:, 0, :], test_prediction, test_loss, "test result")

        save_path = "./checkpoint/weibo_history/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver = tf.train.Saver()
        saver.save(sess, save_path, global_step=i)
        print("save model in :" + save_path)

        # 训练终止条件
        if train_loss > last_iter_loss:
            break
        last_iter_loss = train_loss
>>>>>>> fad0c2c7d5f1bd28b868c1169bc7a25da0b518c5
