#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

class TextCNN(object):
    def __init__(self, seq_length, num_classes, vocab_size, embedding_dim, filter_sizes, num_filters, hidden_dim, learning_rate):
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def _auc_pr(self, true, prob, threshold):
        pred = tf.where(prob > threshold, tf.ones_like(prob), tf.zeros_like(prob))
        tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(true, tf.bool))
        fp = tf.logical_and(tf.cast(pred, tf.bool), tf.logical_not(tf.cast(true, tf.bool)))
        fn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)), tf.cast(true, tf.bool))
        pre = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fp), tf.int32)))
        rec = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fn), tf.int32)))
        return pre, rec

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        self.time0 = tf.timestamp(name='tstamp0')
        self.pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                filter_shape = [filter_size, self.embedding_dim, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                self.conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                print ("conv", str(self.conv))
                h = tf.nn.relu(tf.nn.bias_add(self.conv, b), name="relu")
                # Maxpooling over the outputs
                self.pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print ("pooled", str(self.pooled))
                self.pooled_outputs.append(self.pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(self.pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            self.fc = tf.layers.dense(self.h_pool_flat, self.hidden_dim, name='fc1')
            self.fc = tf.nn.dropout(self.fc, self.keep_prob)
            self.fc = tf.nn.relu(self.fc)

            # 分类器
            self.logits = tf.layers.dense(self.fc, self.num_classes, name='fc2')
            self.soft = tf.nn.softmax(self.logits, name="my_output")
            self.soft_round = tf.round(self.soft)
            self.y_pred_soft = tf.reduce_max(self.soft, 1, name='max_value')
            self.y_pred_cls = tf.argmax(self.soft, 1, name='predict')  # 预测类别
        self.time1 = tf.timestamp(name='tstamp1')
        print ()

        self.time2 = tf.timestamp(name='tstamp2')
        self.time3 = tf.timestamp(name='tstamp3')
        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            self.acc, self.rec = self._auc_pr(self.input_y, self.soft, 0.1)



