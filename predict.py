#!/usr/bin/python
# -*- coding: utf-8 -*-

from model1 import *
from cnews import *
from sklearn import metrics
import numpy as np
import time
import os, re
from datetime import timedelta
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--base_dir", type=str, default="data/")
parser.add_argument("--test_dir", type=str, default="data/dev.tsv")
parser.add_argument("--test_result_dir", type=str, default="data/dev_result")
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--seq_length", type=int, default=128)
parser.add_argument("--num_filters", type=int, default=1)
parser.add_argument("--filter_sizes", type=str, default="3")
parser.add_argument("--vocab_size", type=int, default=5000)
parser.add_argument("--hidden_dim", type=int, default=768)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)

FLAGS = parser.parse_args()

base_dir = FLAGS.base_dir
test_dir = FLAGS.test_dir
vocab_dir = os.path.join(base_dir, "vocab.txt")
label_dir = os.path.join(base_dir, "label.txt")

save_path = os.path.join(base_dir, "textcnn/model.ckpt")
embedding_dim = FLAGS.embedding_dim
seq_length = FLAGS.seq_length
num_filters = FLAGS.num_filters
filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
vocab_size = FLAGS.vocab_size
hidden_dim = FLAGS.hidden_dim
learning_rate = FLAGS.learning_rate
batch_size = FLAGS.batch_size

test_dir = FLAGS.test_dir
test_result_dir = FLAGS.test_result_dir

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def read_input(file):
    contents, labels = [], []
    with open(file, 'r') as f:
        for line in f:
            try:
                line = line.strip().split('\t')
                if len(line) == 3:  # char+word+label
                    char = line[0].strip()
                    label = line[-1].strip()
                if len(line) == 2:  # char+label
                    char = line[0].strip()
                    label = line[-1].strip()
                contents.append(char)
                labels.append(label)
            except:
                pass
    return contents, labels

def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, batch_size)
    total_loss = 0.0
    total_acc = 0.0
    total_rec = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc, rec = sess.run([model.loss, model.acc, model.rec], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
        total_rec += rec * batch_len

    return total_loss / data_len, total_acc / data_len, total_rec / data_len

def tes():
    print("Loading test data...")
    inputs, labels = read_input(test_dir)
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, seq_length, num_classes)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test, rec_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
    y_pred_soft = [0]*len(x_test)
    for i in range(num_batch):   # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id], soft = session.run([model.y_pred_cls, model.soft], feed_dict=feed_dict)
        for j in range(end_id-start_id):
            y_pred_soft[j+start_id] = soft[j]
    f = open(test_result_dir, 'w')

    print (len(y_pred_cls), len(inputs), len(labels))
    for m in range(len(y_pred_cls)):
        f.write(inputs[m] + "\t" + labels[m] + "\t")
        scores = []
        max_label = ""
        max_score = 0.0
        ind = -1
        for n in y_pred_soft[m]:
            ind += 1
            if id_to_cat[ind] == "negative":
                continue
            scores.append(str(n))
            if n > max_score:
                max_score = n
                max_label = id_to_cat[ind]
        f.write(max_label + "\t")
        for i in scores:
            f.write(i + " ")
        f.write("\n")
    f.close()

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories, digits=4))

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    print('Configuring CNN model...')

    categories, cat_to_id, id_to_cat = read_category(label_dir)
    num_classes = len(cat_to_id)
    words, word_to_id = read_vocab(vocab_dir)
    model = TextCNN(seq_length, num_classes, vocab_size, embedding_dim, filter_sizes, num_filters, hidden_dim,
                    learning_rate)

    tes()
