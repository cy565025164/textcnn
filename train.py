#!/usr/bin/python
# -*- coding: utf-8 -*-

from model1 import *
from cnews import *
import os, time
import numpy as np
from datetime import timedelta
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import argparse
from tensorflow.python.client import timeline

parser = argparse.ArgumentParser()

parser.add_argument("--base_dir", type=str, default="data/")
parser.add_argument("--train_dir", type=str, default="data/train.tsv")
parser.add_argument("--dev_dir", type=str, default="data/dev.tsv")
parser.add_argument("--embedding_dim", type=int, default=768)
parser.add_argument("--seq_length", type=int, default=128)
parser.add_argument("--num_filters", type=int, default=768)
parser.add_argument("--filter_sizes", type=str, default="3")
parser.add_argument("--vocab_size", type=int, default=5000)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--dropout_keep_prob", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--print_per_batch", type=int, default=1000)
parser.add_argument("--save_per_batch", type=int, default=10)
FLAGS = parser.parse_args()

base_dir = FLAGS.base_dir
train_dir = FLAGS.train_dir
dev_dir = FLAGS.dev_dir
embedding_dim = FLAGS.embedding_dim
seq_length = FLAGS.seq_length
num_filters = FLAGS.num_filters
filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
vocab_size = FLAGS.vocab_size
hidden_dim = FLAGS.hidden_dim
dropout_keep_prob = FLAGS.dropout_keep_prob
learning_rate = FLAGS.learning_rate
batch_size = FLAGS.batch_size
num_epochs = FLAGS.num_epochs
print_per_batch = FLAGS.print_per_batch
save_per_batch = FLAGS.save_per_batch

vocab_dir = os.path.join(base_dir, 'vocab.txt')
label_dir = os.path.join(base_dir, 'label.txt')
id_to_cat_dir = os.path.join(base_dir, "id.to.cat")
vocab_dict_dir = os.path.join(base_dir, 'vocab.dict')
save_dir = os.path.join(base_dir, 'textcnn')
save_path = os.path.join(save_dir, 'model.ckpt')   # 最佳验证结果保存路径
export_dir = os.path.join(save_dir, 'load')

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

def train_test_split(input_x, input_y, train_size=0.8):
    input_y_len = len(input_y)
    tlen = int(input_y_len*train_size)
    shuffle_indices = np.random.permutation(np.arange(input_y_len))

    x_shuffled = input_x[shuffle_indices]
    y_shuffled = input_y[shuffle_indices]

    train_x = x_shuffled[:tlen]
    test_x = x_shuffled[tlen:]

    train_y = y_shuffled[:tlen]
    test_y = y_shuffled[tlen:]

    return train_x, train_y, test_x, test_y

def train():
    print("Configuring Saver...")
    writer = tf.summary.FileWriter(save_dir)

    # 配置 Saver
    saver = tf.train.Saver()

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length, num_classes)
    x_val, y_val = process_file(dev_dir, word_to_id, cat_to_id, seq_length, num_classes)
    #x_train, y_train, x_val, y_val = train_test_split(x_train, y_train)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0              # 总批次
    min_loss = 100.0
    for epoch in range(num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, dropout_keep_prob)
            if total_batch % print_per_batch == 0:

                ######
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                ######

                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train, rec_train = session.run([model.loss, model.acc, model.rec], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

                loss_val, acc_val, rec_val = evaluate(session, x_val, y_val)
                f1_train = 2 * acc_train * rec_train / (acc_train + rec_train)
                f1_val = 2 * acc_val * rec_val / (acc_val + rec_val)
                if loss_val < min_loss:
                    # 保存最好结果
                    min_loss = loss_val
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                    #proto
                    output_graph_def = convert_variables_to_constants(session, session.graph_def, output_node_names=['score/my_output'])
                    tf.train.write_graph(output_graph_def, export_dir, 'expert-graph.pb', as_text=False)
                else:
                    improved_str = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, Train Rec: {3:>7.2%}, Train F1: {4:>7.2%},'\
                    + ' Val Loss: {5:>6.2}, Val Acc: {6:>7.2%}, Val Rec: {7:>7.2%}, Val F1: {8:>7.2%}, Time: {9} {10}'
                print(msg.format(total_batch, loss_train, acc_train, rec_train, f1_train, loss_val, acc_val, rec_val, f1_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)  # 运行优化
            total_batch += 1

            ######
            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)
            ######

            # time0, time1 = session.run([model.time1, model.time0], feed_dict=feed_dict)
            # t = (time1 - time0)*1000
            # print (time0, time1, t)


if __name__ == '__main__':
    print('Configuring CNN model...')

    build_vocab(train_dir, vocab_dir, label_dir, vocab_dict_dir, vocab_size)
    categories, cat_to_id, id_to_cat = read_category(label_dir)
    num_classes = len(cat_to_id)
    f = open(id_to_cat_dir, 'w')
    ind = 0
    for v in id_to_cat.values():
        f.write(str(ind) + "\t" + v + "\n")
        ind += 1
    f.close()
    words, word_to_id = read_vocab(vocab_dir)
    model = TextCNN(seq_length, num_classes, vocab_size, embedding_dim, filter_sizes, num_filters, hidden_dim, learning_rate)
    train()