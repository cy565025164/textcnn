#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import sys
import re

UNK_ID_dict = dict()
UNK_ID_dict['with_padding'] = 0
UNK_ID_dict['with_unk'] = 1

def get_words(file):
    """停用词"""
    stop_words = set()
    for word in open(file, 'r'):
        stop_words.add(word.strip('\r\n'))
    return stop_words
stopwords = get_words("stopwords")

def erase_word(content):
    for w in stopwords:
        if w in content:
            content = content.replace(w, "")
    return content

def open_file(filename, mode='r'):
    return open(filename, mode)

def read_file(filename):
    """读取文件数据"""
    sentence, labels = [], []
    l_n = 0
    with open_file(filename) as f:
        for line in f:
            line = line.strip()
            l_n += 1
            line = line.split('\t')
            if len(line) == 3:#char+word+label
                char = line[0].strip()
                char = re.sub(r"\d+", "digit", char)
                char = erase_word(char)
                label = line[2].strip()
            if len(line) == 2:#char+label
                char = line[0].strip()
                char = re.sub(r"\d+", "digit", char)
                char = erase_word(char)
                label = line[1].strip()

            if sys.version < '3': #py2
                li = []
                new = char.decode('utf8')
                for elem in new:
                    li.append(elem.encode('utf8'))
                sentence.append(li)
                labels.append(label.decode('utf-8'))
            else: #py3
                sentence.append(list(char))
                labels.append(label)
    return sentence, labels

def build_vocab(train_dir, vocab_dir, label_dir, vocab_dict_dir, vocab_size):
    """根据训练集构建词汇表，存储"""
    data_train, labels_train = read_file(train_dir)
    label_size = len(set(labels_train))

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter_words = Counter(all_data)
    count_pairs_words = counter_words.most_common(vocab_size)
    words, _ = list(zip(*count_pairs_words))

    counter_labels = Counter(labels_train)
    count_pairs_label = counter_labels.most_common(label_size)
    labels, _ = list(zip(*count_pairs_label))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words_list = ["padding"] + ["<unk>"] + list(words)
    f = open_file(vocab_dict_dir, mode='w')
    for i in range(len(words_list))[::-1]:
        f.write(words_list[i] + '\t' + str(i) + '\n')

    open_file(vocab_dir, mode='w').write('\n'.join(words_list) + '\n')
    open_file(label_dir, mode='w').write('\n'.join(labels) + '\n')

def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict([(x,y) for (y,x) in enumerate(words)])

    return words, word_to_id

def read_category(label_dir):
    """读取分类目录"""
    categories = open_file(label_dir).read().strip().split('\n')
    cat_to_id = dict([(x,y) for (y,x) in enumerate(categories)])
    id_to_cat = dict([(y,x) for (y,x) in enumerate(categories)])

    return categories, cat_to_id, id_to_cat

def process_file(filename, word_to_id, cat_to_id, max_length, num_classes):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    unkIndex = word_to_id.get("<unk>", 1)
    data_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id.get(w, unkIndex) for w in contents[i]])

    y_pad = np.zeros((len(labels), num_classes))
    idx = 0
    neg_id = cat_to_id["negative"] if "negative" in cat_to_id else -1
    for w in labels:
        try:
            id = cat_to_id[w]
            y_pad[idx][id] = 1
            if w != "other" and w != "happy" and "negative" in cat_to_id:
                y_pad[idx][neg_id] = 1
            idx += 1
        except KeyError:
            continue
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post')
    return x_pad, y_pad

def batch_iter(x, y, batch_size):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def get_batches(x, y, n_batchs=10):
    batch_size = len(x)//n_batchs

    for ii in range(0, n_batchs*batch_size, batch_size):
        if ii != (n_batchs-1)*batch_size:
            X, Y = x[ii:ii+batch_size], y[ii:ii+batch_size]
        else:
            X, Y = x[ii:], y[ii:]
        yield X, Y