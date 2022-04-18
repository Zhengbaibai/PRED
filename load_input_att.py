<<<<<<< HEAD
from parameter_attention_twitter import *
# from parameter_attention_weibo import *
from utils import load, save
from bert.tokenization import FullTokenizer
import create_input
import numpy as np
import os
from collections import defaultdict


def load_data(data_type):
    texts = load(data_path + data_type + text_path)
    labels = load(data_path + data_type + label_path)

    # 考虑到sequence需要补零得问题，index==0 的位置应是空白tweet
    texts.insert(0, " ")
    labels.insert(0, [0] * num_labels)
    print("load %s data" % data_type)
    return texts, labels


def load_ids(data_type):
    ids = load(data_path + data_type + ids_path)
    # 考虑到sequence需要补零得问题，index==0 的位置应是空白tweet
    ids.insert(0, 0)
    # ids = ids[:3200]
    return ids


def build_history_list_by_id_dic(id_dic, ids):
    history_list = []
    for i in range(0, len(ids)):
        history = []
        for index in id_dic[ids[i]]:
            if index != i and len(history) < max_history_length:
                history.append(index)
        if len(history) == max_history_length:
            history_list.append(history)
        else:
            cur_list = history + [0] * (max_history_length - len(history))
            history_list.append(cur_list)
    return np.asarray(history_list, dtype=np.int32)


def build_train_history(ids):
    id_dic = defaultdict(list)
    for i in range(1, len(ids)):
        id_dic[ids[i]].append(i)
    return build_history_list_by_id_dic(id_dic, ids)


# 为测试集和验证集数据引入训练集的数据
def build_val_or_test_history(test_ids, train_ids):
    id_dic = defaultdict(list)
    test_id_set = set(test_ids)

    for i in range(1, len(train_ids)):
        if train_ids[i] in test_id_set:
            id_dic[train_ids[i]].append(i)
    return build_history_list_by_id_dic(id_dic, test_ids)


def process_train_data(data_type):
    if os.path.exists(data_path + data_type + origin_input_idsList_path):
        input_idsList = np.asarray(load(data_path + data_type + origin_input_idsList_path), dtype=np.float32)
        input_masksList = np.asarray(load(data_path + data_type + origin_input_masksList_path), dtype=np.float32)
        segment_idsList = np.asarray(load(data_path + data_type + origin_segment_idsList_path), dtype=np.float32)
    else:
        texts, labels = load_data(data_type)
        tokenizer = FullTokenizer(vocab_file=vocab_file)  # token 处理器，主要作用就是 分字，将字转换成ID。vocab_file 字典文件路径
        input_idsList = []
        input_masksList = []
        segment_idsList = []
        for t in texts:
            single_input_id, single_input_mask, single_segment_id = create_input.convert_single_example(max_seq_length,
                                                                                                        tokenizer, t)
            input_idsList.append(single_input_id)
            input_masksList.append(single_input_mask)
            segment_idsList.append(single_segment_id)

        input_idsList = np.asarray(input_idsList, dtype=np.int32)
        input_masksList = np.asarray(input_masksList, dtype=np.int32)
        segment_idsList = np.asarray(segment_idsList, dtype=np.int32)
        save(input_idsList, data_path + data_type + origin_input_idsList_path)
        save(input_masksList, data_path + data_type + origin_input_masksList_path)
        save(segment_idsList, data_path + data_type + origin_segment_idsList_path)

    # 在此处进行 onehot 操作
    print("%s_num_labels: %s" % (data_type, str(num_labels)))

    one_hot_label_path = data_path + data_type + "/one_hot_labels.pkl"
    if os.path.exists(one_hot_label_path):
        one_hot_labels = np.asarray(load(one_hot_label_path), dtype=np.float32)
    else:
        one_hot_labels = [[0] * num_labels for _ in range(len(labels))]
        for k in range(len(labels)):
            for x in labels[k]:
                one_hot_labels[k][x] = 1
        one_hot_labels = np.asarray(one_hot_labels, dtype=np.float32)
        save(one_hot_labels, one_hot_label_path)
    # one_hot_labels = one_hot_labels[:3200]

    print("process_train_data one_hot_labels")
    print(len(one_hot_labels))

    return input_idsList, input_masksList, segment_idsList, one_hot_labels, len(input_idsList)


# history_idsList 要进行一次展开, 传给 placeholder 后再做一次展开
def concatenate_tweet_history(current_tweet_list, train_tweet_list, history_ids):
    # [n * C]
    tw = np.expand_dims(current_tweet_list, 1)  # [n , 1 , C]
    his_index = history_ids.flatten()
    print("concatenate_tweet_history his_index")
    print(len(his_index))
    his = train_tweet_list[his_index, :]  # [(n *  max_history_length) , C]
    his = np.reshape(his, (-1, max_history_length, his.shape[1]))
    # [n ,  max_history_length , C]
    return np.concatenate((tw, his), axis=1)  # [n ,( 1+ max_history_length ) , C]


def process_train_history_data(data_type,ids, input_idsList, input_masksList, segment_idsList, labels):
    # [n * C]
    if os.path.exists(data_path + data_type + input_idsList_path):
        input_idsList = np.asarray(load(data_path + data_type + input_idsList_path), dtype=np.float32)
        input_masksList = np.asarray(load(data_path + data_type + input_masksList_path), dtype=np.float32)
        segment_idsList = np.asarray(load(data_path + data_type + segment_idsList_path), dtype=np.float32)
        one_hot_labels = np.asarray(load(data_path + data_type + one_hot_labels_path), dtype=np.int32)
    else:
        history_idsList = build_train_history(ids)
        input_idsList = concatenate_tweet_history(input_idsList, input_idsList, history_idsList)
        input_masksList = concatenate_tweet_history(input_masksList, input_masksList, history_idsList)
        segment_idsList = concatenate_tweet_history(segment_idsList, segment_idsList, history_idsList)
        one_hot_labels = concatenate_tweet_history(labels, labels, history_idsList)
        save(input_idsList, data_path + data_type + input_idsList_path)
        save(input_masksList, data_path + data_type + input_masksList_path)
        save(segment_idsList, data_path + data_type + segment_idsList_path)
        save(one_hot_labels, data_path + data_type + one_hot_labels_path)
    return input_idsList, input_masksList, segment_idsList, one_hot_labels


def process_test_val_history_data(data_type, val_ids, ids, input_idsList, input_masksList, segment_idsList, labels):
    # [n * C]
    if os.path.exists(data_path + data_type + input_idsList_path):
        input_idsList = np.asarray(load(data_path + data_type + input_idsList_path), dtype=np.float32)
        input_masksList = np.asarray(load(data_path + data_type + input_masksList_path), dtype=np.float32)
        segment_idsList = np.asarray(load(data_path + data_type + segment_idsList_path), dtype=np.float32)
        one_hot_labels = np.asarray(load(data_path + data_type + one_hot_labels_path), dtype=np.int32)
    else:
        history_idsList = build_val_or_test_history(val_ids, ids)
        tw_input_idsList, tw_input_masksList, tw_segment_idsList, tw_one_hot_labels, tw_texts_len = process_train_data(
            data_type)
        input_idsList = concatenate_tweet_history(tw_input_idsList, input_idsList, history_idsList)
        input_masksList = concatenate_tweet_history(tw_input_masksList, input_masksList, history_idsList)
        segment_idsList = concatenate_tweet_history(tw_segment_idsList, segment_idsList, history_idsList)
        one_hot_labels = concatenate_tweet_history(tw_one_hot_labels, labels, history_idsList)
        save(input_idsList, data_path + data_type + input_idsList_path)
        save(input_masksList, data_path + data_type + input_masksList_path)
        save(segment_idsList, data_path + data_type + segment_idsList_path)
        save(one_hot_labels, data_path + data_type + one_hot_labels_path)
    return input_idsList, input_masksList, segment_idsList, one_hot_labels, len(input_idsList)
=======
from parameter_attention_twitter import *
# from parameter_attention_weibo import *
from utils import load, save
from bert.tokenization import FullTokenizer
import create_input
import numpy as np
import os
from collections import defaultdict


def load_data(data_type):
    texts = load(data_path + data_type + text_path)
    labels = load(data_path + data_type + label_path)

    # 考虑到sequence需要补零得问题，index==0 的位置应是空白tweet
    texts.insert(0, " ")
    labels.insert(0, [0] * num_labels)
    print("load %s data" % data_type)
    return texts, labels


def load_ids(data_type):
    ids = load(data_path + data_type + ids_path)
    # 考虑到sequence需要补零得问题，index==0 的位置应是空白tweet
    ids.insert(0, 0)
    # ids = ids[:3200]
    return ids


def build_history_list_by_id_dic(id_dic, ids):
    history_list = []
    for i in range(0, len(ids)):
        history = []
        for index in id_dic[ids[i]]:
            if index != i and len(history) < max_history_length:
                history.append(index)
        if len(history) == max_history_length:
            history_list.append(history)
        else:
            cur_list = history + [0] * (max_history_length - len(history))
            history_list.append(cur_list)
    return np.asarray(history_list, dtype=np.int32)


def build_train_history(ids):
    id_dic = defaultdict(list)
    for i in range(1, len(ids)):
        id_dic[ids[i]].append(i)
    return build_history_list_by_id_dic(id_dic, ids)


# 为测试集和验证集数据引入训练集的数据
def build_val_or_test_history(test_ids, train_ids):
    id_dic = defaultdict(list)
    test_id_set = set(test_ids)

    for i in range(1, len(train_ids)):
        if train_ids[i] in test_id_set:
            id_dic[train_ids[i]].append(i)
    return build_history_list_by_id_dic(id_dic, test_ids)


def process_train_data(data_type):
    if os.path.exists(data_path + data_type + origin_input_idsList_path):
        input_idsList = np.asarray(load(data_path + data_type + origin_input_idsList_path), dtype=np.float32)
        input_masksList = np.asarray(load(data_path + data_type + origin_input_masksList_path), dtype=np.float32)
        segment_idsList = np.asarray(load(data_path + data_type + origin_segment_idsList_path), dtype=np.float32)
    else:
        texts, labels = load_data(data_type)
        tokenizer = FullTokenizer(vocab_file=vocab_file)  # token 处理器，主要作用就是 分字，将字转换成ID。vocab_file 字典文件路径
        input_idsList = []
        input_masksList = []
        segment_idsList = []
        for t in texts:
            single_input_id, single_input_mask, single_segment_id = create_input.convert_single_example(max_seq_length,
                                                                                                        tokenizer, t)
            input_idsList.append(single_input_id)
            input_masksList.append(single_input_mask)
            segment_idsList.append(single_segment_id)

        input_idsList = np.asarray(input_idsList, dtype=np.int32)
        input_masksList = np.asarray(input_masksList, dtype=np.int32)
        segment_idsList = np.asarray(segment_idsList, dtype=np.int32)
        save(input_idsList, data_path + data_type + origin_input_idsList_path)
        save(input_masksList, data_path + data_type + origin_input_masksList_path)
        save(segment_idsList, data_path + data_type + origin_segment_idsList_path)

    # 在此处进行 onehot 操作
    print("%s_num_labels: %s" % (data_type, str(num_labels)))

    one_hot_label_path = data_path + data_type + "/one_hot_labels.pkl"
    if os.path.exists(one_hot_label_path):
        one_hot_labels = np.asarray(load(one_hot_label_path), dtype=np.float32)
    else:
        one_hot_labels = [[0] * num_labels for _ in range(len(labels))]
        for k in range(len(labels)):
            for x in labels[k]:
                one_hot_labels[k][x] = 1
        one_hot_labels = np.asarray(one_hot_labels, dtype=np.float32)
        save(one_hot_labels, one_hot_label_path)
    # one_hot_labels = one_hot_labels[:3200]

    print("process_train_data one_hot_labels")
    print(len(one_hot_labels))

    return input_idsList, input_masksList, segment_idsList, one_hot_labels, len(input_idsList)


# history_idsList 要进行一次展开, 传给 placeholder 后再做一次展开
def concatenate_tweet_history(current_tweet_list, train_tweet_list, history_ids):
    # [n * C]
    tw = np.expand_dims(current_tweet_list, 1)  # [n , 1 , C]
    his_index = history_ids.flatten()
    print("concatenate_tweet_history his_index")
    print(len(his_index))
    his = train_tweet_list[his_index, :]  # [(n *  max_history_length) , C]
    his = np.reshape(his, (-1, max_history_length, his.shape[1]))
    # [n ,  max_history_length , C]
    return np.concatenate((tw, his), axis=1)  # [n ,( 1+ max_history_length ) , C]


def process_train_history_data(data_type,ids, input_idsList, input_masksList, segment_idsList, labels):
    # [n * C]
    if os.path.exists(data_path + data_type + input_idsList_path):
        input_idsList = np.asarray(load(data_path + data_type + input_idsList_path), dtype=np.float32)
        input_masksList = np.asarray(load(data_path + data_type + input_masksList_path), dtype=np.float32)
        segment_idsList = np.asarray(load(data_path + data_type + segment_idsList_path), dtype=np.float32)
        one_hot_labels = np.asarray(load(data_path + data_type + one_hot_labels_path), dtype=np.int32)
    else:
        history_idsList = build_train_history(ids)
        input_idsList = concatenate_tweet_history(input_idsList, input_idsList, history_idsList)
        input_masksList = concatenate_tweet_history(input_masksList, input_masksList, history_idsList)
        segment_idsList = concatenate_tweet_history(segment_idsList, segment_idsList, history_idsList)
        one_hot_labels = concatenate_tweet_history(labels, labels, history_idsList)
        save(input_idsList, data_path + data_type + input_idsList_path)
        save(input_masksList, data_path + data_type + input_masksList_path)
        save(segment_idsList, data_path + data_type + segment_idsList_path)
        save(one_hot_labels, data_path + data_type + one_hot_labels_path)
    return input_idsList, input_masksList, segment_idsList, one_hot_labels


def process_test_val_history_data(data_type, val_ids, ids, input_idsList, input_masksList, segment_idsList, labels):
    # [n * C]
    if os.path.exists(data_path + data_type + input_idsList_path):
        input_idsList = np.asarray(load(data_path + data_type + input_idsList_path), dtype=np.float32)
        input_masksList = np.asarray(load(data_path + data_type + input_masksList_path), dtype=np.float32)
        segment_idsList = np.asarray(load(data_path + data_type + segment_idsList_path), dtype=np.float32)
        one_hot_labels = np.asarray(load(data_path + data_type + one_hot_labels_path), dtype=np.int32)
    else:
        history_idsList = build_val_or_test_history(val_ids, ids)
        tw_input_idsList, tw_input_masksList, tw_segment_idsList, tw_one_hot_labels, tw_texts_len = process_train_data(
            data_type)
        input_idsList = concatenate_tweet_history(tw_input_idsList, input_idsList, history_idsList)
        input_masksList = concatenate_tweet_history(tw_input_masksList, input_masksList, history_idsList)
        segment_idsList = concatenate_tweet_history(tw_segment_idsList, segment_idsList, history_idsList)
        one_hot_labels = concatenate_tweet_history(tw_one_hot_labels, labels, history_idsList)
        save(input_idsList, data_path + data_type + input_idsList_path)
        save(input_masksList, data_path + data_type + input_masksList_path)
        save(segment_idsList, data_path + data_type + segment_idsList_path)
        save(one_hot_labels, data_path + data_type + one_hot_labels_path)
    return input_idsList, input_masksList, segment_idsList, one_hot_labels, len(input_idsList)
>>>>>>> fad0c2c7d5f1bd28b868c1169bc7a25da0b518c5
