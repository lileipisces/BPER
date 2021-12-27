import datetime
import pickle
import json
import math
import os


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def load_data(data_dir, index_dir):
    user_set = set()
    item_set = set()
    exp_set = set()

    with open(os.path.join(data_dir, 'id2exp.json'), 'r', encoding='utf-8') as f:
        id2exp = json.load(f)
    IDs = pickle.load(open(os.path.join(data_dir, 'IDs.pickle'), 'rb'))
    for record in IDs:
        user_set.add(record['user'])
        item_set.add(record['item'])
        exp_set |= set(record['exp_idx'])

    # convert id to array index
    user_list = list(user_set)
    item_list = list(item_set)
    exp_list = list(exp_set)
    text_list = [id2exp[e] for e in exp_list]
    user2index = {x: i for i, x in enumerate(user_list)}
    item2index = {x: i for i, x in enumerate(item_list)}
    exp2index = {x: i for i, x in enumerate(exp_list)}

    def format_data(data_type):
        with open(os.path.join(index_dir, data_type + '.index'), 'r') as f:
            line = f.readline()
            indexes = [int(x) for x in line.split(' ')]

        tuple_list = []
        for idx in indexes:
            record = IDs[idx]
            u = user2index[record['user']]
            i = item2index[record['item']]
            exp_list = record['exp_idx']
            exps = set([exp2index[e] for e in exp_list])

            tuple_list.append([u, i, exps])
        return tuple_list

    train_tuple_list = format_data('train')
    test_tuple_list = format_data('test')
    user2items_test = {}
    for x in test_tuple_list:
        u = x[0]
        i = x[1]
        if u in user2items_test:
            user2items_test[u].add(i)
        else:
            user2items_test[u] = {i}

    return train_tuple_list, test_tuple_list, user2items_test, text_list, user2index, item2index, exp2index


def evaluate_exp(test_tuple_list, test_tuple_predict):
    top_k = len(test_tuple_predict[0])
    dcgs = [1 / math.log(i + 2) for i in range(top_k)]

    ndcg = 0
    precision_sum = 0
    recall_sum = 0  # it is also named hit ratio
    f1_sum = 0
    for x, rank_list in zip(test_tuple_list, test_tuple_predict):
        exps = x[2]
        hits = 0
        for idx, e in enumerate(rank_list):
            if e in exps:
                ndcg += dcgs[idx]
                hits += 1

        pre = hits / top_k
        rec = hits / len(exps)
        precision_sum += pre
        recall_sum += rec
        if (pre + rec) > 0:
            f1_sum += 2 * pre * rec / (pre + rec)

    ndcg = ndcg / (sum(dcgs) * len(test_tuple_list))
    precision = precision_sum / len(test_tuple_list)
    recall = recall_sum / len(test_tuple_list)
    f1 = f1_sum / len(test_tuple_list)

    return ndcg, precision, recall, f1
