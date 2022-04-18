import numpy as np
from collections import Counter
import math
from parameter_attention_pos_twitter import num_labels


def p_r_f1(labels_y, ranks, k):
    precisions = 0
    n_labels = 0
    # prediction && ground truth
    for pre, gt in zip(ranks, labels_y):

        # 按照 probability 从大到小进行排序
        # index,probability
        pre = sorted(list(enumerate(pre)), key=lambda x: x[1], reverse=True)

        for i in range(k):
            if gt[pre[i][0]] == 1:
                precisions = precisions + 1.0
        n_labels += Counter(gt)[1]

    P = precisions / (k * len(labels_y))
    R = precisions / n_labels
    if P == 0 and R == 0:
        f1 = 0.0
    else:
        f1 = 2 * P * R / (P + R)
    return P, R, f1


def ndcg(labels_y, ranks, k):
    log = [1 / math.log2(x + 2) for x in range(k)]
    result = []

    for pre, gt in zip(ranks, labels_y):
        # label, probability, 按照 probability 从大到小进行排序
        pre = sorted(list(enumerate(pre)), key=lambda x: x[1], reverse=True)
        res = np.zeros(k)
        for i in range(k):
            if gt[pre[i][0]] == 1:
                res[i] = 1
        if np.sum(res) == 0:
            ndcg_cur = 0
        else:
            ndcg_cur = (np.dot(np.array(res), log) / np.dot(-np.sort(-res), log))
        result.append(ndcg_cur)

    return np.sum(np.array(result)) / len(result)


def report_performance(y, pred, loss_value, type):
    print("\n ********************" + type + "********************")
    for i in [1, 5, 10]:
        prf_ans = p_r_f1(y, pred, i)
        ndcg_ans = ndcg(y, pred, i)
        print(" P_{}:{:.6f}, R_{}:{:.6f}, f1_{}:{:.6f}, ndcg_{}:{:.6f}".format(i, prf_ans[0], i, prf_ans[1],
                                                                               i, prf_ans[2], i, ndcg_ans))
    print("loss:{:.2f} ".format(loss_value))


# shape [text length, max length, label number]
# token_length: [text length]
def report_pos_performance(pred, y, pos_pred, token_length):
    tp, fp, tn, fn = 0, 0, 0, 0
    num = 0

    for index in range(1, len(y)):
        for label_class in range(num_labels):

            max_value = max(pos_pred[index][label_class][:token_length[index]])
            if pred[index][label_class] >= 0.5:
                num += 1
                for tl in range(token_length[index]):
                    if pos_pred[index][label_class][tl] == max_value:
                        if y[index][label_class][tl] == 1:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if y[index][label_class][tl] == 1:
                            fn += 1
                        else:
                            tn += 1
            else:
                pos_num = np.sum(y[index][label_class][:token_length[index]] == 1)
                fn += pos_num
                tn += token_length[index] - pos_num

    print("num: " + str(num))
    print("tp:{}, fp:{}, tn:{} ,fn :{}".format(tp, fp, tn, fn))
    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = tp * 1.0 / (tp + fp)

    if (tp + fn) == 0:
        recall = 0.0
    else:
        recall = tp * 1.0 / (tp + fn)

    if precision == 0 and recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    print("Precision:{:.6f}, Recall:{:.6f}, f1:{:.6f} \n".format(precision, recall, f1))


def report_loss(epoch_num, loss_value):
    print("train epoch {}, loss:{} \n".format(epoch_num, loss_value))
