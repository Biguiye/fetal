import os.path

import numpy as np
from sklearn import metrics

from Util import correlatin_multi, hard_vote, soft_vote

np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':

    # 先不考虑多折,只加载第0折

    start = 50
    end = 60

    name_list = [
        'denseNet_201',
        'ResNet_50',
        'Xception',
    ]
    time_list = [
        '2023-04-13_19-33',
        '2023-03-22_15-34',
        '2023-03-21_21-51'
    ]

    # 对于10个epoch

    hard_acc_all = 0
    hard_pre_all = 0
    hard_recall_all = 0
    hard_f1_all = 0

    soft_acc_all = 0
    soft_pre_all = 0
    soft_recall_all = 0
    soft_f1_all = 0

    for epoch in range(start, end):
        # if epoch == start+1:
        #     break

        y_true_list = \
            np.load(os.path.join('models_log', name_list[0], time_list[0], '0', 'draw_data', 'test_y_true.npy'),
                    allow_pickle=True)[epoch]

        num = len(name_list)
        y_pred_list = []
        y_score_list = []
        for i in range(num):
            temp = np.load(os.path.join('models_log', name_list[i], time_list[i], '0', 'draw_data', 'test_y_pred.npy'),
                           allow_pickle=True)[epoch]
            y_pred_list.append(temp)

            temp = np.load(os.path.join('models_log', name_list[i], time_list[i], '0', 'draw_data', 'test_score.npy'),
                           allow_pickle=True)[epoch]
            y_score_list.append(temp)

        hard_vote_list = hard_vote(y_pred_list)
        soft_vote_list = soft_vote(y_score_list)

        hard_acc = metrics.accuracy_score(y_true_list, hard_vote_list)
        hard_pre = metrics.precision_score(y_true_list, hard_vote_list, average='weighted')
        hard_recall = metrics.recall_score(y_true_list, hard_vote_list, average='weighted')
        hard_f1 = metrics.f1_score(y_true_list, hard_vote_list, average='weighted')

        soft_acc = metrics.accuracy_score(y_true_list, soft_vote_list)
        soft_pre = metrics.precision_score(y_true_list, soft_vote_list, average='weighted')
        soft_recall = metrics.recall_score(y_true_list, soft_vote_list, average='weighted')
        soft_f1 = metrics.f1_score(y_true_list, soft_vote_list, average='weighted')

        hard_acc_all += hard_acc
        hard_pre_all += hard_pre
        hard_recall_all += hard_recall
        hard_f1_all += hard_f1

        soft_acc_all += soft_acc
        soft_pre_all += soft_pre
        soft_recall_all += soft_recall
        soft_f1_all += soft_f1

    print('Hard')
    print('acc fusion: ', hard_acc_all / (end - start))
    print('pre fusion: ', hard_pre_all / (end - start))
    print('recall fusion: ', hard_recall_all / (end - start))
    print('f1 fusion: ', hard_f1_all / (end - start))

    print('Soft')

    print('acc fusion: ', soft_acc_all / (end - start))
    print('pre fusion: ', soft_pre_all / (end - start))
    print('recall fusion: ', soft_recall_all / (end - start))
    print('f1 fusion: ', soft_f1_all / (end - start))
