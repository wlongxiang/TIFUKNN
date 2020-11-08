"""
this is a refactored version of the original implementation published with the paper.
"""
import argparse
import csv
import math
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HIST_DATA_IDX = 0
FUTURE_DATA_IDX = 1
# todo: what does this mean??
next_k_step = 1


def load_and_preprocess_data(files):
    """
    This function returns the processed baskets and the number of unique items in both.

    :param files:
    :return:
    """
    data_chunk = []
    set_of_items = set()
    for file in files:
        df = pd.read_csv(file)
        print(f"number of records in {file}: {len(df)}")
        customer_basket_dict = dict()
        for cust_id, cust_baskets_df in df.groupby("CUSTOMER_ID"):
            customer_basket_dict[cust_id] = []
            for order_id, basket in cust_baskets_df.groupby("ORDER_NUMBER"):
                items_in_basket = basket["MATERIAL_NUMBER"].values
                customer_basket_dict[cust_id].append(items_in_basket)
                set_of_items.update(items_in_basket)  # use set to avoid duplicates
        data_chunk.append(customer_basket_dict)

    return data_chunk, len(set_of_items)


def train_val_test_split_user(data_chunk, user_ids):
    filtered_user_ids = []
    for user_id in user_ids:
        if len(data_chunk[HIST_DATA_IDX][user_id]) <= 3:
            # remove users who has less than or equal 3 baskets in historal sets
            # i.e., we only consider those users with at least 3 hist baskets and 1 future basket
            continue
        # suppose we have one basket in future or test data per user, we only care about those basket no less than 3
        if len(data_chunk[FUTURE_DATA_IDX][user_id][0]) < 3:
            continue
        filtered_user_ids.append(user_id)

    number_of_filtered_users = len(filtered_user_ids)
    print("filtered number of users", number_of_filtered_users)

    # training_key_set and other key_set is a list of user Ids
    # note that for every user in hist there is always one basket in future
    # the train-val-test split groups different users (80%*90%, 80%*10%, 20%)  = (0.72, 0.08, 0.2)
    # 0.72 users are used for training, etc...
    training_user_ids = filtered_user_ids[0:int(4 / 5 * number_of_filtered_users * 0.9)]
    validation_user_ids = filtered_user_ids[
                          int(4 / 5 * number_of_filtered_users * 0.9):int(4 / 5 * number_of_filtered_users)]
    test_user_ids = filtered_user_ids[int(4 / 5 * number_of_filtered_users):]
    print('Number of training users: ', len(training_user_ids))
    print('Number of valid users: ', len(validation_user_ids))
    print('Number of test users: ', len(test_user_ids))

    return training_user_ids, validation_user_ids, test_user_ids


def group_history_list(his_list, group_size):
    grouped_vec_list = []
    if len(his_list) < group_size:
        # sum = np.zeros(len(his_list[0]))
        for j in range(len(his_list)):
            grouped_vec_list.append(his_list[j])

        return grouped_vec_list, len(his_list)
    else:
        est_num_vec_each_block = len(his_list) / group_size
        base_num_vec_each_block = int(np.floor(len(his_list) / group_size))
        residual = est_num_vec_each_block - base_num_vec_each_block

        num_vec_has_extra_vec = int(np.round(residual * group_size))

        if residual == 0:
            for i in range(group_size):
                if len(his_list) < 1:
                    print('len(his_list)<1')
                sum = np.zeros(len(his_list[0]))
                for j in range(base_num_vec_each_block):
                    if i * base_num_vec_each_block + j >= len(his_list):
                        print('i*num_vec_each_block+j')
                    sum += his_list[i * base_num_vec_each_block + j]
                grouped_vec_list.append(sum / base_num_vec_each_block)
        else:

            for i in range(group_size - num_vec_has_extra_vec):
                sum = np.zeros(len(his_list[0]))
                for j in range(base_num_vec_each_block):
                    if i * base_num_vec_each_block + j >= len(his_list):
                        print('i*base_num_vec_each_block+j')
                    sum += his_list[i * base_num_vec_each_block + j]
                    last_idx = i * base_num_vec_each_block + j
                grouped_vec_list.append(sum / base_num_vec_each_block)

            est_num = int(np.ceil(est_num_vec_each_block))
            start_group_idx = group_size - num_vec_has_extra_vec
            if len(his_list) - start_group_idx * base_num_vec_each_block >= est_num_vec_each_block:
                for i in range(start_group_idx, group_size):
                    sum = np.zeros(len(his_list[0]))
                    for j in range(est_num):
                        # if residual+(i-1)*est_num_vec_each_block+j >= len(his_list):
                        #     print('residual+(i-1)*num_vec_each_block+j')
                        #     print('len(his_list)')
                        iidxx = last_idx + 1 + (i - start_group_idx) * est_num + j
                        if iidxx >= len(his_list) or iidxx < 0:
                            print('last_idx + 1+(i-start_group_idx)*est_num+j')
                        sum += his_list[iidxx]
                    grouped_vec_list.append(sum / est_num)

        return grouped_vec_list, group_size


def temporal_decay_sum_history(data_set, key_set, output_size, group_size, within_decay_rate, group_decay_rate):
    """this function calculate a vector representation of a user from past history"""
    sum_history = {}
    for key in key_set:
        vec_list = data_set[key]
        num_vec = len(vec_list)
        his_list = []
        # hist_list is a list of all basket, now all basket are converted to q-hot encode of the same
        # size = #items
        for idx in range(num_vec):
            # each basket is converted to q-hot encoded wrt #items with decay
            his_vec = np.zeros(output_size)
            decayed_val = np.power(within_decay_rate, num_vec - 1 - idx)
            for ele in vec_list[idx]:
                # ele is the item id, deduct by 1 if used as index
                his_vec[ele - 1] = decayed_val
            his_list.append(his_vec)

        grouped_list, real_group_size = group_history_list(his_list, group_size)
        his_vec = np.zeros(output_size)
        for idx in range(real_group_size):
            decayed_val = np.power(group_decay_rate, group_size - 1 - idx)
            if idx >= len(grouped_list):
                print('idx: ' + str(idx))
                print('len(grouped_list): ', len(grouped_list))
            his_vec += grouped_list[idx] * decayed_val
        sum_history[key] = his_vec / real_group_size
    return sum_history


def KNN(query_set, target_set, k):
    history_mat = []
    for key in target_set.keys():
        history_mat.append(target_set[key])
    test_mat = []
    for key in query_set.keys():
        test_mat.append(query_set[key])
    print('Finding k nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(history_mat)
    distances, indices = nbrs.kneighbors(test_mat)
    print('Finish KNN search.')
    return indices, distances


def merge_history(sum_history_test, test_key_set, training_sum_history_test, training_key_set, index, alpha):
    merged_history = {}
    for test_key_id in range(len(test_key_set)):
        test_key = test_key_set[test_key_id]
        test_history = sum_history_test[test_key]
        sum_training_history = np.zeros(len(test_history))
        for indecis in index[test_key_id]:
            training_key = training_key_set[indecis]
            sum_training_history += training_sum_history_test[training_key]
        # sum_training_history is the sum of all similar users
        sum_training_history = sum_training_history / len(index[test_key_id])
        # the merged one is a weight sum of user's own vector representation plus their similar users' vector
        merge = test_history * alpha + sum_training_history * (1 - alpha)
        merged_history[test_key] = merge

    return merged_history


def predict_with_elements_in_input(sum_history, key):
    output_vectors = []

    for idx in range(next_k_step):
        vec = sum_history[key]
        output_vectors.append(vec)
    return output_vectors


def get_precision_recall_Fscore(groundtruth, pred):
    a = groundtruth
    b = pred
    correct = 0
    truth = 0
    positive = 0

    for idx in range(len(a)):
        if a[idx] == 1:
            truth += 1
            if b[idx] == 1:
                correct += 1
        if b[idx] == 1:
            positive += 1

    flag = 0
    if 0 == positive:
        precision = 0
        flag = 1
        # print('postivie is 0')
    else:
        precision = correct / positive
    if 0 == truth:
        recall = 0
        flag = 1
        # print('recall is 0')
    else:
        recall = correct / truth

    if flag == 0 and precision + recall > 0:
        F = 2 * precision * recall / (precision + recall)
    else:
        F = 0
    return precision, recall, F, correct


def get_NDCG1(groundtruth, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1) / math.log2(count + 1 + 1)
        count += 1
    idcg = 0
    num_real_item = np.sum(groundtruth)
    num_item = int(num_real_item)
    for i in range(num_item):
        idcg += (1) / math.log2(i + 1 + 1)
    ndcg = dcg / idcg
    return ndcg


def get_HT(groundtruth, pred_rank_list, k):
    count = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            return 1
        count += 1

    return 0


def evaluate(data_chunk, training_user_ids, test_user_ids, number_of_items, group_size,
             within_decay_rate, group_decay_rate, num_nearest_neighbors, alpha, topk):
    activate_metric_calculation = True
    temporal_decay_sum_history_training = temporal_decay_sum_history(data_chunk[HIST_DATA_IDX],
                                                                     training_user_ids, number_of_items,
                                                                     group_size, within_decay_rate,
                                                                     group_decay_rate)
    temporal_decay_sum_history_test = temporal_decay_sum_history(data_chunk[HIST_DATA_IDX],
                                                                 test_user_ids, number_of_items,
                                                                 group_size, within_decay_rate,
                                                                 group_decay_rate)
    index, distance = KNN(temporal_decay_sum_history_test, temporal_decay_sum_history_training,
                          num_nearest_neighbors)
    # with above, we get the top 300 most similar users to those from test set
    # sum history if for test users, aka users to be predicted
    sum_history = merge_history(temporal_decay_sum_history_test, test_user_ids, temporal_decay_sum_history_training,
                                training_user_ids, index, alpha)

    if activate_metric_calculation:
        prec = []
        rec = []
        F = []
        prec1 = []
        rec1 = []
        F1 = []
        prec2 = []
        rec2 = []
        F2 = []
        prec3 = []
        rec3 = []
        F3 = []
        NDCG = []
        n_hit = 0

        num_ele = topk

        count = 0
        for iter in range(len(test_user_ids)):
            target_variable = data_chunk[FUTURE_DATA_IDX][test_user_ids[iter]]
            count += 1
            output_vectors = predict_with_elements_in_input(sum_history, test_user_ids[iter])
            top = 400
            hit = 0
            for idx in range(len(output_vectors)):
                output = np.zeros(number_of_items)
                target_topi = output_vectors[idx].argsort()[::-1][:top]
                c = 0
                for i in range(top):
                    if c >= num_ele:
                        break
                    output[target_topi[i]] = 1
                    c += 1

                vectorized_target = np.zeros(number_of_items)
                for ii in target_variable[idx]:
                    # ii is the item id, minus to convert to index
                    vectorized_target[ii - 1] = 1
                precision, recall, Fscore, correct = get_precision_recall_Fscore(vectorized_target, output)
                prec.append(precision)
                rec.append(recall)
                F.append(Fscore)
                if idx == 0:
                    prec1.append(precision)
                    rec1.append(recall)
                    F1.append(Fscore)
                elif idx == 1:
                    prec2.append(precision)
                    rec2.append(recall)
                    F2.append(Fscore)
                elif idx == 2:
                    prec3.append(precision)
                    rec3.append(recall)
                    F3.append(Fscore)
                hit += get_HT(vectorized_target, target_topi, num_ele)
                ndcg = get_NDCG1(vectorized_target, target_topi, num_ele)
                NDCG.append(ndcg)
            if hit == next_k_step:
                n_hit += 1

        # print('average precision of ' + ': ' + str(np.mean(prec)) + ' with std: ' + str(np.std(prec)))
        recall = np.mean(rec)
        ndcg = np.mean(NDCG)
        hr = n_hit / len(test_user_ids)

    return recall, ndcg, hr


def main_runner(args):
    topn = args.topn
    group_size = args.group_size
    within_decay_rate = args.within_decay_rate
    group_decay_rate = args.group_decay_rate
    num_nearest_neighbors = args.num_nearest_neighbors
    alpha = args.alpha

    data_chunk, count_of_items = load_and_preprocess_data([args.hist_file, args.future_file])

    training_user_ids, validation_user_ids, test_user_ids = train_val_test_split_user(data_chunk,
                                                                                      list(data_chunk[FUTURE_DATA_IDX]))

    recall, ndcg, hr = evaluate(data_chunk, training_user_ids, test_user_ids, count_of_items,
                                group_size, within_decay_rate, group_decay_rate,
                                num_nearest_neighbors, alpha, topn)
    print('top n: ', topn)
    print('recall: ', recall)
    print('NDCG: ', ndcg)
    print('hit ratio: ', hr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist_file", help="historical baskets", default="./data/TaFang_history_NB.csv")
    parser.add_argument("--future_file", help="the last basket", default="./data/TaFang_future_NB.csv")
    parser.add_argument("--num_nearest_neighbors", default=300, help="number of neighbours")
    parser.add_argument("--within_decay_rate", default=0.9, help="time decay ratio within a group")
    parser.add_argument("--group_decay_rate", default=0.7, help="time decay ratio across groups")
    parser.add_argument("--alpha", default=0.7, help="the weight ratio to combine two parts in prediction")
    parser.add_argument("--group_size", default=7, help="the size of a group")
    parser.add_argument("--topn", default=10, help="topn items to recommend")
    args = parser.parse_args()
    main_runner(args)
