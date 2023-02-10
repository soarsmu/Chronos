import os
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pandas as pd
from tabulate import tabulate
from io import StringIO
from tqdm import tqdm
from utils import *
import copy
import json
import argparse

def _filter(score_mat, filter_mat, copy=True):
    if filter_mat is None:
        return score_mat
    if copy:
        score_mat = score_mat.copy()

    temp = filter_mat.tocoo()
    score_mat[temp.row, temp.col] = 0
    del temp
    score_mat = score_mat.tocsr()
    score_mat.eliminate_zeros()
    return score_mat
def parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        help="the name of the dataset for evaluation"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help="the file to print"
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=300
    )
    parser.add_argument(
        "--updated_by_cache",
        action='store_true'
    )
    parser.add_argument(
        "--replacement_by_versions",
        action='store_true'
    )
    parser.add_argument(
        "--para",
        type=str
    )
    parser.add_argument(
        "--update_position",
        type=str
    )

    args, _ = parser.parse_known_args()
    return args
def prepare_data(dataset):
    RES_DIR = f'Results/{dataset}'
    DATA_DIR = f'GZXML-Datasets/{dataset}'

    # print(_c("Loading files", attr="yellow"))
    print("Loading files")
    trn_X_Y = read_sparse_mat('%s/trn_X_Y.txt'%DATA_DIR, use_xclib=False)
    tst_X_Y = read_sparse_mat('%s/tst_X_Y.txt'%DATA_DIR, use_xclib=False)

    score_mat = _filter(read_bin_spmat(f'{RES_DIR}/score_mat.bin').copy(), None)
    # Shape should be:
    # nrows = number of test data
    # ncols = scores for possible labels
    x = score_mat.toarray()
    # print(x.shape) #[2740,2817]

    # getting the set of seen labels in training dataset
    seen_labels = set()
    train_label = []
    with open(f'{DATA_DIR}/trn_X_Y.txt', "r", encoding="utf-8") as re:
        train_label = re.readlines()[1:]
    for text in train_label:
        list_labels = []
        split = text.split(" ")
        for label in split:
            label_num = label.split(":")[0]
            seen_labels.add(int(label_num))


    # loop through the score matrix
    text_labels = []
    with open(f'{DATA_DIR}/tst_X_Y.txt', "r", encoding="utf-8") as re:
        text_labels = re.readlines()[1:]
    actuals = []
    for text in text_labels:
        list_labels = []
        split = text.split(" ")
        for label in split:
            label_num = label.split(":")[0]
            list_labels.append(int(label_num))
        actuals.append(list_labels)

    return actuals,seen_labels,x
def init_evaluation(out_file_name=None,replacement=False):
    sum_index = [1,2,3,100]
    max_k = max(sum_index)
    sum_recall = [0]*(max_k+1)
    sum_precision = [0]*(max_k+1)


    num_test_data = 0
    total_labels = 0

    prediction_not_seen = {}
    prediction_not_seen_correct = {}

    '''
    cache setting
    '''
    cache = []
    if out_file_name is not None:
        out_file=open(f'{out_file_name}.txt', 'w')
        out_file=open(f'{out_file_name}.txt', 'a')
    else:
        out_file = None

    version_map = []
    if replacement:
        with open('version_map.json','r')as f:
            version_map = json.load(f)
    return sum_index,max_k,sum_recall,sum_precision,num_test_data,total_labels,prediction_not_seen,prediction_not_seen_correct,cache,out_file,version_map 

def evaluation_loop(args):
    actuals,seen_labels,x = prepare_data(args.dataset)
    sum_index,max_k,sum_recall,sum_precision,num_test_data,total_labels,prediction_not_seen,prediction_not_seen_correct,cache,out_file,version_map = init_evaluation(out_file_name=args.output_file,replacement=args.replacement_by_versions)
    print("version_map",version_map)
    for i, rows in enumerate(x):

        orignal_predictions = np.argpartition(rows, -5)[-5:] #排序前3名的可能性序号
        # print(rows)
        # print(rows[predictions])
        # print(predictions)
        orignal_predictions = orignal_predictions[::-1]
        


        updated_predictions = np.argpartition(rows, -100)[-100:] #排序前3名的可能性序号
        # print(rows)
        # print(rows[predictions])
        # print(predictions)
        updated_predictions = updated_predictions[::-1]
        updated_probability = rows[updated_predictions]
        updated_index = np.argsort(-updated_probability)
        updated_predictions = updated_predictions[updated_index]


        '''consider the version_map'''
        for up_idx,e in enumerate(updated_predictions[:args.update_position]):
            # if str(e) in version_map and e not in cache:
            if str(e) in version_map:
                for updated_package in version_map[str(e)][::-1]:
                    if updated_package in cache and updated_package not in updated_predictions[:up_idx]:
                        # if (e in cache and cache.index(updated_package) < cache.index(e)) or e not in cache:
                        # temp = rows[updated_package]
                        rows[updated_package] = rows[e]
                        rows[e] = 0
                        updated_predictions[up_idx] = updated_package
                        print(f'old new version {e} -> {updated_package}',file=out_file)
                        
                        break




        '''
        cache rerank
        '''
        orignal_rows = copy.deepcopy(rows)
        
        p_useful_predictions = rows[updated_predictions[:args.update_position]]

        p_max = np.max(p_useful_predictions)
        p_min = np.min(p_useful_predictions)
        p_mean  =np.mean(p_useful_predictions)
        p_middle = p_useful_predictions[int(args.update_position/2)]
        print(f'p_max: {p_max}\np_min: {p_min}\np_mean: {p_mean}\np_middle: {p_middle}',file=out_file)
        if len(cache) != 0 and args.updated_by_cache == True:
            for rank,e in enumerate(cache):
                if e in updated_predictions[:args.update_position]:
                    P_orignal = rows[e]
                    # P_updated = (1+0.8/(rank+1))*(P_orignal+p_mean)
                    P_updated = P_orignal + args.para/(rank+1)*p_mean
                    # P_updated = 1*(P_orignal+p_middle)

                    rows[e] = P_updated
                    if P_orignal !=0:
                        print(f"orignal: {P_orignal} -> updated: {P_updated}  ,{e}, {args.para/(rank+1)}",file=out_file)
        
        '''
        '''
        updated_predictions = np.argpartition(rows, -100)[-100:] #排序前3名的可能性序号
        # print(rows)
        # print(rows[predictions])
        # print(predictions)
        updated_predictions = updated_predictions[::-1]
        updated_probability = rows[updated_predictions]
        updated_index = np.argsort(-updated_probability)
        updated_predictions = updated_predictions[updated_index]

        num_test_data += 1
        local_correct_prediction = 0
        labels = actuals[i]
        print(orignal_predictions,updated_predictions[:5],file=out_file)
        print(orignal_rows[orignal_predictions],rows[updated_predictions[:5]],file=out_file)
        print(labels,file=out_file)
        print([updated_predictions.tolist().index(l) if l in updated_predictions else -1 for l in labels],file=out_file)
        print(rows[labels],file=out_file)
        print([ l for l  in  labels if l in cache  ],file=out_file)
        print([ cache.index(l) for l  in  labels if l in cache  ],file=out_file)

        print('\n\n\n',file=out_file)
        if args.updated_by_cache == True or args.replacement_by_versions == True:
            for l in labels:
                if l in cache:
                    cache.remove(l)
                    cache.insert(0,l)
                elif l not in cache and len(cache) < args.cache_size -1:
                    cache.insert(0,l)
                else:
                    cache = cache[:-1]
                    cache.insert(0,l)
            

        # get only the top k prediction
        total_labels += len(labels)
        correct_prediction = 0
        

        
        for K in range(max_k):
            if updated_predictions[K] in labels:
                correct_prediction += 1
                if updated_predictions[K] not in seen_labels:
                    prediction_not_seen_correct[updated_predictions[K]] = prediction_not_seen_correct.get(updated_predictions[K], 0) + 1
            if updated_predictions[K] not in seen_labels:
                prediction_not_seen[updated_predictions[K]] = prediction_not_seen.get(updated_predictions[K], 0) + 1
            if K+1 in sum_index:
                sum_precision[K+1] += (correct_prediction / min(K+1, len(labels)))
                # sum_precision_1 += (correct_prediction / 1)
                sum_recall[K+1] += (correct_prediction / len(labels))

            

    print("TOTAL LABELS: " + total_labels.__str__(),file=out_file)


    for each in sum_index:
        print(f"K = {each}")
        print(f"P@{each} =  {sum_precision[each]/num_test_data}")
        print(f"R@{each} =  {sum_recall[each]/num_test_data}")
        print(f"F@{each} =  {2*(sum_precision[each]/num_test_data)*(sum_recall[each]/num_test_data)/((sum_recall[each]/num_test_data)+sum_precision[each]/num_test_data)}")

    # print(prediction_not_seen)
    print("How many unseen labels:",file=out_file)
    print(len(prediction_not_seen),file=out_file)
    print("How many unseen labels usage",file=out_file)
    sum = 0
    print(prediction_not_seen,file=out_file)
    for key, items in prediction_not_seen.items():
        sum += items
    print(sum,file=out_file)
    print()
    print()
    # print(prediction_not_seen_correct)
    print("How many unseen labels correct:",file=out_file)
    print(len(prediction_not_seen_correct),file=out_file)
    print("How many unseen labels used correctly",file=out_file)
    sum = 0
    for key, items in prediction_not_seen_correct.items():
        sum += items
    print(sum,file=out_file)
    # print(prediction_not_seen_correct[662])

if __name__ == '__main__':
    args = parameters()
    para = args.para
    update_position = args.update_position
    update_position = update_position.split(',')
    para = para.split(',')
    for p in para:
        for u_p in update_position:
            args.para = float(p)
            args.update_position = int(u_p)
            print('\n[Para]:',p,'\n[Update_position]:',u_p)
            evaluation_loop(args)