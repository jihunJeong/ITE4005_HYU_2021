import sys
import math
import copy
import numpy as np
import pandas as pd
import random
from collections import defaultdict, Counter
from itertools import combinations

sys.setrecursionlimit(10**6)

class DecisionTree:
    def __init__(self):
        self.attribute = None
        self.child = dict()
        self.ans = None

    def add_attribute(self, attribute):
        self.attribute = attribute

    def add_child(self, attri, subtree):
        self.child[attri] = subtree

def entropy(input_data):
    global target 
    info = 0
    cnt = dict(input_data[target].value_counts())

    for key in cnt.keys():
        probability = cnt[key] / len(input_data)
        info -= probability * math.log2(probability)

    return info

def information_gain(input_data, attribute):
    info_gain = entropy(input_data)
    cnt_data = dict(input_data[attribute].value_counts())

    for key in cnt_data.keys():
        is_key_index = input_data[attribute] == key
        info_gain -= (entropy(input_data[is_key_index]) * cnt_data[key] / len(input_data))

    return info_gain        

def split_info(input_data, attri):
    cnt_data = dict(input_data[attri].value_counts())
    info = 0
    for key in cnt_data.keys():
        is_key_index = input_data[attri] == key
        probal = len(input_data[is_key_index]) / len(input_data)
        info -= probal * math.log2(probal)
    
    return info    

def gini(input_data):
    global target
    gini = 1
    cnt = dict(input_data[target].value_counts())

    for key in cnt.keys():
        probability = cnt[key] / len(input_data)
        gini -= probability*probability
    return gini
    
def gini_index(input_data, attri):
    gini_before = gini(input_data)
    type_list = sorted(input_data[attri].unique())
    min_gini = 100000000
    for num in range(1, len(type_list)//2+1):
        for tp_set in list(combinations(type_list, num)):
            index_dict = dict()
            for idx, val in input_data[attri].items():
                index_dict[idx] = False
                if val in tp_set:
                    index_dict[idx] = True

            is_df = input_data.loc[pd.Series(index_dict)]
            not_df = input_data.drop(is_df.index)
            gini_attri = (gini(is_df)*len(is_df) + gini(not_df)*len(not_df))/len(input_data)

            if min_gini > gini_attri:
                min_gini = gini_attri
                ans_class = tp_set
                
    return gini_before - min_gini, ans_class

def select_attribute(input_data):
    max_gain, max_gini = 0, 0
    ans, ans_class = None, None
    for attri in list(input_data.columns[:-1]):
        if len(input_data[attri].unique()) == 1:
            continue
        gini_delta, split_class = gini_index(input_data, attri)
        if max_gini < gini_delta:
            max_gini = gini_delta
            ans_attri = attri
            ans_split = split_class
    
    return ans_attri, ans_split

def select_max_class(series):
    global class_info
    ans = None
    
    max_series = series[series.values == max(series)]
    min_pro = class_info[max_series.index.values[0]]
    
    for idx in max_series.index.values:
        if class_info[idx] <= min_pro:
            min_pro = class_info[idx]
            ans = idx

    return ans
    
    
def build_tree(input_data, parent_class, depth):
    global target
    tree = DecisionTree()
    
    if len(input_data) == 0:
        # 해당 가지에서 더 이상 data 없을 때 -> 부모에서 분리 완료
        return parent_class
    elif len(input_data[target].unique()) <= 1:
        # 해당 가지에 목표 Class 종류 1개
        return input_data[target].iloc[0]
    elif len(input_data) <= 3:
        # 해당 가지에서 더 이상 attributes 없을 때 Majority voting
        # 같다면 같은 속성 중 원 DF에서 확률 큰 값
        return select_max_class(input_data[target].value_counts())
        
    
    select_attri, select_class = select_attribute(input_data)
    if select_attri is None:
        return select_max_class(input_data[target].value_counts())
    
    tree.add_attribute(select_attri)
    
    # Max값이 여러개일떄 Random으로 위에것 가져온다 추후 변경 필요
    parent_class = select_max_class(input_data[target].value_counts())
    tree.ans = parent_class
    
    index_dict = dict()
    for idx, val in input_data[select_attri].items():
        index_dict[idx] = False
        if val in select_class:
            index_dict[idx] = True

    select_data = input_data.loc[pd.Series(index_dict)]
    not_select_data = input_data.drop(select_data.index)
    left_subtree = build_tree(select_data, parent_class, depth+1)
    right_subtree = build_tree(not_select_data, parent_class, depth+1)
    tree.add_child(select_class, left_subtree)
    tree.add_child('other', right_subtree)
        
    return tree

def classify(tree, input):
    for key in tree.child.keys():
        if key != 'other':
            if input[tree.attribute] in key:
                ans = key
            else :
                ans = 'other'
    
    if (type(tree.child[ans]) is str or 
        type(tree.child[ans]) is np.int64):
        return tree.child[ans]

    return classify(tree.child[ans], input)

def print_tree(tree):
    print(tree.attribute, end= " ")
    for key in tree.child.keys():
        print(f"{key} : ( ", end= " ")
        if (type(tree.child[key]) is str or 
            type(tree.child[key]) is int):
            print("{} )".format(tree.child[key]))
        else :    
            print_tree(tree.child[key])

if __name__ == "__main__":
    global target, class_info
    
    train = sys.argv[1]
    test = sys.argv[2]
    out = sys.argv[3]

    Epoches = 5
    answer_change = dict()
    train_df = pd.read_table('./train/'+train, sep='\s+')
    test_df = pd.read_table('./train/'+test, sep='\s+')
    target = train_df.columns[-1]
    ans_trees = []
    
    for col in list(train_df.columns):
        element_list = train_df[col].unique()
        for idx, element in enumerate(element_list):
            train_df[col] = train_df[col].replace(element, idx)
            if col != target:
                test_df[col] = test_df[col].replace(element, idx)
            answer_change[idx] = element
    test_df[target] = None

    class_info = dict(train_df[target].value_counts())
    mean = 0
    min_error = 10000000
    for epoch in range(Epoches):
        train = train_df.sample(frac=0.85, random_state = 100)
        valid = train_df.drop(train.index, inplace=False)
        trees = []
        for idx in range(10):
            print(f"{epoch+1} Epoches, Forest {idx+1} ... ", end= "", flush=True)
            rand_df = train.sample(frac=3, replace=True)
            tree = build_tree(rand_df, None, 0)
            trees.append(tree)
            print("Done")
        
        my_valid_df = valid.drop([target], axis=1, inplace=False).copy(deep=True)
        my_valid_df[target] = None
        for idx, row in my_valid_df.iterrows():
            ans = []
            for tree in trees:
                ans.append(classify(tree, row))
            cnt = Counter(ans).most_common()
            my_valid_df[target][idx] = cnt[0][0]
        
        wrong = 0
        for idx in range(len(valid)):
            if valid[valid.columns[-1]].iloc[idx] != my_valid_df[valid.columns[-1]].iloc[idx]:
                wrong += 1
        print("Validation Accuracy : {:.3f}".format((len(valid)-wrong)/len(valid)*100))    
        
        if wrong < min_error:
            min_error = wrong
            print("Save Model")
            ans_trees = trees     
        print()
        
    print("Claasify ... ", end="")
    for idx, row in test_df.iterrows():
        ans = []
        for tree in ans_trees:
            ans.append(classify(tree, row))
            
        cnt = Counter(ans).most_common()
        test_df[target][idx] = answer_change[cnt[0][0]]
        
    test_df.to_csv('./test/'+out, sep='\t')
    print("Done")