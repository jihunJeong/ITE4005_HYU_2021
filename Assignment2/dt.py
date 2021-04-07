import sys
import math
import copy
from numpy import split
import pandas as pd
import random
from collections import defaultdict, Counter

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

def select_attribute(input_data, remain_attri):
    max_gain = 0
    ans, ans_class = None, None
    rand_attri = random.sample(remain_attri, min(3, len(remain_attri)))
    for attri in rand_attri:
        attri_info = information_gain(input_data, attri)
        spli_info = split_info(input_data, attri)
        
        if spli_info == 0:
            return attri
        
        gain_ratio = attri_info / split_info(input_data, attri)
            
        if max_gain < gain_ratio:
            max_gain = gain_ratio
            ans = attri
                
    return ans

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
    
    
def build_tree(input_data, parent_class, remain_attri):
    global target
    tree = DecisionTree()
    
    if len(input_data) == 0:
        # 해당 가지에서 더 이상 data 없을 때 -> 부모에서 분리 완료
        return parent_class
    elif len(input_data[target].unique()) <= 1:
        # 해당 가지에 목표 Class 종류 1개
        return input_data[target].iloc[0]
    elif len(remain_attri) <= 0 or len(input_data) <= 2:
        # 해당 가지에서 더 이상 attributes 없을 때 Majority voting
        # 같다면 같은 속성 중 원 DF에서 확률 큰 값
        return select_max_class(input_data[target].value_counts())
        
    
    select_attri = select_attribute(input_data, remain_attri)
    if select_attri is None:
        return select_max_class(input_data[target].value_counts())
    
    cp_attri = copy.deepcopy(remain_attri)
    cp_attri.remove(select_attri)

    tree.add_attribute(select_attri)
    
    # Max값이 여러개일떄 Random으로 위에것 가져온다 추후 변경 필요
    parent_class = select_max_class(input_data[target].value_counts())
    tree.ans = parent_class
    
    for element in input_data[select_attri].unique():
        is_element_index = input_data[select_attri] == element
        select_data = input_data[is_element_index]
        subtree = build_tree(select_data, parent_class, cp_attri)
        tree.add_child(element, subtree)
        
    return tree

def classify(tree, input):
    
    if input[tree.attribute] not in tree.child.keys():
        return tree.ans
    
    if (type(tree.child[input[tree.attribute]]) is str or 
        type(tree.child[input[tree.attribute]]) is int):
        return tree.child[input[tree.attribute]]

    return classify(tree.child[input[tree.attribute]], input)

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

    Epoches = 10
    
    train_df = pd.read_table('./train/'+train, sep='\s+')
    target = train_df.columns[-1]
   
    class_info = dict(train_df[target].value_counts())
    mean = 0
    min_error = 10000000
    for i in range(10):
        trees = []
        for idx in range(Epoches):
            print(f"Forest {idx+1} ... ", end= "")
            rand_df = train_df.sample(frac=1, replace=False)
            tree = build_tree(train_df, None, list(train_df.columns[:-1]))
            trees.append(tree)
            print("Done")
        
        print("Claasify ... ", end="")
        test_df = pd.read_table('./train/'+test, sep='\s+')
        test_df[target] = None
        
        for idx, row in test_df.iterrows():
            ans = []
            for tree in trees:
                ans.append(classify(tree, row))
                
            cnt = Counter(ans).most_common()
            print("{} {}".format(idx,cnt))
            test_df[target][idx] = cnt[0][0]
        
        print("Done")
        
        answer_df = pd.read_table('./test/'+'dt_answer1.txt', sep='\s+')
        cnt = 0
        for idx in range(len(answer_df)):
            if answer_df[answer_df.columns[-1]].iloc[idx] != test_df[answer_df.columns[-1]].iloc[idx]:
                cnt += 1
                print()
                print(f"{idx} is wrong")
                print("Predict")
                print("{}".format(test_df.iloc[idx]))
                print("Answer {}".format(answer_df[answer_df.columns[-1]].iloc[idx]))
        
        mean += 346-cnt
        print(cnt)
        print(346-cnt)
        
        if cnt < min_error:
            print("Save Model")
            test_df.to_csv('./test/'+out, sep='\t')
    print(mean/10)