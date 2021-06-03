import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from collections import Counter, defaultdict

def dist(x, y):
    return np.linalg.norm(np.array(x)-np.array(y))

def _assign(array, cidx, classify, cur, ndict):
    global minPts
    
    nlist = ndict[cidx]
    if len(nlist) < minPts:
        classify[cidx] = None
        return False
    
    classify[cidx] = cur
    while nlist:
        nidx = nlist.pop()

        if classify[nidx] == None:
            classify[nidx] = cur
        elif classify[nidx] == -1:
            classify[nidx] = cur
            pnew_neighbor = ndict[nidx]
            
            if len(pnew_neighbor) >= minPts:
                nlist = list(set(nlist) | set(pnew_neighbor))
    
    return True

if __name__ == "__main__":
    global eps, minPts

    np.random.seed(0)

    input_file = sys.argv[1]
    number = input_file[5:-4]
    n = int(sys.argv[2])
    eps = int(sys.argv[3])
    minPts = int(sys.argv[4])

    input_df = pd.read_csv('./data-3/'+input_file, sep='\s+', names=['object_id', 'x', 'y'], dtype={'object_id':'Int64'}, index_col='object_id')
    input = input_df.to_numpy()

    n_points = len(input)
    classify = [-1] * n_points
    cur_cluster = 0
    desity_reachable = []
    
    print("Calculate Neighbor ... ", end="", flush=True)
    neighbor_list = defaultdict(list)
    for i in range(len(input)):
        for j in range(i+1, len(input)):
            if dist(input[i], input[j]) <= eps:
                neighbor_list[i].append(j)
                neighbor_list[j].append(i)
    print("Done")

    print("Calculate DBSCAN ... ", end="", flush=True)
    for idx in range(len(input)):
        if classify[idx] != -1:
            continue
        if _assign(input, idx, classify, cur_cluster, neighbor_list):
            cur_cluster += 1
    print("Done")

    unique_labels = set(classify)
    print(f"{len(unique_labels)} clusters created")
    colors = [plt.cm.gist_rainbow(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for cluster_idx, col in zip(unique_labels, colors):
        if cluster_idx == None:
            col = [0, 0, 0, 1]
        class_mask = list(filter(lambda x : classify[x] == cluster_idx, range(n_points)))
        plt.plot(input_df.values[class_mask][:, 0], input_df.values[class_mask][:, 1], 
            'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), 
            markersize=1)
    plt.show()
    
    counter = Counter(classify)
    counter = sorted(counter.items(), key=lambda x : x[1], reverse=True)
    print(counter)

    print("Make Result File ... ", end="", flush=True)
    path = "./result"
    if not os.path.isdir(path):
        os.mkdir(path)
    cnt = 0
    for cls, num in counter:
        if cls == None:
            continue
        if cnt >= n:
            break
        li = list(filter(lambda x : classify[x] == cls, range(n_points)))
        li_df = pd.DataFrame(li)
        li_df.to_csv(path+f'/input{number}_cluster_{cnt}.txt', index=False,header=False)
        cnt += 1
    print("Done")