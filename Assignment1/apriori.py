import sys
import csv
from collections import defaultdict
from itertools import combinations

min_spt = float(sys.argv[1])

datas = []

print("Data read ... ", end="")
f = open('./'+sys.argv[2], 'r', encoding='utf-8')
rd = csv.reader(f, delimiter='\t')
for line in rd:
    datas.append(list(map(int, line)))

total = len(datas)
print("Done")

clist = []
flist = []

print("Apriori Algorithm Begin")

print("1 itemset ... ", end="", flush=True)
candidate = defaultdict(int)
for i in range(len(datas)):
    for j in range(len(datas[i])):
        candidate[tuple([datas[i][j]])] += 1
candidate = dict(sorted(candidate.items()))
clist.append(candidate)

frequent = {key:val for key, val in candidate.items() if (val / total)*100 >= min_spt}
flist.append(frequent)
print("Done")

idx = 2
while True:
    print(str(idx)+" itemset ... ", end="", flush=True)
    ncandidate = defaultdict(int)
    
    candidate = set()
    key_list = list(frequent.keys())
    
    for i in range(len(key_list)):
        for j in range(i+1, len(key_list)):
            temp = set(key_list[i]) | set(key_list[j])
            temp = sorted(temp)
            candidate = candidate.union(set(combinations(temp, idx)))
            
    candidate = sorted(candidate)
    
    for can in candidate:
        temp = list(combinations(can, idx-1))
        for tp in temp:        
            if tp not in frequent.keys():
                if can in candidate:
                    candidate.remove(can)

    for data in datas:
        for c in candidate:
            if set(c).issubset(set(data)):
                ncandidate[tuple(sorted(c))] += 1
                
    ncandidate = dict(sorted(ncandidate.items()))
    clist.append(ncandidate)

    frequent = {key:val for key, val in ncandidate.items() if ((val / total)*100 >= min_spt)}
    
    if len(frequent.keys()) == 0:
        break
    flist.append(frequent)

    idx += 1
    print("Done")
    
print("Doesn't exist")
print("Data write ... ", end="")

with open("./"+sys.argv[3], "w", encoding="utf-8") as csvfile:
    wr = csv.writer(csvfile, delimiter='\t')
    for idx, frequent in enumerate(flist):
        if idx == 0:
            continue
        
        for key, val in frequent.items():
            for i in range(1, len(key)):
                item_set = list(sorted(combinations(key, i)))
                for item in item_set:
                    association = sorted(list(set(key) - set(item)))
                    ans1 = [set(item), set(association), "{:.2f}".format(round((val/total)*100,2)), "{:.2f}".format(round((val/flist[len(item)-1][tuple(item)])*100,2))]
                    wr.writerow(ans1)
                    
                
print("Done")