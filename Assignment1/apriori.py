import sys
import csv
from collections import defaultdict
from itertools import combinations

min_spt = float(sys.argv[1])

datas = []

f = open('./'+sys.argv[2], 'r', encoding='utf-8')
rd = csv.reader(f, delimiter='\t')
for line in rd:
    datas.append(list(map(int, line)))

total = len(datas)

clist = []
llist = []
 
candidate = defaultdict(int)
for i in range(len(datas)):
    for j in range(len(datas[i])):
        candidate[datas[i][j]] += 1
candidate = dict(sorted(candidate.items()))
clist.append(candidate)

nL = {key:val for key, val in candidate.items() if (val / total)*100 >= min_spt}
llist.append(nL)

'''
nc = set(combinations(L1.keys(), 2))

ncandidate = defaultdict(int)
for i in range(len(data)):
    for can in nc:
        if set(can).issubset(set(data[i])):
            ncandidate[tuple(sorted(can))] += 1
clist.append(ncandidate)

L2 = {key:round((val / total)*100,2) for key, val in ncandidate.items() if ((val / total)*100 >= min_spt)}
llist.append(L2)
'''

idx = 2
while True:
    ncandidate = defaultdict(int)
    
    nc = set()
    key_list = list(nL.keys())
    for i in range(len(key_list)):
        for j in range(i+1, len(key_list)):
            if idx == 2:
                temp = {key_list[i]} | {key_list[j]}
            else :
                temp = set(key_list[i]) | set(key_list[j])
            nc = nc.union(set(combinations(temp, idx)))
    nc = sorted(nc)

    for can in nc:
        temp = list(combinations(can, idx-1))
        for tp in temp:
            if idx == 2:
                tp = tp[0]
            if tp not in nL.keys():
                if can in nc:
                    nc.remove(can)

    for data in datas:
        for c in nc:
            if set(c).issubset(set(data)):
                ncandidate[tuple(sorted(c))] += 1
    
    clist.append(ncandidate)

    nL = {key:round((val / total)*100,2) for key, val in ncandidate.items() if ((val / total)*100 >= min_spt)}
    
    if len(nL.keys()) == 0:
        break

    llist.append(nL)

    idx += 1

print(llist[1][(1,8)])
print(llist[1][(1,9)])
print(llist[1][(1,10)])
print(len(llist))
print(llist[3])
print(llist[3][(1,3,4,16)])
for can in clist:
    pass
    # csv에 쓰기
    
# candidate, L에 관한 배열을 만들어 이전 자료를 참고할 수 있게 해결