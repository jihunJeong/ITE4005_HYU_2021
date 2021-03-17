import sys
import csv
from collections import defaultdict
from itertools import combinations

min_spt = float(sys.argv[1])

data = []

f = open('./'+sys.argv[2], 'r', encoding='utf-8')
rd = csv.reader(f, delimiter='\t')
for line in rd:
    data.append(list(map(int, line)))

total = len(data)

clist = []
llist = []
 
candidate = defaultdict(int)
for i in range(len(data)):
    for j in range(len(data[i])):
        candidate[data[i][j]] += 1 
clist.append(candidate)

L1 = {key:val for key, val in candidate.items() if (val / total)*100 >= min_spt}
llist.append(L1)

nc = set(combinations(L1.keys(), 2))

ncandidate = defaultdict(int)
for i in range(len(data)):
    for can in nc:
        if set(can).issubset(set(data[i])):
            ncandidate[tuple(sorted(can))] += 1
clist.append(ncandidate)

L2 = {key:round((val / total)*100,2) for key, val in ncandidate.items() if ((val / total)*100 >= min_spt)}
llist.append(L2)

for can in llist:
    print(can.items())


for can in list:
    pass
    # csv에 쓰기
    
# candidate, L에 관한 배열을 만들어 이전 자료를 참고할 수 있게 해결