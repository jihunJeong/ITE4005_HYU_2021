import sys
import math
import copy
import pandas as pd
from collections import defaultdict

if __name__ == "__main__":
    answer = sys.argv[1]
    result = sys.argv[2]
    
    answer_df = pd.read_table('./test/'+answer, sep='\s+')
    result_df = pd.read_table('./test/'+result, sep='\s+')
    cnt = 0
    for idx in range(len(answer_df)):
        if answer_df[answer_df.columns[-1]].iloc[idx] != result_df[answer_df.columns[-1]].iloc[idx]:
            cnt += 1
            print()
            print(f"{idx} is wrong")
            print("Predict")
            print("{}".format(result_df.iloc[idx]))
            print("Answer {}".format(answer_df[answer_df.columns[-1]].iloc[idx]))
            
    print(cnt)
    print(347-cnt)