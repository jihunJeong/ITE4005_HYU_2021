import pandas as pd

answer_df = pd.read_table('./test/dt_result1.txt', sep='\s+')
answer_key = answer_df['car_evaluation']
answer_key = pd.DataFrame(answer_key)
answer_key['id'] = answer_key.index
answer_key = answer_key[['id', 'car_evaluation']]
answer_key.to_csv('./test/submission.csv', sep=',', index=False)