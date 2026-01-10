import pandas as pd
import numpy as np

df = pd.read_csv('c:/Users/Public/Documents/DIK/deTACTer/results/vaep_values.csv')
shots = df[df['type_name'].str.contains('shot', case=False, na=False)].index

print(f"Total shots found: {len(shots)}")

results = []
for s in shots:
    # 슈팅 직전 15개 이벤트까지 살펴봄
    sub = df.iloc[max(0, s-15):s+1]
    # vaep_value가 양수인 이벤트들이 연속되는 길이 확인 (역순으로)
    pos_vaep_count = 0
    for val in reversed(sub['vaep_value'].tolist()[:-1]): # 슈팅 자체 제외
        if val > 0:
            pos_vaep_count += 1
        else:
            break
    results.append(pos_vaep_count)

if results:
    print(f"Average positive VAEP sequence length before shot: {np.mean(results):.2f}")
    print(f"Median: {np.median(results)}")
    print(f"Max: {np.max(results)}")
    print(f"75th percentile: {np.percentile(results, 75)}")
