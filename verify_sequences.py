import pandas as pd
import yaml

# 설정 로드
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
VERSION = config['version'] # v4.4

path = f'data/refined/{VERSION}/attack_sequences.csv'
seq_df = pd.read_csv(path)

print(f'=== v4.4 Verification ({path}) ===')
fail_count = 0
total_seqs = seq_df['sequence_id'].unique()

for sid in total_seqs:
    seq = seq_df[seq_df['sequence_id'] == sid]
    
    # [1] 패스 개수 검증 (성과 제외)
    non_outcome = seq[seq['is_outcome'] == False]
    pass_count = (non_outcome['type_name'] == 'Pass').sum()
    
    # [2] y좌표 범위 검증 (성과 이벤트)
    outcome = seq[seq['is_outcome'] == True]
    y_range_pass = True
    for _, row in outcome.iterrows():
        y = row['start_y']
        if y < 0.36 or y > 0.635:
            y_range_pass = False
            break
    
    if pass_count < 3 or not y_range_pass:
        fail_count += 1
        print(f'FAIL: Seq {sid} -> Passes: {pass_count}, Y-Range: {"OK" if y_range_pass else "OUT"}')

print(f'\nTotal: {len(total_seqs)} sequences')
print(f'Violations: {fail_count}')
print(f'Result: {"PASS" if fail_count == 0 else "FAIL"}')
