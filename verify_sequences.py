import pandas as pd

seq_df = pd.read_csv('data/refined/v3.6/attack_sequences.csv')

print('=== Pass Count Verification ===')
fail_count = 0
total_seqs = seq_df['sequence_id'].unique()

for sid in total_seqs:
    seq = seq_df[seq_df['sequence_id'] == sid]
    pass_count = (seq['type_name'] == 'Pass').sum()
    if pass_count < 3:
        fail_count += 1
        print(f'FAIL: Seq {sid} has only {pass_count} Pass actions')

print(f'\nTotal: {len(total_seqs)} sequences')
print(f'Violations: {fail_count}')
print(f'Result: {"PASS" if fail_count == 0 else "FAIL"}')
