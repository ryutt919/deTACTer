import pandas as pd
import numpy as np
import os

# 경로 설정
DATA_DIR = 'c:/Users/Public/Documents/DIK/deTACTer/data/refined/'
CLUSTER_FILE = DATA_DIR + 'cluster_labels.csv'
SEQUENCE_FILE = DATA_DIR + 'attack_sequences.csv'
PREPROCESSED_FILE = DATA_DIR + 'preprocessed_data.csv'

def find_target_sequence_ids():
    cluster_df = pd.read_csv(CLUSTER_FILE, encoding='utf-8-sig')
    
    # 클러스터 컬럼 자동 감지 (agglom_cluster 우선)
    cluster_col = 'agglom_cluster' if 'agglom_cluster' in cluster_df.columns else 'optics_cluster'
    
    # 새로운 클러스터 결과에서 샘플 선정 (가장 빈번한 클러스터 5, 6, 2, 3)
    targets = [
        (5, 1), (5, 2),
        (6, 1), (6, 2),
        (2, 1), (2, 2),
        (3, 1), (3, 2)
    ]
    
    mapping = []
    for cid, s_idx in targets:
        sids = cluster_df[cluster_df[cluster_col] == cid]['sequence_id'].values
        if len(sids) >= s_idx:
            sid = sids[s_idx - 1]
            mapping.append({
                'label': f'c{cid}_s{s_idx}',
                'sequence_id': sid
            })
            
    return mapping

def extract_clean_context(mapping):
    print("Loading attack_sequences.csv...")
    seq_df = pd.read_csv(SEQUENCE_FILE, encoding='utf-8-sig')
    print("Loading preprocessed_data.csv (this may take a while)...")
    raw_df = pd.read_csv(PREPROCESSED_FILE, encoding='utf-8-sig')
    
    all_context_frames = []
    
    for item in mapping:
        label = item['label']
        target_sid = item['sequence_id']
        
        # 1. 해당 시퀀스 정보 추출
        this_seq_meta = seq_df[seq_df['sequence_id'] == target_sid]
        if this_seq_meta.empty:
            print(f"Warning: Sequence {target_sid} not found in sequences file.")
            continue
            
        game_id = this_seq_meta['game_id'].iloc[0]
        min_aid = this_seq_meta['action_id'].min()
        max_aid = this_seq_meta['action_id'].max()
        
        # 2. 해당 경기의 전체 타임라인 추출
        game_timeline = raw_df[raw_df['game_id'] == game_id].sort_values(['period_id', 'time_seconds', 'action_id']).reset_index(drop=True)
        
        # 3. 타겟 액션들의 위치 찾기
        target_indices = game_timeline[game_timeline['action_id'].between(min_aid, max_aid)].index
        
        if not target_indices.empty:
            start_pos = max(0, target_indices.min() - 2)
            end_pos = min(len(game_timeline) - 1, target_indices.max() + 2)
            
            context = game_timeline.iloc[start_pos : end_pos + 1].copy()
            
            # 메타데이터 마킹
            context['source_label'] = label
            context['target_sid'] = target_sid
            context['is_in_sequence'] = context['action_id'].between(min_aid, max_aid)
            
            all_context_frames.append(context)
            print(f"Extracted context for {label} (Sequence {target_sid}): Actions {context['action_id'].min()} to {context['action_id'].max()}")
            
    return pd.concat(all_context_frames)

if __name__ == '__main__':
    mapping = find_target_sequence_ids()
    print("Mapping Found:", mapping)
    combined_data = extract_clean_context(mapping)
    combined_data.to_csv('target_sequences_context_refined.csv', index=False, encoding='utf-8-sig')
    print("Data saved to target_sequences_context_refined.csv")
