
import pandas as pd
import os
import numpy as np

def preprocess_raw_data(input_path, output_path):
    print(f"Reading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. 중복 제거
    initial_len = len(df)
    df = df.drop_duplicates(subset=['game_id', 'period_id', 'time_seconds', 'player_id', 'type_name', 'result_name'])
    print(f"Removed {initial_len - len(df)} duplicate rows.")
    
    # 2. 공격 방향 표준화 (Direction Unification)
    # 모든 팀이 왼쪽 -> 오른쪽(x 0 -> 105)으로 공격하도록 통일
    # 각 팀/전반/후반 별로 공격 방향을 판단하여 필요 시 좌표 반전
    
    df_std = []
    for (game_id, period_id, team_id), group in df.groupby(['game_id', 'period_id', 'team_id']):
        # 해당 팀의 이 기간(Period) 동안의 슛 위치나 평균 액션 위치로 방향 판단
        shots = group[group['type_name'].str.lower().str.contains('shot', na=False)]
        
        # 슛 데이터가 있으면 슛 x 좌표 평균, 없으면 전체 액션 x 좌표 평균 사용
        attack_x_mean = shots['start_x'].mean() if len(shots) > 0 else group['start_x'].mean()
        
        # 경기장 중간(52.5)보다 오른쪽에 슛이 많으면 오른쪽 공격 중
        # 만약 왼쪽 공격 중이라면 모든 좌표를 반전(105-x, 68-y)
        if attack_x_mean < 52.5:
            group = group.copy()
            group['start_x'] = 105 - group['start_x']
            group['end_x'] = 105 - group['end_x']
            group['start_y'] = 68 - group['start_y']
            group['end_y'] = 68 - group['end_y']
            
        df_std.append(group)
        
    df = pd.concat(df_std).sort_values(['game_id', 'period_id', 'time_seconds'])
    print("Attack direction standardized (all teams attack left-to-right).")
    
    # 3. 시퀀스 ID 할당 (Sequence Identification)
    # 소유권이 바뀌거나 경기가 중단되면 시퀀스 번호 증가
    df['is_possession_change'] = (df['team_id'] != df['team_id'].shift(1)) | (df['period_id'] != df['period_id'].shift(1))
    df['sequence_id'] = df['is_possession_change'].cumsum()
    
    # 4. 좌표 클리핑
    df['start_x'] = df['start_x'].clip(0, 105)
    df['end_x'] = df['end_x'].clip(0, 105)
    df['start_y'] = df['start_y'].clip(0, 68)
    df['end_y'] = df['end_y'].clip(0, 68)
    
    # 5. 저장
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path} (Total sequences: {df['sequence_id'].nunique()})")

if __name__ == "__main__":
    RAW_DATA = 'data/raw_data.csv'
    CLEAN_DATA = 'data/cleaned_raw_data.csv'
    preprocess_raw_data(RAW_DATA, CLEAN_DATA)
