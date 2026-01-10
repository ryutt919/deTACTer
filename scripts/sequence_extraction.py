# =========================================================
# sequence_extraction.py
# deTACTer 프로젝트용 유효 공격 시퀀스 추출 모듈
# =========================================================
# 주요 기능:
# 1. 성과 이벤트(슈팅/박스 진입) 식별
# 2. 역추적 기반 시퀀스 추출 (공격 1-5, 빌드업 6-9)
# 3. 시퀀스별 메타데이터 및 템포 정보 포함
# =========================================================

import pandas as pd
import numpy as np
import yaml
import sys

# 터미널 한글 출력 인코딩 설정 (Windows)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =========================================================
# 설정 로드
# =========================================================
CONFIG_PATH = 'c:/Users/Public/Documents/DIK/deTACTer/config.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 설정 로드
CONFIG_PATH = 'c:/Users/Public/Documents/DIK/deTACTer/config.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 버전 설정 로드 (v3.2)
# 버전 관리 (v3.5 소프트 코딩 적용)
VERSION = config['version']
BASE_DIR = 'c:/Users/Public/Documents/DIK/deTACTer'

# 입력 및 출력 경로 설정 (버전별 폴더)
REFINED_DIR = f"{BASE_DIR}/data/refined/{VERSION}"
PREPROCESSED_PATH = f"{REFINED_DIR}/preprocessed_data.csv"
SEQUENCES_OUTPUT_PATH = f"{REFINED_DIR}/attack_sequences.csv"

# 폴더 생성 보장
import os
os.makedirs(REFINED_DIR, exist_ok=True)

# 시퀀스 길이 하이퍼파라미터 (config.yaml에서 로드하거나 기본값 사용)
ATTACK_LEN = config.get('sequence', {}).get('attack_len', 5)
BUILDUP_LEN = config.get('sequence', {}).get('buildup_len', 4)
TOTAL_SEQ_LEN = ATTACK_LEN + BUILDUP_LEN  # 총 9개

# 패널티 박스 정의 (정규화 좌표 기준, L->R 공격 방향)
# X > 0.84 (88.5m/105m), 0.2 < Y < 0.8
PENALTY_BOX_X = 0.84
PENALTY_BOX_Y_MIN = 0.2
PENALTY_BOX_Y_MAX = 0.8

# =========================================================
# 1. 데이터 로드
# =========================================================
def load_preprocessed_data():
    """전처리된 데이터를 로드합니다."""
    print(f"[1/4] 전처리 데이터 로드 중... ({PREPROCESSED_PATH})")
    df = pd.read_csv(PREPROCESSED_PATH, encoding='utf-8-sig')
    print(f"    -> 로드 완료: {len(df):,} rows")
    return df

# =========================================================
# 2. 성과 이벤트 식별 (슈팅 또는 박스 진입)
# =========================================================
def identify_outcome_events(df):
    """
    성과 이벤트를 식별합니다 (v3.5 고도화).
    - 팀별/기간별 공격 방향을 고려하여 상대 진영 박스 진입을 판단합니다.
    - 수비적 액션은 성과에서 제외합니다.
    - 성과 이벤트 간 최소 5개 액션의 간격(불응기)을 보장합니다.
    """
    print("[2/4] 성과 이벤트 식별 중 (v3.5: 불응기 도입 및 영역 정밀화)...")
    
    # 정렬 (연속성 및 순서 보장)
    df = df.sort_values(['game_id', 'period_id', 'time_seconds', 'action_id']).reset_index(drop=True)
    
    # 1. 팀별/기간별 GK 위치 분석으로 공격 방향 판별
    gk_stats = df[(df['position_name'] == 'GK') | (df['main_position'] == 'GK')].groupby(
        ['game_id', 'period_id', 'team_id']
    )['start_x'].mean().reset_index()
    gk_stats.rename(columns={'start_x': 'gk_x'}, inplace=True)
    
    df = df.merge(gk_stats, on=['game_id', 'period_id', 'team_id'], how='left')
    
    # 공격 방향 플래그 (True: L->R, False: R->L)
    # GK가 왼쪽(<0.5)에 있으면 L->R 공격. GK 데이터 없으면 L->R 가정(기본값)
    df['is_l_to_r'] = df['gk_x'].fillna(0.0) < 0.5
    
    # 2. 성과 조건 정의
    is_shot = df['spadl_type'].str.contains('shot', case=False, na=False)
    
    # 박스 진입 (선수가 상대 진영 박스 안에서 액션을 시작했는지 판별)
    # L->R 공격: start_x > 0.84, 0.2 < start_y < 0.8
    # R->L 공격: start_x < 0.16, 0.2 < start_y < 0.8
    is_in_box = (
        (df['is_l_to_r'] & (df['start_x'] > PENALTY_BOX_X) & (df['start_y'] > PENALTY_BOX_Y_MIN) & (df['start_y'] < PENALTY_BOX_Y_MAX)) |
        (~df['is_l_to_r'] & (df['start_x'] < (1.0 - PENALTY_BOX_X)) & (df['start_y'] > PENALTY_BOX_Y_MIN) & (df['start_y'] < PENALTY_BOX_Y_MAX))
    )
    
    # 수비적 액션 제외 (인터셉트, 클리어링, 태클, 키퍼 세이브 등)
    defensive_types = ['interception', 'tackle', 'clearance', 'keeper_save']
    is_defensive = df['spadl_type'].isin(defensive_types)
    is_valid_action = (df['spadl_type'] != 'non_action') & (~is_defensive)
    
    # 일시적 성과 후보
    df['temp_outcome'] = (is_shot | is_in_box) & is_valid_action
    
    # 3. 불응기(Refractory Period) 적용 - 최소 5개 액션 간격 (v3.5)
    # 사용자 요청: "최소 5개의 이벤트 다음에 성과 이벤트로 지정 가능"
    is_outcome = np.zeros(len(df), dtype=bool)
    last_outcome_idx = {} # 팀/기간별 마지막 성과 인덱스 저장
    
    for i in range(len(df)):
        if df.loc[i, 'temp_outcome']:
            key = (df.loc[i, 'game_id'], df.loc[i, 'period_id'], df.loc[i, 'team_id'])
            last_idx = last_outcome_idx.get(key, -999)
            
            # 이전 성과로부터 최소 5개 액션이 지났을 때만 새로운 성과로 인정
            if i - last_idx > 5:
                is_outcome[i] = True
                last_outcome_idx[key] = i
    
    df['is_outcome'] = is_outcome
    
    # 임시 컬럼 제거
    df.drop(columns=['temp_outcome', 'gk_x', 'is_l_to_r'], inplace=True)
    
    print(f"    -> 유니크 성과 이벤트: {df['is_outcome'].sum():,}개 (불응기 5회 적용)")
    return df

def extract_sequences(df):
    """
    성과 이벤트로부터 역추적하여 시퀀스를 추출합니다.
    - [v3.1] 시퀀스 간 액션 중복을 허용하지 않습니다 (No Overlap).
    - [v3.1] 시퀀스 단위로 공격 방향을 통합합니다.
    - [v3.1] 리시브 이벤트를 정밀 보정합니다.
    """
    print(f"[3/4] 시퀀스 추출 및 보정 중 (No Overlap)...")
    
    df = df.sort_values(['game_id', 'period_id', 'time_seconds', 'action_id']).reset_index(drop=True)
    outcome_indices = df[df['is_outcome']].index.tolist()
    
    # [v3.4] 방향 판별을 위한 GK 정보 재추출 (identify_outcome_events에서 썼던 것과 동일 로직)
    gk_stats = df[(df['position_name'] == 'GK') | (df['main_position'] == 'GK')].groupby(
        ['game_id', 'period_id', 'team_id']
    )['start_x'].mean().reset_index()
    
    used_action_ids = set()
    sequences = []
    seq_id = 0
    
    for outcome_idx in outcome_indices:
        current_team = df.loc[outcome_idx, 'team_id']
        
        # 1. 역추적 및 중복 체크
        if df.loc[outcome_idx, 'action_id'] in used_action_ids:
            continue
            
        seq_indices = []
        for i in range(outcome_idx, outcome_idx - TOTAL_SEQ_LEN, -1):
            if i < 0: break
            # 경기나 기간이 바뀌면 중단
            if df.loc[i, 'game_id'] != df.loc[outcome_idx, 'game_id'] or \
               df.loc[i, 'period_id'] != df.loc[outcome_idx, 'period_id']:
                break
            # 이미 다른 시퀀스에서 사용된 액션을 만나면 중단 (배타적 경계)
            if df.loc[i, 'action_id'] in used_action_ids:
                break
            seq_indices.append(i)
        
        if not seq_indices: continue
        seq_indices.reverse()
        seq_df = df.loc[seq_indices].copy()
        
        # [v3.9] 최소 시퀀스 길이 조건 추가 (최소 5개 이벤트)
        if len(seq_df) < 5:
            continue
        
        # [v3.6] 의미 있는 전술 패턴 확보를 위해 최소 3개의 패스(순수 Pass만, Cross 제외)가 포함되어야 함
        # - 단순 롱볼 한 번이나 드리블 후 슛 같은 단발성 패턴은 제외
        pass_count = (seq_df['type_name'] == 'Pass').sum()
        if pass_count < 3:
            continue
        
        # 사용된 액션 등록
        used_action_ids.update(seq_df['action_id'].tolist())
        
        # 3. 공격 방향 정규화 (L->R로 통일)
        gk_row = gk_stats[(gk_stats['game_id'] == seq_df['game_id'].iloc[0]) & 
                          (gk_stats['period_id'] == seq_df['period_id'].iloc[0]) & 
                          (gk_stats['team_id'] == current_team)]
        
        if not gk_row.empty and gk_row['start_x'].iloc[0] > 0.5:
            # R->L 공격이므로 반전하여 L->R로 만듦
            for col in ['start_x', 'end_x']: seq_df[col] = 1.0 - seq_df[col]
            for col in ['start_y', 'end_y']: seq_df[col] = 1.0 - seq_df[col]
        elif gk_row.empty:
            # GK 정보 부족 시 흐름 기반 (추천하지 않으나 폴백)
            if seq_df['start_x'].iloc[0] > seq_df['start_x'].iloc[-1]:
                for col in ['start_x', 'end_x']: seq_df[col] = 1.0 - seq_df[col]
                for col in ['start_y', 'end_y']: seq_df[col] = 1.0 - seq_df[col]

        # 3. 리시브 로직 보정 (v3.1)
        # 리시브는 제자리로, 이동은 Carry로
        refined_rows = []
        rows = [row for _, row in seq_df.iterrows()]
        
        i = 0
        while i < len(rows):
            curr = rows[i].copy()
            if curr['type_name'] == 'Pass Received':
                dist = np.sqrt((curr['start_x'] - curr['end_x'])**2 + (curr['start_y'] - curr['end_y'])**2)
                is_moving = dist > 1e-5
                
                next_node = rows[i+1] if i + 1 < len(rows) else None
                
                # [A-1] 중복 리시브 삭제
                if next_node is not None and \
                   abs(curr['start_x'] - next_node['start_x']) < 1e-5 and \
                   abs(curr['end_x'] - next_node['end_x']) < 1e-5:
                    i += 1
                    continue
                
                # [A-2] 이동 리시브 처리
                if is_moving:
                    if next_node is not None and next_node['type_name'] == 'Carry':
                        # 다음 Carry의 시작을 이 리시브의 시작으로 땡기고 리시브 삭제
                        rows[i+1]['start_x'] = curr['start_x']
                        rows[i+1]['start_y'] = curr['start_y']
                        i += 1
                        continue
                    else:
                        # 리시브를 Carry로 변환
                        curr['type_name'] = 'Carry'
                        curr['spadl_type'] = 'dribble'
                
                # [B] 정적 리시브 간극 보정
                elif next_node is not None and next_node['type_name'] != 'Carry':
                    gap = np.sqrt((curr['end_x'] - next_node['start_x'])**2 + (curr['end_y'] - next_node['start_y'])**2)
                    if gap > 1e-5:
                        # 리시브를 Carry로 변환하여 종료 지점을 다음 액션 시작점으로 보정
                        curr['type_name'] = 'Carry'
                        curr['spadl_type'] = 'dribble'
                        curr['end_x'] = next_node['start_x']
                        curr['end_y'] = next_node['start_y']
                
                # 만약 정적 리시브이고 위 조건에 안걸리면 (삭제 대상)
                if curr['type_name'] == 'Pass Received' and not is_moving:
                    # 사용자 요청: 리시브 이벤트 시작, 종료 위치가 같은 경우는 모두 삭제
                    i += 1
                    continue

            refined_rows.append(curr)
            i += 1
            
        if not refined_rows: continue
        
        refined_seq = pd.DataFrame(refined_rows)
        
        # 4. 후처리 (포지션 재배정 및 ID 부여)
        seq_len = len(refined_seq)
        refined_seq['seq_position'] = range(seq_len, 0, -1)
        refined_seq['seq_phase'] = refined_seq['seq_position'].apply(lambda x: 'attack' if x <= ATTACK_LEN else 'buildup')
        refined_seq['sequence_id'] = seq_id
        
        # 메타데이터 추출 (원본 outcome 기준)
        orig_outcome = df.loc[outcome_idx]
        refined_seq['outcome_type'] = orig_outcome['spadl_type']
        refined_seq['outcome_result'] = orig_outcome['spadl_result']
        
        sequences.append(refined_seq)
        seq_id += 1
    
    if sequences:
        result_df = pd.concat(sequences, ignore_index=True)
        print(f"    -> 최종 추출: {seq_id:,}개 시퀀스, {len(result_df):,} rows")
        return result_df
    else:
        return pd.DataFrame()
    
# =========================================================
# 4. 시퀀스별 통계 및 템포 집계
# =========================================================
def aggregate_sequence_stats(seq_df):
    """시퀀스별 통계를 집계합니다."""
    print("[4/4] 시퀀스 통계 집계 중...")
    
    # 시퀀스별 집계
    stats = seq_df.groupby('sequence_id').agg({
        'game_id': 'first',
        'team_id': 'first',
        'outcome_type': 'first',
        'outcome_result': 'first',
        'dt': 'sum',           # 총 소요 시간
        'distance': 'sum',     # 총 이동 거리
        'speed': 'mean',       # 평균 속도 (템포)
        'start_x': 'first',    # 시작 X (빌드업 시작점)
        'start_y': 'first',    # 시작 Y
        'end_x': 'last',       # 종료 X (성과 지점)
        'end_y': 'last',       # 종료 Y
        'seq_position': 'count'  # 시퀀스 길이
    }).rename(columns={'seq_position': 'seq_length'})
    
    print(f"    -> 집계 완료: {len(stats):,}개 시퀀스 통계")
    return stats

# =========================================================
# 메인 파이프라인
# =========================================================
def run_sequence_extraction():
    """시퀀스 추출 파이프라인을 실행합니다."""
    print("=" * 60)
    print("deTACTer 시퀀스 추출 시작")
    print(f"  - 공격 시퀀스 길이: {ATTACK_LEN}")
    print(f"  - 빌드업 시퀀스 길이: {BUILDUP_LEN}")
    print("=" * 60)
    
    # 1. 데이터 로드
    df = load_preprocessed_data()
    
    # 2. 성과 이벤트 식별
    df = identify_outcome_events(df)
    
    # 3. 시퀀스 추출
    seq_df = extract_sequences(df)
    
    if len(seq_df) == 0:
        print("시퀀스가 추출되지 않았습니다. 종료합니다.")
        return None, None
    
    # 4. 통계 집계
    stats = aggregate_sequence_stats(seq_df)
    
    # 저장
    print("=" * 60)
    print(f"저장 중... ({SEQUENCES_OUTPUT_PATH})")
    seq_df.to_csv(SEQUENCES_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    stats_path = SEQUENCES_OUTPUT_PATH.replace('.csv', '_stats.csv')
    stats.to_csv(stats_path, encoding='utf-8-sig')
    
    print(f"시퀀스 저장 완료: {len(seq_df):,} rows")
    print(f"통계 저장 완료: {stats_path}")
    print("=" * 60)
    
    return seq_df, stats

if __name__ == "__main__":
    run_sequence_extraction()
