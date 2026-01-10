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

# 경로 설정
PREPROCESSED_PATH = 'c:/Users/Public/Documents/DIK/deTACTer/data/refined/preprocessed_data.csv'
SEQUENCES_OUTPUT_PATH = 'c:/Users/Public/Documents/DIK/deTACTer/data/refined/attack_sequences.csv'

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
    성과 이벤트를 식별합니다.
    - 슈팅(shot) 이벤트
    - 패널티 박스 진입 이벤트 (end_x > 0.84 and 0.2 < end_y < 0.8)
    """
    print("[2/4] 성과 이벤트 식별 중...")
    
    # 슈팅 이벤트
    is_shot = df['spadl_type'].str.contains('shot', case=False, na=False)
    
    # 박스 진입 이벤트 (end 좌표 기준)
    is_box_entry = (
        (df['end_x'] > PENALTY_BOX_X) & 
        (df['end_y'] > PENALTY_BOX_Y_MIN) & 
        (df['end_y'] < PENALTY_BOX_Y_MAX)
    )
    
    # non_action 제외
    is_valid_action = df['spadl_type'] != 'non_action'
    
    # 성과 이벤트 마킹
    df['is_outcome'] = (is_shot | is_box_entry) & is_valid_action
    
    outcome_count = df['is_outcome'].sum()
    print(f"    -> 성과 이벤트: {outcome_count:,}개 (슈팅: {is_shot.sum():,}, 박스진입: {is_box_entry.sum():,})")
    
    return df

# =========================================================
# 3. 역추적 기반 시퀀스 추출
# =========================================================
def extract_sequences(df):
    """
    성과 이벤트로부터 역추적하여 시퀀스를 추출합니다.
    - 공격 시퀀스: 성과 시점부터 역순 1~ATTACK_LEN개
    - 빌드업 시퀀스: 그 이전 BUILDUP_LEN개
    """
    print(f"[3/4] 시퀀스 추출 중 (공격: {ATTACK_LEN}개, 빌드업: {BUILDUP_LEN}개)...")
    
    # 게임별, 피리어드별로 정렬
    df = df.sort_values(['game_id', 'period_id', 'time_seconds']).reset_index(drop=True)
    
    # 성과 이벤트의 인덱스 추출
    outcome_indices = df[df['is_outcome']].index.tolist()
    
    sequences = []
    seq_id = 0
    
    for outcome_idx in outcome_indices:
        # 역추적 범위 계산
        start_idx = max(0, outcome_idx - TOTAL_SEQ_LEN + 1)
        end_idx = outcome_idx + 1
        
        # 시퀀스 추출 (outcome 포함)
        seq_df = df.iloc[start_idx:end_idx].copy()
        
        # 같은 game_id, period_id인지 확인 (다르면 건너뜀)
        if seq_df['game_id'].nunique() > 1 or seq_df['period_id'].nunique() > 1:
            continue
        
        # 시퀀스 내 위치 라벨링 (역순 기준)
        seq_len = len(seq_df)
        positions = list(range(seq_len, 0, -1))  # [n, n-1, ..., 2, 1]
        seq_df['seq_position'] = positions
        
        # 공격/빌드업 구분
        seq_df['seq_phase'] = seq_df['seq_position'].apply(
            lambda x: 'attack' if x <= ATTACK_LEN else 'buildup'
        )
        
        # 시퀀스 ID 부여
        seq_df['sequence_id'] = seq_id
        
        # 메타데이터 추가
        seq_df['outcome_type'] = df.loc[outcome_idx, 'spadl_type']
        seq_df['outcome_result'] = df.loc[outcome_idx, 'spadl_result']
        seq_df['team_id'] = df.loc[outcome_idx, 'team_id']
        
        sequences.append(seq_df)
        seq_id += 1
    
    if sequences:
        result_df = pd.concat(sequences, ignore_index=True)
        print(f"    -> 추출 완료: {seq_id:,}개 시퀀스, {len(result_df):,} rows")
        return result_df
    else:
        print("    -> 추출된 시퀀스 없음!")
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
