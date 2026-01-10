# =========================================================
# analyze_sequences.py
# deTACTer 프로젝트용 버전별 시퀀스 데이터 분석 스크립트
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import sys

# 터미널 한글 출력 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_analysis(version):
    """특정 버전의 시퀀스 데이터를 분석하고 결과를 저장합니다."""
    base_dir = 'c:/Users/Public/Documents/DIK/deTACTer'
    data_dir = f"{base_dir}/data/refined/{version}/"
    output_dir = f"{base_dir}/results/anal/{version}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 로드
    try:
        seq_df_path = data_dir + 'attack_sequences.csv'
        stats_df_path = data_dir + 'attack_sequences_stats.csv'
        
        if not os.path.exists(seq_df_path) or not os.path.exists(stats_df_path):
            print(f"  [건너뜀] {version} 데이터가 존재하지 않습니다.")
            return

        seq_df = pd.read_csv(seq_df_path, encoding='utf-8-sig')
        stats_df = pd.read_csv(stats_df_path, encoding='utf-8-sig', index_col=0)
    except Exception as e:
        print(f"  [오류] 데이터 로드 중 문제 발생: {e}")
        return

    # 팀 이름 매핑 (seq_df에서 team_id와 team_name_ko 관계 추출)
    if 'team_name_ko' in seq_df.columns:
        team_map = seq_df.groupby('team_id')['team_name_ko'].first().to_dict()
    else:
        team_map = {}
        
    stats_df['team_name'] = stats_df['team_id'].map(team_map).fillna(stats_df['team_id'])
    
    # 1. 팀별 시퀀스 수 분석
    team_counts = stats_df['team_name'].value_counts().reset_index()
    team_counts.columns = ['team_name', 'sequence_count']
    team_counts.to_csv(f"{output_dir}team_sequence_counts.csv", index=False, encoding='utf-8-sig')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=team_counts, x='team_name', y='sequence_count', palette='viridis')
    plt.title(f'[{version}] 팀별 시퀀스 생성 수', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel('시퀀스 수')
    plt.savefig(f"{output_dir}01_team_counts.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 팀별 성과 성공률 (outcome_result == 'success')
    # SPADL 결과가 1이면 성공으로 간주 (또는 outcome_result 문자열 체크)
    if 'outcome_result' in stats_df.columns:
        stats_df['is_success'] = stats_df['outcome_result'].apply(lambda x: 1 if str(x).lower() in ['success', '1', '1.0'] else 0)
    else:
        stats_df['is_success'] = 0
        
    team_success = stats_df.groupby('team_name')['is_success'].mean().reset_index()
    team_success.columns = ['team_name', 'success_rate']
    team_success.to_csv(f"{output_dir}team_success_rate.csv", index=False, encoding='utf-8-sig')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=team_success, x='team_name', y='success_rate', palette='magma')
    plt.title(f'[{version}] 팀별 시퀀스 성공률 (슈팅/박스진입 기준)', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel('성공률')
    plt.savefig(f"{output_dir}02_team_success_rate.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 추가 분석: 템포(평균 속도) 및 이동 거리 분포
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=stats_df, x='team_name', y='speed', palette='Set2')
    plt.title('팀별 공격 템포 (평균 속도)')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=stats_df, x='team_name', y='distance', palette='Set3')
    plt.title('팀별 시퀀스 총 이동 거리')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}03_tempo_distance.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. 추가 분석: 시퀀스 시작 위치 히트맵 (전략적 시작점)
    plt.figure(figsize=(10, 7))
    # 축구장 배경 모사 (단순 사각형)
    plt.plot([0, 1], [0, 0], 'w-', alpha=0.5)
    plt.plot([0, 1], [1, 1], 'w-', alpha=0.5)
    plt.plot([0, 0], [0, 1], 'w-', alpha=0.5)
    plt.plot([1, 1], [0, 1], 'w-', alpha=0.5)
    plt.plot([0.5, 0.5], [0, 1], 'w--', alpha=0.3)
    
    sns.kdeplot(data=stats_df, x='start_x', y='start_y', fill=True, cmap='Greens', alpha=0.8, levels=10)
    plt.title(f'[{version}] 공격 시퀀스 시작 위치 밀도 (Heatmap)', fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.gca().set_facecolor('#1a472a') # 진녹색 배경
    plt.savefig(f"{output_dir}04_start_location_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 5. 추가 분석: 시퀀스 당 액션 구성 비율
    # 시퀀스 아이디별 패스, 드리블 비율 계산
    action_counts = seq_df.groupby(['sequence_id', 'type_name']).size().unstack(fill_value=0)
    # 비율로 변환
    action_pct = action_counts.div(action_counts.sum(axis=1), axis=0)
    
    # 팀 정보 병합
    action_pct = action_pct.join(stats_df[['team_name']], how='left')
    team_action_profile = action_pct.groupby('team_name').mean()
    
    plt.figure(figsize=(12, 6))
    team_action_profile.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
    plt.title(f'[{version}] 팀별 시퀀스 내 액션 구성 비율 (평균)', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{output_dir}05_action_composition.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 완료: {output_dir}")

if __name__ == "__main__":
    # 최신 버전들 위주로 분석
    versions_to_analyze = ['v4.1', 'v4.2', 'v4.4', 'v4.5']
    
    # config.yaml에서 현재 버전 확인
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            curr_v = yaml.safe_load(f).get('version')
            if curr_v not in versions_to_analyze:
                versions_to_analyze.append(curr_v)
    except:
        pass
        
    for v in versions_to_analyze:
        run_analysis(v)
