# =========================================================
# visualize_tactics.py
# deTACTer 프로젝트용 전술 시각화 모듈
# =========================================================
# 주요 기능:
# 1. 클러스터별 대표 시퀀스 시각화
# 2. 전술 경로 오버레이 플롯
# 3. 클러스터 통계 대시보드
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import sys
import os

# 터미널 한글 출력 인코딩 설정 (Windows)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 설정 및 경로 로드 (v3.2)
# =========================================================
import yaml

CONFIG_PATH = 'c:/Users/Public/Documents/DIK/deTACTer/config.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

VERSION = config.get('version', 'v3.1')
BASE_DIR = 'c:/Users/Public/Documents/DIK/deTACTer'

# 입력 및 출력 경로 설정 (버전별 폴더)
DATA_DIR = f"{BASE_DIR}/data/refined/{VERSION}/"
OUTPUT_DIR = f"{BASE_DIR}/results/visualizations/{VERSION}/"

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 축구장 그리기 함수
# =========================================================
def draw_pitch(ax, pitch_color='#1a472a', line_color='white', alpha=0.8):
    """정규화된 좌표(0~1)에 맞춘 축구장을 그립니다."""
    
    # 배경
    ax.set_facecolor(pitch_color)
    
    # 외곽선
    ax.plot([0, 1], [0, 0], color=line_color, linewidth=2)
    ax.plot([0, 1], [1, 1], color=line_color, linewidth=2)
    ax.plot([0, 0], [0, 1], color=line_color, linewidth=2)
    ax.plot([1, 1], [0, 1], color=line_color, linewidth=2)
    
    # 중앙선
    ax.plot([0.5, 0.5], [0, 1], color=line_color, linewidth=1.5)
    
    # 센터 서클
    center_circle = plt.Circle((0.5, 0.5), 0.087, fill=False, color=line_color, linewidth=1.5)
    ax.add_patch(center_circle)
    
    # 패널티 박스 (왼쪽 - 수비)
    ax.add_patch(patches.Rectangle((0, 0.211), 0.157, 0.578, fill=False, color=line_color, linewidth=1.5))
    # 골 에어리어 (왼쪽)
    ax.add_patch(patches.Rectangle((0, 0.368), 0.052, 0.264, fill=False, color=line_color, linewidth=1.5))
    
    # 패널티 박스 (오른쪽 - 공격)
    ax.add_patch(patches.Rectangle((0.843, 0.211), 0.157, 0.578, fill=False, color=line_color, linewidth=1.5))
    # 골 에어리어 (오른쪽)
    ax.add_patch(patches.Rectangle((0.948, 0.368), 0.052, 0.264, fill=False, color=line_color, linewidth=1.5))
    
    # 골대
    ax.plot([0, 0], [0.456, 0.544], color='white', linewidth=4)
    ax.plot([1, 1], [0.456, 0.544], color='white', linewidth=4)
    
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return ax

# =========================================================
# 시퀀스 경로 시각화
# =========================================================
def plot_sequence_path(ax, seq_df, sequence_id, color='yellow', alpha=0.8, linewidth=2):
    """단일 시퀀스의 경로를 그립니다."""
    seq = seq_df[seq_df['sequence_id'] == sequence_id].sort_values('seq_position', ascending=False)
    
    x = seq['start_x'].values
    y = seq['start_y'].values
    
    # 경로 선
    ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth, marker='o', markersize=4)
    
    # 시작점 (초록)
    ax.scatter([x[0]], [y[0]], color='lime', s=100, zorder=5, edgecolor='white', linewidth=2)
    
    # 종료점 (빨강)
    ax.scatter([x[-1]], [y[-1]], color='red', s=100, zorder=5, edgecolor='white', linewidth=2, marker='*')
    
    return ax

# =========================================================
# 클러스터 대표 시퀀스 시각화
# =========================================================
def visualize_cluster_samples(seq_df, cluster_df, cluster_col, n_samples=5):
    """각 클러스터에서 샘플 시퀀스들을 시각화합니다."""
    print(f"[시각화] {cluster_col} 클러스터별 샘플 시퀀스 시각화 중...")
    
    clusters = cluster_df[cluster_col].unique()
    clusters = [c for c in clusters if c != -1]  # 노이즈 제외
    
    for cluster_id in clusters[:10]:  # 상위 10개 클러스터만
        fig, ax = plt.subplots(figsize=(12, 8))
        draw_pitch(ax)
        
        # 해당 클러스터의 시퀀스들
        seq_ids = cluster_df[cluster_df[cluster_col] == cluster_id]['sequence_id'].values
        sample_ids = seq_ids[:n_samples]
        
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(sample_ids)))
        
        for i, seq_id in enumerate(sample_ids):
            plot_sequence_path(ax, seq_df, seq_id, color=colors[i], alpha=0.7)
        
        ax.set_title(f'{cluster_col} 클러스터 {cluster_id} (샘플 {len(sample_ids)}개)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}{cluster_col}_cluster_{cluster_id}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"    -> 저장 완료: {OUTPUT_DIR}")

# =========================================================
# 클러스터 통계 대시보드
# =========================================================
def plot_cluster_stats_dashboard(optics_stats, agglom_stats):
    """클러스터 통계 대시보드를 생성합니다."""
    print("[시각화] 클러스터 통계 대시보드 생성 중...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # OPTICS 통계
    if len(optics_stats) > 0:
        optics_valid = optics_stats[optics_stats.index != -1]
        
        # 클러스터 크기
        axes[0, 0].barh(range(len(optics_valid)), optics_valid['count'].values, color='steelblue')
        axes[0, 0].set_ylabel('클러스터 ID')
        axes[0, 0].set_xlabel('시퀀스 수')
        axes[0, 0].set_title('OPTICS: 클러스터 크기')
        
        # 평균 템포
        axes[0, 1].barh(range(len(optics_valid)), optics_valid['avg_tempo'].values, color='coral')
        axes[0, 1].set_ylabel('클러스터 ID')
        axes[0, 1].set_xlabel('평균 템포 (속도)')
        axes[0, 1].set_title('OPTICS: 평균 템포')
        
        # 성공률
        axes[0, 2].barh(range(len(optics_valid)), optics_valid['success_rate'].values * 100, color='seagreen')
        axes[0, 2].set_ylabel('클러스터 ID')
        axes[0, 2].set_xlabel('성공률 (%)')
        axes[0, 2].set_title('OPTICS: 성공률')
    
    # Agglomerative 통계
    if len(agglom_stats) > 0:
        # 클러스터 크기
        axes[1, 0].barh(range(len(agglom_stats)), agglom_stats['count'].values, color='steelblue')
        axes[1, 0].set_ylabel('클러스터 ID')
        axes[1, 0].set_xlabel('시퀀스 수')
        axes[1, 0].set_title('Agglomerative: 클러스터 크기')
        
        # 평균 템포
        axes[1, 1].barh(range(len(agglom_stats)), agglom_stats['avg_tempo'].values, color='coral')
        axes[1, 1].set_ylabel('클러스터 ID')
        axes[1, 1].set_xlabel('평균 템포 (속도)')
        axes[1, 1].set_title('Agglomerative: 평균 템포')
        
        # 성공률
        axes[1, 2].barh(range(len(agglom_stats)), agglom_stats['success_rate'].values * 100, color='seagreen')
        axes[1, 2].set_ylabel('클러스터 ID')
        axes[1, 2].set_xlabel('성공률 (%)')
        axes[1, 2].set_title('Agglomerative: 성공률')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}cluster_stats_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    -> 저장 완료: {OUTPUT_DIR}cluster_stats_dashboard.png")

# =========================================================
# 메인 파이프라인
# =========================================================
def run_visualization():
    """시각화 파이프라인을 실행합니다."""
    print("=" * 60)
    print("deTACTer 전술 시각화 시작")
    print("=" * 60)
    
    # 데이터 로드
    print("[1/3] 데이터 로드 중...")
    seq_df = pd.read_csv(DATA_DIR + 'attack_sequences.csv', encoding='utf-8-sig')
    
    try:
        cluster_df = pd.read_csv(DATA_DIR + 'cluster_labels.csv', encoding='utf-8-sig')
        optics_stats = pd.read_csv(DATA_DIR + 'optics_cluster_stats.csv', encoding='utf-8-sig', index_col=0)
        agglom_stats = pd.read_csv(DATA_DIR + 'agglom_cluster_stats.csv', encoding='utf-8-sig', index_col=0)
    except FileNotFoundError:
        print("    [경고] 클러스터링 결과 파일이 없습니다. 클러스터링을 먼저 실행해주세요.")
        return
    
    print(f"    -> 로드 완료: {len(seq_df):,} rows")
    
    # 클러스터별 샘플 시각화
    print("[2/3] 클러스터별 샘플 시각화...")
    visualize_cluster_samples(seq_df, cluster_df, 'optics_cluster')
    visualize_cluster_samples(seq_df, cluster_df, 'agglom_cluster')
    
    # 통계 대시보드
    print("[3/3] 통계 대시보드 생성...")
    plot_cluster_stats_dashboard(optics_stats, agglom_stats)
    
    print("=" * 60)
    print("시각화 완료!")
    print(f"결과 폴더: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    run_visualization()
