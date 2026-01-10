# =========================================================
# tactical_clustering.py
# deTACTer 프로젝트용 전술 클러스터링 모듈
# =========================================================
# 주요 기능:
# 1. DTW 거리 행렬 계산 (시퀀스 간 유사도)
# 2. OPTICS 클러스터링 적용
# 3. Agglomerative 클러스터링 적용
# 4. 클러스터별 통계 및 템포 집계
# =========================================================

import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
from sklearn.cluster import OPTICS, AgglomerativeClustering
import sys
import warnings
warnings.filterwarnings('ignore')

# 터미널 한글 출력 인코딩 설정 (Windows)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# DTW 라이브러리 (fastdtw 사용)
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    DTW_AVAILABLE = True
except ImportError:
    print("Warning: fastdtw not installed. Run: pip install fastdtw")
    DTW_AVAILABLE = False

# =========================================================
# 경로 설정
# =========================================================
SEQUENCES_PATH = 'c:/Users/Public/Documents/DIK/deTACTer/data/refined/attack_sequences.csv'
STATS_PATH = 'c:/Users/Public/Documents/DIK/deTACTer/data/refined/attack_sequences_stats.csv'
OUTPUT_DIR = 'c:/Users/Public/Documents/DIK/deTACTer/data/refined/'

# 클러스터링 파라미터
MAX_SEQUENCES = 2000  # DTW 계산 시간을 위해 샘플링 (필요시 조정)
OPTICS_MIN_SAMPLES = 5
OPTICS_XI = 0.05
AGGLOM_N_CLUSTERS = 10  # 초기값, 실험 후 조정

# =========================================================
# 1. 데이터 로드
# =========================================================
def load_sequences():
    """시퀀스 데이터를 로드합니다."""
    print(f"[1/5] 시퀀스 데이터 로드 중...")
    seq_df = pd.read_csv(SEQUENCES_PATH, encoding='utf-8-sig')
    stats_df = pd.read_csv(STATS_PATH, encoding='utf-8-sig')
    print(f"    -> 로드 완료: {stats_df.shape[0]:,}개 시퀀스")
    return seq_df, stats_df

# =========================================================
# 2. 시퀀스를 2D 경로로 변환
# =========================================================
def sequence_to_path(seq_df, sequence_id):
    """시퀀스를 (x, y) 좌표 배열로 변환합니다."""
    seq = seq_df[seq_df['sequence_id'] == sequence_id].sort_values('seq_position', ascending=False)
    path = seq[['start_x', 'start_y']].values
    return path

def prepare_paths(seq_df, stats_df, max_sequences=MAX_SEQUENCES):
    """모든 시퀀스를 경로로 변환합니다."""
    print(f"[2/5] 시퀀스 경로 변환 중 (최대 {max_sequences}개)...")
    
    # 샘플링 (너무 많으면 DTW 계산이 오래 걸림)
    unique_seqs = stats_df['sequence_id'].unique()
    if len(unique_seqs) > max_sequences:
        np.random.seed(42)
        sampled_seqs = np.random.choice(unique_seqs, max_sequences, replace=False)
    else:
        sampled_seqs = unique_seqs
    
    paths = {}
    for seq_id in sampled_seqs:
        path = sequence_to_path(seq_df, seq_id)
        if len(path) >= 3:  # 최소 3개 포인트 필요
            paths[seq_id] = path
    
    print(f"    -> 변환 완료: {len(paths):,}개 경로")
    return paths

# =========================================================
# 3. DTW 거리 행렬 계산
# =========================================================
def compute_dtw_matrix(paths):
    """DTW 거리 행렬을 계산합니다."""
    if not DTW_AVAILABLE:
        print("    [ERROR] fastdtw 라이브러리가 설치되지 않았습니다.")
        return None, None
    
    print(f"[3/5] DTW 거리 행렬 계산 중 ({len(paths)}x{len(paths)})...")
    
    seq_ids = list(paths.keys())
    n = len(seq_ids)
    dist_matrix = np.zeros((n, n))
    
    total = n * (n - 1) // 2
    count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            path_i = paths[seq_ids[i]]
            path_j = paths[seq_ids[j]]
            
            # DTW 거리 계산
            distance, _ = fastdtw(path_i, path_j, dist=euclidean)
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
            
            count += 1
            if count % 10000 == 0:
                print(f"        진행: {count}/{total} ({100*count/total:.1f}%)")
    
    print(f"    -> DTW 계산 완료")
    return dist_matrix, seq_ids

# =========================================================
# 4. 클러스터링 (OPTICS + Agglomerative)
# =========================================================
def run_optics_clustering(dist_matrix, seq_ids):
    """OPTICS 클러스터링을 수행합니다."""
    print(f"[4a/5] OPTICS 클러스터링 수행 중...")
    
    optics = OPTICS(
        min_samples=OPTICS_MIN_SAMPLES, 
        xi=OPTICS_XI,
        metric='precomputed'
    )
    labels = optics.fit_predict(dist_matrix)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"    -> OPTICS 완료: {n_clusters}개 클러스터, {n_noise}개 노이즈")
    
    return pd.DataFrame({
        'sequence_id': seq_ids,
        'optics_cluster': labels
    })

def run_agglomerative_clustering(dist_matrix, seq_ids, n_clusters=AGGLOM_N_CLUSTERS):
    """Agglomerative 클러스터링을 수행합니다."""
    print(f"[4b/5] Agglomerative 클러스터링 수행 중 (n_clusters={n_clusters})...")
    
    agglom = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = agglom.fit_predict(dist_matrix)
    
    print(f"    -> Agglomerative 완료: {n_clusters}개 클러스터")
    
    return pd.DataFrame({
        'sequence_id': seq_ids,
        'agglom_cluster': labels
    })

# =========================================================
# 5. 클러스터별 통계 집계
# =========================================================
def aggregate_cluster_stats(stats_df, cluster_df, cluster_col):
    """클러스터별 통계를 집계합니다."""
    merged = stats_df.merge(cluster_df, on='sequence_id', how='inner')
    
    cluster_stats = merged.groupby(cluster_col).agg({
        'sequence_id': 'count',
        'speed': 'mean',           # 평균 템포
        'dt': 'mean',              # 평균 소요 시간
        'distance': 'mean',        # 평균 이동 거리
        'outcome_result': lambda x: (x == 'success').mean()  # 성공률
    }).rename(columns={
        'sequence_id': 'count',
        'speed': 'avg_tempo',
        'dt': 'avg_duration',
        'distance': 'avg_distance',
        'outcome_result': 'success_rate'
    })
    
    return cluster_stats

# =========================================================
# 메인 파이프라인
# =========================================================
def run_clustering():
    """클러스터링 파이프라인을 실행합니다."""
    print("=" * 60)
    print("deTACTer 전술 클러스터링 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    seq_df, stats_df = load_sequences()
    
    # 2. 경로 변환
    paths = prepare_paths(seq_df, stats_df)
    
    # 3. DTW 거리 행렬
    dist_matrix, seq_ids = compute_dtw_matrix(paths)
    
    if dist_matrix is None:
        print("DTW 계산 실패. 종료합니다.")
        return
    
    # 4. 클러스터링
    optics_result = run_optics_clustering(dist_matrix, seq_ids)
    agglom_result = run_agglomerative_clustering(dist_matrix, seq_ids)
    
    # 결과 병합
    cluster_result = optics_result.merge(agglom_result, on='sequence_id')
    
    # 5. 통계 집계
    print("[5/5] 클러스터 통계 집계 중...")
    optics_stats = aggregate_cluster_stats(stats_df, cluster_result, 'optics_cluster')
    agglom_stats = aggregate_cluster_stats(stats_df, cluster_result, 'agglom_cluster')
    
    # 저장
    print("=" * 60)
    print("저장 중...")
    
    cluster_result.to_csv(OUTPUT_DIR + 'cluster_labels.csv', index=False, encoding='utf-8-sig')
    optics_stats.to_csv(OUTPUT_DIR + 'optics_cluster_stats.csv', encoding='utf-8-sig')
    agglom_stats.to_csv(OUTPUT_DIR + 'agglom_cluster_stats.csv', encoding='utf-8-sig')
    
    # DTW 행렬도 저장 (나중에 재사용 가능)
    np.save(OUTPUT_DIR + 'dtw_distance_matrix.npy', dist_matrix)
    
    print(f"클러스터 레이블: {OUTPUT_DIR}cluster_labels.csv")
    print(f"OPTICS 통계: {OUTPUT_DIR}optics_cluster_stats.csv")
    print(f"Agglomerative 통계: {OUTPUT_DIR}agglom_cluster_stats.csv")
    print("=" * 60)
    
    # 요약 출력
    print("\n[OPTICS 클러스터 요약]")
    print(optics_stats.sort_values('count', ascending=False).head(10))
    print("\n[Agglomerative 클러스터 요약]")
    print(agglom_stats.sort_values('count', ascending=False).head(10))
    
    return cluster_result, optics_stats, agglom_stats

if __name__ == "__main__":
    run_clustering()
