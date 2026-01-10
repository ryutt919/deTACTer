import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print('DTW 거리 행렬 로드 중...')
dtw_matrix = np.load('c:/Users/Public/Documents/DIK/deTACTer/data/refined/dtw_distance_matrix.npy')
print(f'행렬 크기: {dtw_matrix.shape}')

print('\n최적 클러스터 수 탐색 (Silhouette Score)...')
print('-' * 40)

k_range = range(3, 25)
scores = []
best_score = -1
best_k = 3

for k in k_range:
    agglom = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    labels = agglom.fit_predict(dtw_matrix)
    score = silhouette_score(dtw_matrix, labels, metric='precomputed')
    scores.append(score)
    
    marker = ''
    if score > best_score:
        best_score = score
        best_k = k
        marker = ' <-- best'
    
    print(f'k={k:2d}: Silhouette = {score:.4f}{marker}')

print('-' * 40)
print(f'최적 클러스터 수: k = {best_k}')
print(f'Silhouette Score: {best_score:.4f}')
