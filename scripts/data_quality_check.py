import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

# 설정 로드
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

RAW_DATA_PATH = config['data']['raw_data_path']
OUTPUT_DIR = 'results/quality_check'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. 데이터 로드 (UTF-8-SIG 적용)
print(f"Loading data from {RAW_DATA_PATH}...")
try:
    df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8-sig')
    print("Data loaded successfully with utf-8-sig.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# 2. 기본 정보 요약
with open(f"{OUTPUT_DIR}/summary.txt", "w", encoding='utf-8') as f:
    f.write("=== Data Quality Check Summary ===\n\n")
    f.write(f"Total Rows: {len(df)}\n")
    f.write(f"Total Columns: {len(df.columns)}\n\n")
    
    f.write("--- Missing Values ---\n")
    missing = df.isnull().sum()
    f.write(missing[missing > 0].to_string())
    f.write("\n\n")
    
    f.write("--- Data Types ---\n")
    f.write(df.dtypes.to_string())
    f.write("\n\n")
    
    f.write("--- Unique Values (Categorical) ---\n")
    for col in ['type_name', 'result_name', 'position_name', 'main_position']:
        f.write(f"{col}: {df[col].nunique()} unique values\n")

# 3. 결측치 및 이상치 분석
print("Analyzing missing values and outliers...")

# 좌표 범위 확인 (일반적으로 0-100 또는 0-105/0-68)
coord_cols = ['start_x', 'start_y', 'end_x', 'end_y']
coord_stats = df[coord_cols].describe()
coord_stats.to_csv(f"{OUTPUT_DIR}/coordinate_stats.csv")

# 4. 시각화
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows용 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False

# 시각화: 이벤트 타입 분포
plt.figure(figsize=(12, 6))
df['type_name'].value_counts().plot(kind='bar')
plt.title('이벤트 타입 분포')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/event_type_distribution.png")

# 시각화: 결과(성공/실패) 분포
plt.figure(figsize=(8, 6))
df['result_name'].value_counts(dropna=False).plot(kind='pie', autopct='%1.1f%%')
plt.title('이벤트 결과 분포 (NaN 포함)')
plt.savefig(f"{OUTPUT_DIR}/result_distribution.png")

# 시각화: 좌표 데이터 분포 (Heatmap용 산점도)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.sample(min(5000, len(df))), x='start_x', y='start_y', alpha=0.1)
plt.title('시작 위치 분포 (샘플 5000건)')
plt.savefig(f"{OUTPUT_DIR}/coordinate_scatter.png")

# 5. 전처리 제안 도출
print("\n=== Preprocessing Suggestions ===")
suggestions = []

# 결측치 기반 제안
if df['player_id'].isnull().any():
    null_count = df['player_id'].isnull().sum()
    suggestions.append(f"- player_id 결측치({null_count}건) 처리 필요: 비플레이어 이벤트(팀 단위 혹은 오류) 여부 확인 필요.")

if df['result_name'].isnull().any():
    null_count = df['result_name'].isnull().sum()
    suggestions.append(f"- result_name 결측치({null_count}건) 처리 필요: 'Pass Received' 등 특정 타입에서 발생하는지 확인하여 기본값 할당 필요.")

# 좌표 기반 제안
if (df[coord_cols] < 0).any().any() or (df[coord_cols] > 105).any().any():
    suggestions.append("- 좌표 데이터 중 범위를 벗어나는 값 탐지: 경기장 규격(0~100/105) 내로 클리핑 또는 필터링 필요.")

# 타입 매핑 제안
known_types = ['Pass', 'Pass Received', 'Ball Recovery', 'Interception', 'Challenge', 'Tackle', 'Clearance', 'Foul', 'Shot', 'Goal', 'Dribble', 'Carry']
unknown = [t for t in df['type_name'].unique() if t not in known_types]
if unknown:
    suggestions.append(f"- 미정의 이벤트 타입 확인됨: {unknown}. convert_to_spadl.py 매핑 로직에 추가 필요.")

# 시간 연속성 제안
df_sorted = df.sort_values(['game_id', 'period_id', 'time_seconds'])
time_diffs = df_sorted.groupby(['game_id', 'period_id'])['time_seconds'].diff()
if (time_diffs < 0).any():
    suggestions.append("- 시간 역전 현상 발생: 동일 경기/기수 내에서 time_seconds가 줄어드는 행 확인 및 정렬/수정 필요.")

for s in suggestions:
    print(s)

with open(f"{OUTPUT_DIR}/preprocessing_guide.md", "w", encoding='utf-8') as f:
    f.write("# 전처리 추천 가이드\n\n")
    for s in suggestions:
        f.write(f"{s}\n")

print(f"\nQuality check completed. Results saved in {OUTPUT_DIR}/")
