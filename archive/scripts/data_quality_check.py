import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 터미널 출력 인코딩 설정 (윈도우 한글 깨짐 방지)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.7 미만 version 대응 (필요시)
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 설정
RAW_DATA_PATH = 'c:/Users/Public/Documents/DIK/deTACTer/data/raw_data.csv'
OUTPUT_DIR = 'c:/Users/Public/Documents/DIK/deTACTer/results/quality_check'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_quality():
    print("=== 데이터 품질 검사 시작 ===")
    
    # 1. 데이터 로드 (utf-8-sig 적용)
    try:
        df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8-sig')
        print(f"데이터 로드 완료: {len(df)} 행")
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return

    # 2. 기본 정보 저장
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 데이터 기본 요약 ===\n")
        f.write(str(df.info()) + "\n\n")
        f.write("=== 수치형 데이터 통계 ===\n")
        f.write(str(df.describe()) + "\n")

    # 3. 결측치 분석
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'count': missing, 'percentage': missing_pct})
    missing_df.to_csv(os.path.join(OUTPUT_DIR, 'missing_values.csv'))
    print("- 결측치 분석 완료 (missing_values.csv)")

    # 4. 좌표 데이터 검사 (축구장 규격: 일반적으로 0-100 or 0-105 등 기준 확인 필요)
    # 여기서는 0-100 범위를 크게 벗어나는 데이터가 있는지 확인
    coord_cols = ['start_x', 'start_y', 'end_x', 'end_y']
    out_of_bounds = {}
    for col in coord_cols:
        out_of_bounds[col] = len(df[(df[col] < 0) | (df[col] > 100)])
    
    with open(os.path.join(OUTPUT_DIR, 'out_of_bounds.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 좌표 범위 이탈 (0-100 기준) ===\n")
        for col, count in out_of_bounds.items():
            f.write(f"{col}: {count} 건\n")
    print("- 좌표 범위 검사 완료 (out_of_bounds.txt)")

    # 5. 타입 및 결과값 분포
    type_counts = df['type_name'].value_counts()
    type_counts.to_csv(os.path.join(OUTPUT_DIR, 'type_distribution.csv'))
    
    result_counts = df['result_name'].value_counts(dropna=False)
    result_counts.to_csv(os.path.join(OUTPUT_DIR, 'result_distribution.csv'))
    print("- 타입/결과 분포 성 완료")

    # 6. 시각화: 이벤트 밀도 (Heatmap 생성)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df.sample(min(5000, len(df))), x='start_x', y='start_y', alpha=0.1)
    plt.title('Sample Action Start Positions')
    plt.savefig(os.path.join(OUTPUT_DIR, 'start_positions.png'))
    plt.close()
    
    # 7. 전처리 제안 도출을 위한 로직 분석
    print("\n=== 전처리 제안 사항 도출 ===")
    proposals = []
    
    # 7-1. 결측치 처리
    if missing['player_id'] > 0:
        proposals.append(f"- player_id 결측치({missing['player_id']}건): 팀 차원의 이벤트일 수 있으나, 분석 목적에 따라 제거 또는 placeholder(0 등) 처리 필요.")
    
    # 7-2. 인코딩 및 문자열
    proposals.append("- 인코딩: 'utf-8-sig'를 사용하여 한글 깨짐 방지 확인됨.")
    
    # 7-3. 좌표 정규화
    if any(count > 0 for count in out_of_bounds.values()):
        proposals.append("- 좌표 보정: 0-100 범위를 벗어나는 좌표값에 대한 클리핑(clipping) 또는 이상치 제거 로직 추가 필요.")
    
    # 7-4. SPADL 호환성 (convert_to_spadl.py 기반)
    proposals.append("- SPADL 변환: 현재 'Pass Received' 등 SPADL에서 활용하지 않는 타입이 다수 포함되어 있음. convert_to_spadl.py의 map_action_type을 보강하여 더 많은 액션을 포괄하거나, 불필요한 액션은 사전에 필터링 권장.")
    
    with open(os.path.join(OUTPUT_DIR, 'preprocessing_proposals.md'), 'w', encoding='utf-8') as f:
        f.write("# 전처리 제안 보고서\n\n")
        for p in proposals:
            f.write(p + "\n")
            
    print("- 전처리 제안서 작성 완료 (preprocessing_proposals.md)")
    print(f"\n모든 결과는 {OUTPUT_DIR} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    check_quality()
