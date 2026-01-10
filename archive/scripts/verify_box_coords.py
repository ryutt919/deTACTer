import pandas as pd
import numpy as np
import sys

# 터미널 출력 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PATH = 'c:/Users/Public/Documents/DIK/deTACTer/data/raw_data.csv'

try:
    df = pd.read_csv(PATH, encoding='utf-8-sig')
    
    # 슈팅 이벤트 필터링
    shots = df[df['type_name'].str.contains('shot|Goal', case=False, na=False)]
    
    print(f"총 슈팅/골 이벤트 수: {len(shots)}")
    print("\n--- 슈팅 시작 위치(start_x, start_y) 통계 ---")
    print(shots[['start_x', 'start_y']].describe())
    
    # 0-100 스케일이라고 가정 시 정규화된 통계
    # 실제 데이터의 max가 100인지 105인지 확인
    max_x = df['start_x'].max()
    max_y = df['start_y'].max()
    print(f"\n전체 데이터 Max X: {max_x}, Max Y: {max_y}")

    # 슈팅의 90% 이상이 발생하는 X 구간 확인
    print("\n--- 슈팅 X좌표 분위수 ---")
    print(shots['start_x'].quantile([0.1, 0.2, 0.5, 0.8, 0.9]))

except Exception as e:
    print(f"오류 발생: {e}")
