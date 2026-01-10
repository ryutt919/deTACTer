# 전처리 제안 보고서

- player_id 결측치(1건): 팀 차원의 이벤트일 수 있으나, 분석 목적에 따라 제거 또는 placeholder(0 등) 처리 필요.
- 인코딩: 'utf-8-sig'를 사용하여 한글 깨짐 방지 확인됨.
- 좌표 보정: 0-100 범위를 벗어나는 좌표값에 대한 클리핑(clipping) 또는 이상치 제거 로직 추가 필요.
- SPADL 변환: 현재 'Pass Received' 등 SPADL에서 활용하지 않는 타입이 다수 포함되어 있음. convert_to_spadl.py의 map_action_type을 보강하여 더 많은 액션을 포괄하거나, 불필요한 액션은 사전에 필터링 권장.
