# 전처리 추천 가이드

- player_id 결측치(1건) 처리 필요: 비플레이어 이벤트(팀 단위 혹은 오류) 여부 확인 필요.
- result_name 결측치(349604건) 처리 필요: 'Pass Received' 등 특정 타입에서 발생하는지 확인하여 기본값 할당 필요.
- 미정의 이벤트 타입 확인됨: ['Block', 'Out', 'Throw-In', 'Intervention', 'Recovery', 'Duel', 'Offside', 'Pass_Freekick', 'Error', 'Cross', 'Goal Kick', 'Aerial Clearance', 'Catch', 'Take-On', 'Pause', 'Ball Received', 'Parry', 'Pass_Corner', 'Hit', 'Defensive Line Support', 'Goal Miss', 'Deflection', 'Penalty Kick', 'Goal Post', 'Handball_Foul', 'Shot_Freekick', 'Own Goal', 'Foul_Throw']. convert_to_spadl.py 매핑 로직에 추가 필요.
