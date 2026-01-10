# =========================================================
# animate_tactics.py
# deTACTer 프로젝트용 전술 애니메이션 생성 모듈 (v2.0)
# =========================================================
# 실제 경기장 규격 (105m x 68m)에 맞춘 애니메이션 생성
# 모든 선수 추적 + 이벤트 타입별 애니메이션
# 출력: GIF/MP4 파일
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, Circle, RegularPolygon
from sklearn.cluster import OPTICS
import sys
import os

# 터미널 한글 출력 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 경로 설정
# =========================================================
DATA_DIR = 'c:/Users/Public/Documents/DIK/deTACTer/data/refined/'
OUTPUT_DIR = 'c:/Users/Public/Documents/DIK/deTACTer/results/animations/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 실제 경기장 규격 (미터)
FIELD_LENGTH = 105
FIELD_WIDTH = 68

# 애니메이션 설정
FRAME_INTERVAL = 300  # 밀리초 (프레임 간 간격)
TRANSITION_FRAMES = 10  # 이벤트 간 보간 프레임 수

# 시각적 요소 크기 (미터 단위)
PLAYER_RADIUS = 1.2  # 선수 원 반경
PLAYER_GLOW_RADIUS = 2.0  # 활성 선수 글로우 반경
BALL_RADIUS = 0.75  # 공 반경 (직경 1.5m)

# =========================================================
# 색상 팔레트 (이벤트 타입별)
# =========================================================
EVENT_COLORS = {
    'Pass': '#3498db',       # 파란색 - 패스
    'Cross': '#9b59b6',      # 보라색 - 크로스
    'Carry': '#f39c12',      # 주황색 - 드리블/운반
    'Shot': '#e74c3c',       # 빨간색 - 슛
    'Goal': '#27ae60',       # 초록색 - 골
    'Take-On': '#e67e22',    # 주황 - 돌파 시도
    'dribble': '#f39c12',    # SPADL 드리블
    'pass': '#3498db',       # SPADL 패스
    'shot': '#e74c3c',       # SPADL 슛
    'cross': '#9b59b6',      # SPADL 크로스
    'receive': '#1abc9c',    # 청록색 - 수신
    'default': '#7f8c8d'     # 기본 색상
}

TEAM_COLORS = ['#3498db', '#e74c3c']  # 팀 색상 (파란, 빨강)

# =========================================================
# 축구장 그리기 (실제 규격 + 잔디 패턴)
# =========================================================
def draw_pitch_real_scale(ax, grass_stripes=True):
    """
    실제 경기장 규격(105x68m)에 맞춘 축구장을 그립니다.
    
    Args:
        ax: matplotlib axes 객체
        grass_stripes: 잔디 줄무늬 패턴 적용 여부
    """
    
    # 배경색 (진한 녹색)
    ax.set_facecolor('#1a6b1a')
    
    # 잔디 줄무늬 패턴 (실제 축구장처럼)
    if grass_stripes:
        stripe_width = FIELD_LENGTH / 12  # 12개의 줄무늬
        for i in range(12):
            if i % 2 == 0:
                color = '#228B22'  # 밝은 녹색
            else:
                color = '#1e7b1e'  # 어두운 녹색
            rect = patches.Rectangle(
                (i * stripe_width, 0), stripe_width, FIELD_WIDTH,
                facecolor=color, edgecolor='none', zorder=0
            )
            ax.add_patch(rect)
    else:
        ax.set_facecolor('#228B22')
    
    # 외곽선 (두꺼운 흰색 라인)
    line_color = 'white'
    line_width = 2.5
    
    # 필드 경계선
    ax.plot([0, FIELD_LENGTH], [0, 0], color=line_color, linewidth=line_width, zorder=1)
    ax.plot([0, FIELD_LENGTH], [FIELD_WIDTH, FIELD_WIDTH], color=line_color, linewidth=line_width, zorder=1)
    ax.plot([0, 0], [0, FIELD_WIDTH], color=line_color, linewidth=line_width, zorder=1)
    ax.plot([FIELD_LENGTH, FIELD_LENGTH], [0, FIELD_WIDTH], color=line_color, linewidth=line_width, zorder=1)
    
    # 중앙선
    ax.plot([FIELD_LENGTH/2, FIELD_LENGTH/2], [0, FIELD_WIDTH], color=line_color, linewidth=2, zorder=1)
    
    # 센터 서클 (반경 9.15m)
    center_circle = plt.Circle((FIELD_LENGTH/2, FIELD_WIDTH/2), 9.15, 
                                fill=False, color=line_color, linewidth=2, zorder=1)
    ax.add_patch(center_circle)
    
    # 센터 스팟
    center_spot = plt.Circle((FIELD_LENGTH/2, FIELD_WIDTH/2), 0.3, 
                              fill=True, color=line_color, zorder=2)
    ax.add_patch(center_spot)
    
    # ===== 왼쪽 골대 영역 (수비) =====
    # 패널티 박스 (16.5m x 40.32m)
    penalty_box_width = 16.5
    penalty_box_height = 40.32
    ax.add_patch(patches.Rectangle(
        (0, (FIELD_WIDTH - penalty_box_height) / 2), 
        penalty_box_width, penalty_box_height,
        fill=False, color=line_color, linewidth=2, zorder=1
    ))
    
    # 골 에어리어 (5.5m x 18.32m)
    goal_area_width = 5.5
    goal_area_height = 18.32
    ax.add_patch(patches.Rectangle(
        (0, (FIELD_WIDTH - goal_area_height) / 2), 
        goal_area_width, goal_area_height,
        fill=False, color=line_color, linewidth=2, zorder=1
    ))
    
    # 페널티 스팟
    ax.add_patch(plt.Circle((11, FIELD_WIDTH/2), 0.25, fill=True, color=line_color, zorder=2))
    
    # 페널티 아크
    arc_left = patches.Arc((11, FIELD_WIDTH/2), 18.3, 18.3, 
                            angle=0, theta1=308, theta2=52, 
                            color=line_color, linewidth=2, zorder=1)
    ax.add_patch(arc_left)
    
    # 골대 (7.32m 너비)
    goal_width = 7.32
    goal_depth = 2.44
    goal_y_start = (FIELD_WIDTH - goal_width) / 2
    goal_y_end = (FIELD_WIDTH + goal_width) / 2
    
    # 골대 프레임 (네트 영역)
    ax.add_patch(patches.Rectangle(
        (-goal_depth, goal_y_start), goal_depth, goal_width,
        facecolor='#555555', edgecolor='white', linewidth=3, alpha=0.3, zorder=1
    ))
    ax.plot([0, 0], [goal_y_start, goal_y_end], color='white', linewidth=5, zorder=2)
    
    # ===== 오른쪽 골대 영역 (공격) =====
    ax.add_patch(patches.Rectangle(
        (FIELD_LENGTH - penalty_box_width, (FIELD_WIDTH - penalty_box_height) / 2), 
        penalty_box_width, penalty_box_height,
        fill=False, color=line_color, linewidth=2, zorder=1
    ))
    
    ax.add_patch(patches.Rectangle(
        (FIELD_LENGTH - goal_area_width, (FIELD_WIDTH - goal_area_height) / 2), 
        goal_area_width, goal_area_height,
        fill=False, color=line_color, linewidth=2, zorder=1
    ))
    
    ax.add_patch(plt.Circle((FIELD_LENGTH - 11, FIELD_WIDTH/2), 0.25, 
                             fill=True, color=line_color, zorder=2))
    
    arc_right = patches.Arc((FIELD_LENGTH - 11, FIELD_WIDTH/2), 18.3, 18.3, 
                             angle=0, theta1=128, theta2=232, 
                             color=line_color, linewidth=2, zorder=1)
    ax.add_patch(arc_right)
    
    ax.add_patch(patches.Rectangle(
        (FIELD_LENGTH, goal_y_start), goal_depth, goal_width,
        facecolor='#555555', edgecolor='white', linewidth=3, alpha=0.3, zorder=1
    ))
    ax.plot([FIELD_LENGTH, FIELD_LENGTH], [goal_y_start, goal_y_end], 
            color='white', linewidth=5, zorder=2)
    
    # 코너 아크 (반경 1m)
    corner_radius = 1
    ax.add_patch(patches.Arc((0, 0), corner_radius*2, corner_radius*2, 
                              angle=0, theta1=0, theta2=90, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(patches.Arc((0, FIELD_WIDTH), corner_radius*2, corner_radius*2, 
                              angle=0, theta1=270, theta2=360, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(patches.Arc((FIELD_LENGTH, 0), corner_radius*2, corner_radius*2, 
                              angle=0, theta1=90, theta2=180, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(patches.Arc((FIELD_LENGTH, FIELD_WIDTH), corner_radius*2, corner_radius*2, 
                              angle=0, theta1=180, theta2=270, color=line_color, linewidth=2, zorder=1))
    
    # 축 설정
    ax.set_xlim(-5, FIELD_LENGTH + 5)
    ax.set_ylim(-5, FIELD_WIDTH + 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return ax


# =========================================================
# 이벤트 타입 식별 헬퍼 함수
# =========================================================
def get_event_category(type_name, spadl_type=None):
    """
    이벤트 타입을 카테고리로 분류합니다.
    
    Returns:
        'carry': 선수가 공과 함께 이동 (드리블/운반)
        'pass': 공만 이동 (패스/크로스/슛)
        'receive': 공 받기 (수신)
        'other': 기타 이벤트
    """
    # Carry 계열 (선수 + 공 함께 이동)
    carry_types = ['Carry', 'Take-On', 'dribble', 'take_on']
    
    # Pass 계열 (공만 이동)
    pass_types = ['Pass', 'Cross', 'Shot', 'Goal', 'Pass_Corner', 'Pass_Freekick', 
                  'Shot_Freekick', 'Throw-In', 'Goal Kick', 
                  'pass', 'cross', 'shot', 'shot_freekick', 'freekick_short', 
                  'throw_in', 'goalkick', 'corner_short']
    
    # Receive 계열 (공 받기)
    receive_types = ['Pass Received', 'Ball Received']
    
    if type_name in carry_types or spadl_type in carry_types:
        return 'carry'
    elif type_name in pass_types or spadl_type in pass_types:
        return 'pass'
    elif type_name in receive_types:
        return 'receive'
    else:
        return 'other'


def get_event_color(type_name, spadl_type=None):
    """이벤트 타입에 맞는 색상을 반환합니다."""
    if type_name in EVENT_COLORS:
        return EVENT_COLORS[type_name]
    elif spadl_type and spadl_type in EVENT_COLORS:
        return EVENT_COLORS[spadl_type]
    
    # 카테고리별 기본 색상
    category = get_event_category(type_name, spadl_type)
    if category == 'receive':
        return EVENT_COLORS['receive']
    
    return EVENT_COLORS['default']


# =========================================================
# 선수별 이벤트 추출
# =========================================================
def extract_player_events(seq_df):
    """
    시퀀스에서 각 선수의 이벤트를 추출하여 딕셔너리로 반환합니다.
    
    Args:
        seq_df: 단일 시퀀스의 데이터프레임 (이미 정렬됨)
    
    Returns:
        player_events: {
            player_id: [
                {
                    'event_idx': 이벤트 인덱스,
                    'start_x': 시작 x,
                    'start_y': 시작 y,
                    'end_x': 종료 x,
                    'end_y': 종료 y,
                    'type_name': 이벤트 타입,
                    'category': 이벤트 카테고리,
                    'color': 색상,
                    'player_name': 선수 이름
                },
                ...
            ],
            ...
        }
    """
    player_events = {}
    
    for idx, row in seq_df.iterrows():
        player_id = row['player_id']
        
        if pd.isna(player_id):
            continue
        
        if player_id not in player_events:
            player_events[player_id] = []
        
        # 이벤트 정보 저장
        event_info = {
            'event_idx': idx,
            'start_x': row['start_x'] * FIELD_LENGTH,
            'start_y': row['start_y'] * FIELD_WIDTH,
            'end_x': row['end_x'] * FIELD_LENGTH if pd.notna(row['end_x']) else row['start_x'] * FIELD_LENGTH,
            'end_y': row['end_y'] * FIELD_WIDTH if pd.notna(row['end_y']) else row['start_y'] * FIELD_WIDTH,
            'type_name': row['type_name'],
            'category': get_event_category(row['type_name'], row.get('spadl_type', '')),
            'color': get_event_color(row['type_name'], row.get('spadl_type', '')),
            'player_name': row.get('player_name_ko', '')[:6] if pd.notna(row.get('player_name_ko', '')) else ''
        }
        
        player_events[player_id].append(event_info)
    
    return player_events


# =========================================================
# 선수 위치 보간 함수
# =========================================================
def get_player_position(player_id, player_events, current_event_idx, progress):
    """
    현재 프레임에서 선수의 위치를 계산합니다.
    
    Args:
        player_id: 선수 ID
        player_events: 선수별 이벤트 딕셔너리
        current_event_idx: 현재 이벤트 인덱스
        progress: 현재 이벤트 내 진행률 (0~1)
    
    Returns:
        (x, y, is_active): 위치 좌표 및 활성 여부
    """
    if player_id not in player_events or len(player_events[player_id]) == 0:
        # 이벤트가 없는 선수는 초기 위치 반환
        return FIELD_LENGTH / 2, FIELD_WIDTH / 2, False
    
    events = player_events[player_id]
    event_indices = [e['event_idx'] for e in events]
    
    # 현재 이벤트가 이 선수의 이벤트인지 확인
    if current_event_idx in event_indices:
        # 이 선수가 현재 이벤트 수행 중
        event = next(e for e in events if e['event_idx'] == current_event_idx)
        
        # Carry: start -> end 이동
        # Pass: start 위치 고정 (공만 이동)
        # Receive: end 위치로 이동
        if event['category'] == 'carry':
            x = event['start_x'] + (event['end_x'] - event['start_x']) * progress
            y = event['start_y'] + (event['end_y'] - event['start_y']) * progress
        elif event['category'] == 'pass':
            # 패스할 때는 제자리
            x, y = event['start_x'], event['start_y']
        elif event['category'] == 'receive':
            # 공 받을 때는 end 위치로 약간 이동
            x = event['start_x'] + (event['end_x'] - event['start_x']) * progress
            y = event['start_y'] + (event['end_y'] - event['start_y']) * progress
        else:
            x, y = event['start_x'], event['start_y']
        
        return x, y, True  # is_active=True
    
    # 이 선수의 다음 이벤트를 찾기
    future_events = [e for e in events if e['event_idx'] > current_event_idx]
    past_events = [e for e in events if e['event_idx'] < current_event_idx]
    
    if future_events:
        # 다음 이벤트를 위해 이동 중
        next_event = future_events[0]
        
        if past_events:
            # 이전 이벤트의 끝 위치에서 다음 이벤트의 시작 위치로 보간
            prev_event = past_events[-1]
            
            # 이벤트 간 프레임 수 계산
            frames_between = (next_event['event_idx'] - prev_event['event_idx']) * TRANSITION_FRAMES
            current_frame = (current_event_idx - prev_event['event_idx']) * TRANSITION_FRAMES + int(progress * TRANSITION_FRAMES)
            
            inter_progress = min(current_frame / frames_between, 1.0) if frames_between > 0 else 0
            
            x = prev_event['end_x'] + (next_event['start_x'] - prev_event['end_x']) * inter_progress
            y = prev_event['end_y'] + (next_event['start_y'] - prev_event['end_y']) * inter_progress
        else:
            # 첫 이벤트 이전: 다음 이벤트의 시작 위치로 이동
            x, y = next_event['start_x'], next_event['start_y']
        
        return x, y, False
    
    elif past_events:
        # 마지막 이벤트 이후: 마지막 위치에서 대기
        last_event = past_events[-1]
        x, y = last_event['end_x'], last_event['end_y']
        return x, y, False
    
    else:
        # 이벤트가 없는 경우 (should not happen)
        return FIELD_LENGTH / 2, FIELD_WIDTH / 2, False


# =========================================================
# 공 위치 계산 함수
# =========================================================
def get_ball_position(current_event, progress, player_events):
    """
    현재 프레임에서 공의 위치를 계산합니다.
    
    Args:
        current_event: 현재 이벤트 정보
        progress: 현재 이벤트 내 진행률 (0~1)
        player_events: 선수별 이벤트 딕셔너리
    
    Returns:
        (ball_x, ball_y): 공 위치 좌표
    """
    category = current_event['category']
    
    if category == 'carry':
        # Carry: 선수와 함께 이동 (선수 위치 + offset)
        player_id = current_event.get('player_id')
        if player_id and player_id in player_events:
            player_x = current_event['start_x'] + (current_event['end_x'] - current_event['start_x']) * progress
            player_y = current_event['start_y'] + (current_event['end_y'] - current_event['start_y']) * progress
            # 공은 선수 약간 앞에 위치
            ball_x = player_x + 1.5
            ball_y = player_y
        else:
            ball_x = current_event['start_x'] + (current_event['end_x'] - current_event['start_x']) * progress
            ball_y = current_event['start_y'] + (current_event['end_y'] - current_event['start_y']) * progress
    
    elif category == 'pass':
        # Pass: 공만 start -> end 이동
        ball_x = current_event['start_x'] + (current_event['end_x'] - current_event['start_x']) * progress
        ball_y = current_event['start_y'] + (current_event['end_y'] - current_event['start_y']) * progress
    
    elif category == 'receive':
        # Receive: end 위치에 공 고정
        ball_x = current_event['end_x']
        ball_y = current_event['end_y']
    
    else:
        # 기타: 기본 이동
        ball_x = current_event['start_x'] + (current_event['end_x'] - current_event['start_x']) * progress
        ball_y = current_event['start_y'] + (current_event['end_y'] - current_event['start_y']) * progress
    
    return ball_x, ball_y


# =========================================================
# 시퀀스 애니메이션 생성 (선수 추적 버전)
# =========================================================
def create_event_based_animation(seq_df, sequence_id, output_path, title="전술 패턴"):
    """
    모든 선수를 추적하는 시퀀스 애니메이션을 생성합니다.
    
    Args:
        seq_df: 전체 시퀀스 데이터프레임
        sequence_id: 애니메이션할 시퀀스 ID
        output_path: 출력 파일 경로
        title: 애니메이션 제목
    """
    # 시퀀스 추출 및 정렬 (역순에서 정순으로)
    seq = seq_df[seq_df['sequence_id'] == sequence_id].sort_values('seq_position', ascending=False).reset_index(drop=True)
    
    if len(seq) == 0:
        print(f"    [경고] 시퀀스 {sequence_id}를 찾을 수 없습니다.")
        return None
    
    # 선수별 이벤트 추출
    player_events = extract_player_events(seq)
    
    if len(player_events) == 0:
        print(f"    [경고] 시퀀스 {sequence_id}에 선수 정보가 없습니다.")
        return None
    
    # 팀 ID 추출 (첫 이벤트의 팀)
    team_id = seq.iloc[0]['team_id'] if 'team_id' in seq.columns else None
    
    # 전체 이벤트 리스트 생성
    events = []
    for idx, row in seq.iterrows():
        event = {
            'event_idx': idx,
            'player_id': row['player_id'],
            'start_x': row['start_x'] * FIELD_LENGTH,
            'start_y': row['start_y'] * FIELD_WIDTH,
            'end_x': row['end_x'] * FIELD_LENGTH if pd.notna(row['end_x']) else row['start_x'] * FIELD_LENGTH,
            'end_y': row['end_y'] * FIELD_WIDTH if pd.notna(row['end_y']) else row['start_y'] * FIELD_WIDTH,
            'type_name': row['type_name'],
            'category': get_event_category(row['type_name'], row.get('spadl_type', '')),
            'color': get_event_color(row['type_name'], row.get('spadl_type', '')),
            'player_name': row.get('player_name_ko', '')[:6] if pd.notna(row.get('player_name_ko', '')) else ''
        }
        events.append(event)
    
    # Figure 설정
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')
    draw_pitch_real_scale(ax)
    
    # 제목
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', 
                 pad=15, fontfamily='Malgun Gothic')
    
    # 시퀀스 정보
    info_text = f"시퀀스 ID: {sequence_id} | 이벤트 수: {len(events)} | 선수 수: {len(player_events)}"
    ax.text(FIELD_LENGTH/2, -3, info_text, ha='center', fontsize=10, 
            color='#cccccc', fontfamily='Malgun Gothic')
    
    # ===== 애니메이션 요소 초기화 =====
    
    # 공 궤적 라인
    ball_path_line, = ax.plot([], [], '-', color='yellow', linewidth=1.5, alpha=0.5, zorder=3)
    ball_path_x, ball_path_y = [], []
    
    # 선수 마커 딕셔너리 (player_id -> {circle, glow, label})
    player_markers = {}
    
    for i, (player_id, events_list) in enumerate(player_events.items()):
        # 초기 위치 (첫 이벤트의 start 위치)
        init_x = events_list[0]['start_x']
        init_y = events_list[0]['start_y']
        
        # 팀 색상 (간단히 인덱스로 구분)
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        
        # 선수 글로우 (활성화 표시)
        glow = plt.Circle((init_x, init_y), PLAYER_GLOW_RADIUS, 
                          color='yellow', alpha=0, zorder=4)
        ax.add_patch(glow)
        
        # 선수 원
        circle = plt.Circle((init_x, init_y), PLAYER_RADIUS, 
                            facecolor=color, edgecolor='white', linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        
        # 선수 이름 레이블
        player_name = events_list[0]['player_name']
        label = ax.text(init_x, init_y - PLAYER_RADIUS - 0.8, player_name, 
                       ha='center', fontsize=7, color='white', zorder=6,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='#333', alpha=0.7, edgecolor='none'))
        
        player_markers[player_id] = {
            'circle': circle,
            'glow': glow,
            'label': label,
            'color': color
        }
    
    # 공 마커
    ball_circle = plt.Circle((events[0]['start_x'], events[0]['start_y']), BALL_RADIUS, 
                             facecolor='white', edgecolor='black', linewidth=2, zorder=7)
    ax.add_patch(ball_circle)
    
    # 공 패턴 (오각형)
    ball_pattern = RegularPolygon((events[0]['start_x'], events[0]['start_y']), 
                                   numVertices=5, radius=BALL_RADIUS * 0.6, 
                                   facecolor='black', edgecolor='none', zorder=8)
    ax.add_patch(ball_pattern)
    
    # 이벤트 레이블 (현재 이벤트 정보)
    event_label = ax.text(FIELD_LENGTH/2, FIELD_WIDTH + 3, '', 
                          ha='center', fontsize=11, color='white', fontweight='bold', zorder=10,
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#333', alpha=0.9))
    
    # 시작점 마커
    ax.scatter([events[0]['start_x']], [events[0]['start_y']], 
               color='lime', s=200, zorder=3, edgecolor='white', linewidth=2, marker='o', label='시작점')
    
    # 총 프레임 수
    total_frames = len(events) * TRANSITION_FRAMES + 10
    
    # 플로팅된 이벤트 포인트
    event_markers = []
    
    def init():
        """애니메이션 초기화"""
        ball_path_line.set_data([], [])
        return [ball_path_line, ball_circle, ball_pattern, event_label] + \
               [m['circle'] for m in player_markers.values()] + \
               [m['glow'] for m in player_markers.values()] + \
               [m['label'] for m in player_markers.values()]
    
    def animate(frame):
        """프레임별 애니메이션 업데이트"""
        nonlocal ball_path_x, ball_path_y
        
        # 현재 이벤트 인덱스 및 진행률
        event_idx = min(frame // TRANSITION_FRAMES, len(events) - 1)
        progress = (frame % TRANSITION_FRAMES) / TRANSITION_FRAMES
        
        if event_idx >= len(events):
            return [ball_path_line, ball_circle, ball_pattern, event_label] + \
                   [m['circle'] for m in player_markers.values()] + \
                   [m['glow'] for m in player_markers.values()] + \
                   [m['label'] for m in player_markers.values()]
        
        current_event = events[event_idx]
        
        # 1. 모든 선수 위치 업데이트
        for player_id, markers in player_markers.items():
            x, y, is_active = get_player_position(player_id, player_events, event_idx, progress)
            
            # 선수 원 위치
            markers['circle'].center = (x, y)
            
            # 활성 선수 글로우
            if is_active:
                markers['glow'].set_alpha(0.5)
                markers['glow'].center = (x, y)
                markers['circle'].set_linewidth(3)
                markers['circle'].set_edgecolor('yellow')
            else:
                markers['glow'].set_alpha(0)
                markers['circle'].set_linewidth(1.5)
                markers['circle'].set_edgecolor('white')
            
            # 선수 이름 레이블 위치
            markers['label'].set_position((x, y - PLAYER_RADIUS - 0.8))
        
        # 2. 공 위치 업데이트
        ball_x, ball_y = get_ball_position(current_event, progress, player_events)
        ball_circle.center = (ball_x, ball_y)
        ball_pattern.xy = (ball_x, ball_y)
        
        # 3. 공 궤적 업데이트
        if len(ball_path_x) == 0 or (ball_x != ball_path_x[-1] or ball_y != ball_path_y[-1]):
            ball_path_x.append(ball_x)
            ball_path_y.append(ball_y)
            ball_path_line.set_data(ball_path_x, ball_path_y)
        
        # 4. 이벤트 레이블 업데이트
        event_type = current_event['type_name']
        player_name = current_event['player_name']
        label_text = f"이벤트 {event_idx + 1}/{len(events)}: {player_name} - {event_type}"
        event_label.set_text(label_text)
        event_label.set_color(current_event['color'])
        
        # 5. 이벤트 완료 시 마커 추가
        if progress >= 0.9 and len(event_markers) <= event_idx:
            marker = ax.scatter([current_event['end_x']], [current_event['end_y']], 
                               color=current_event['color'], s=60, zorder=4, 
                               edgecolor='white', linewidth=1, alpha=0.7)
            event_markers.append(marker)
        
        return [ball_path_line, ball_circle, ball_pattern, event_label] + \
               [m['circle'] for m in player_markers.values()] + \
               [m['glow'] for m in player_markers.values()] + \
               [m['label'] for m in player_markers.values()]
    
    # 애니메이션 생성
    anim = FuncAnimation(fig, animate, init_func=init, frames=total_frames, 
                         interval=FRAME_INTERVAL // 2, blit=False, repeat=True)
    
    # 저장
    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=10, metadata={'title': title})
        anim.save(output_path, writer=writer, dpi=100)
    except Exception as e:
        # ffmpeg 없으면 GIF로 대체
        print(f"    [알림] FFMpeg 사용 불가, GIF로 저장합니다: {e}")
        output_path = output_path.replace('.mp4', '.gif')
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=5)
        anim.save(output_path, writer=writer, dpi=100)
    
    plt.close(fig)
    
    return output_path


# =========================================================
# 정적 시퀀스 플롯 (이벤트 표시)
# =========================================================
def plot_sequence_static(seq_df, sequence_id, output_path, title="전술 패턴"):
    """
    시퀀스를 정적 이미지로 플롯합니다.
    각 이벤트를 점으로 표시하고 화살표로 연결합니다.
    """
    seq = seq_df[seq_df['sequence_id'] == sequence_id].sort_values('seq_position', ascending=False).reset_index(drop=True)
    
    if len(seq) == 0:
        print(f"    [경고] 시퀀스 {sequence_id}를 찾을 수 없습니다.")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')
    draw_pitch_real_scale(ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', 
                 pad=15, fontfamily='Malgun Gothic')
    
    # 이벤트별 플롯
    prev_x, prev_y = None, None
    
    for idx, row in seq.iterrows():
        x = row['start_x'] * FIELD_LENGTH
        y = row['start_y'] * FIELD_WIDTH
        end_x = row['end_x'] * FIELD_LENGTH if pd.notna(row['end_x']) else x
        end_y = row['end_y'] * FIELD_WIDTH if pd.notna(row['end_y']) else y
        
        color = get_event_color(row['type_name'], row.get('spadl_type', ''))
        category = get_event_category(row['type_name'], row.get('spadl_type', ''))
        
        # 이전 이벤트와 연결
        if prev_x is not None:
            ax.plot([prev_x, x], [prev_y, y], '-', color='yellow', 
                   linewidth=1.5, alpha=0.6, zorder=2)
        
        # 이벤트 화살표
        if category == 'pass':
            arrow = FancyArrowPatch(
                (x, y), (end_x, end_y),
                arrowstyle='fancy,head_width=6,head_length=8',
                color=color, linewidth=2, alpha=0.8, zorder=3
            )
            ax.add_patch(arrow)
        
        # 이벤트 포인트
        if idx == 0:
            ax.scatter([x], [y], color='lime', s=250, zorder=5, 
                      edgecolor='white', linewidth=2, marker='o')
        else:
            ax.scatter([x], [y], color=color, s=120, zorder=4, 
                      edgecolor='white', linewidth=1.5)
        
        # 레이블
        label = f"{idx+1}. {row['type_name']}"
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=7, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
        
        prev_x, prev_y = end_x, end_y
    
    # 마지막 점
    if len(seq) > 0:
        last_row = seq.iloc[-1]
        end_x = last_row['end_x'] * FIELD_LENGTH if pd.notna(last_row['end_x']) else last_row['start_x'] * FIELD_LENGTH
        end_y = last_row['end_y'] * FIELD_WIDTH if pd.notna(last_row['end_y']) else last_row['start_y'] * FIELD_WIDTH
        ax.scatter([end_x], [end_y], color='red', s=200, zorder=5, 
                  edgecolor='white', linewidth=2, marker='*')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    
    return output_path


# =========================================================
# 클러스터별 대표 시퀀스 애니메이션
# =========================================================
def create_cluster_animations(seq_df, cluster_df, cluster_col, n_clusters=10, n_samples=3):
    """각 클러스터의 대표 시퀀스들을 애니메이션으로 생성합니다."""
    print(f"[애니메이션] {cluster_col} 클러스터별 애니메이션 생성 중...")
    
    cluster_sizes = cluster_df[cluster_col].value_counts()
    top_clusters = [c for c in cluster_sizes.index if c != -1][:n_clusters]
    
    for cluster_id in top_clusters:
        seq_ids = cluster_df[cluster_df[cluster_col] == cluster_id]['sequence_id'].values[:n_samples]
        
        for i, seq_id in enumerate(seq_ids):
            output_path = f"{OUTPUT_DIR}{cluster_col}_cluster{cluster_id}_sample{i+1}.mp4"
            title = f"클러스터 {cluster_id} - 샘플 {i+1}"
            
            result = create_event_based_animation(seq_df, seq_id, output_path, title)
            if result:
                print(f"    -> 저장: {result}")
            
            static_path = f"{OUTPUT_DIR}{cluster_col}_cluster{cluster_id}_sample{i+1}_static.png"
            result_static = plot_sequence_static(seq_df, seq_id, static_path, title)
            if result_static:
                print(f"    -> 정적 플롯 저장: {result_static}")
    
    print(f"    -> 완료!")


# =========================================================
# 메인
# =========================================================
def main():
    print("=" * 60)
    print("deTACTer 전술 애니메이션 생성 (v2.0 - 선수 추적)")
    print("=" * 60)
    
    # 데이터 로드
    print("[1/4] 데이터 로드 중...")
    seq_df = pd.read_csv(DATA_DIR + 'attack_sequences.csv', encoding='utf-8-sig')
    
    # 클러스터 분석
    print("[2/4] OPTICS 세부 분류 (min_samples=3, xi=0.03)...")
    dtw_matrix = np.load(DATA_DIR + 'dtw_distance_matrix.npy')
    
    optics = OPTICS(min_samples=3, xi=0.03, metric='precomputed')
    labels = optics.fit_predict(dtw_matrix)
    
    cluster_df = pd.read_csv(DATA_DIR + 'cluster_labels.csv', encoding='utf-8-sig')
    cluster_df['optics_detailed'] = labels
    cluster_df.to_csv(DATA_DIR + 'cluster_labels.csv', index=False, encoding='utf-8-sig')
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"    -> {n_clusters}개 클러스터, {n_noise}개 노이즈")
    
    # 클러스터별 애니메이션
    print("[3/4] 클러스터별 애니메이션 생성 중...")
    create_cluster_animations(seq_df, cluster_df, 'optics_detailed', n_clusters=5, n_samples=2)
    
    # 데모 애니메이션
    print("[4/4] 데모 애니메이션 생성...")
    sample_seq = cluster_df[cluster_df['optics_detailed'] != -1]['sequence_id'].iloc[0]
    demo_path = create_event_based_animation(seq_df, sample_seq, OUTPUT_DIR + 'demo_sequence.mp4', "전술 패턴 데모")
    print(f"    -> 데모 저장: {demo_path}")
    
    static_demo_path = plot_sequence_static(seq_df, sample_seq, OUTPUT_DIR + 'demo_sequence_static.png', "전술 패턴 데모 (정적)")
    print(f"    -> 데모 정적 플롯 저장: {static_demo_path}")
    
    print("=" * 60)
    print(f"애니메이션 저장 완료: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
