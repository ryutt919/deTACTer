# =========================================================
# animate_tactics.py
# deTACTer 프로젝트용 전술 애니메이션 생성 모듈
# =========================================================
# 실제 경기장 규격 (105m x 68m)에 맞춘 애니메이션 생성
# 이벤트 타입별 애니메이션: Carry(선수+공 이동), Pass(공만 이동)
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
TRANSITION_FRAMES = 8  # 이벤트 간 보간 프레임 수

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
    'default': '#7f8c8d'     # 기본 색상
}

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
    
    # 골대 (7.32m 너비) - 두꺼운 라인으로 표현
    goal_width = 7.32
    goal_depth = 2.44  # 골대 깊이 표현
    goal_y_start = (FIELD_WIDTH - goal_width) / 2
    goal_y_end = (FIELD_WIDTH + goal_width) / 2
    
    # 골대 프레임 (네트 영역)
    ax.add_patch(patches.Rectangle(
        (-goal_depth, goal_y_start), goal_depth, goal_width,
        facecolor='#555555', edgecolor='white', linewidth=3, alpha=0.3, zorder=1
    ))
    ax.plot([0, 0], [goal_y_start, goal_y_end], color='white', linewidth=5, zorder=2)
    
    # ===== 오른쪽 골대 영역 (공격) =====
    # 패널티 박스
    ax.add_patch(patches.Rectangle(
        (FIELD_LENGTH - penalty_box_width, (FIELD_WIDTH - penalty_box_height) / 2), 
        penalty_box_width, penalty_box_height,
        fill=False, color=line_color, linewidth=2, zorder=1
    ))
    
    # 골 에어리어
    ax.add_patch(patches.Rectangle(
        (FIELD_LENGTH - goal_area_width, (FIELD_WIDTH - goal_area_height) / 2), 
        goal_area_width, goal_area_height,
        fill=False, color=line_color, linewidth=2, zorder=1
    ))
    
    # 페널티 스팟
    ax.add_patch(plt.Circle((FIELD_LENGTH - 11, FIELD_WIDTH/2), 0.25, 
                             fill=True, color=line_color, zorder=2))
    
    # 페널티 아크
    arc_right = patches.Arc((FIELD_LENGTH - 11, FIELD_WIDTH/2), 18.3, 18.3, 
                             angle=0, theta1=128, theta2=232, 
                             color=line_color, linewidth=2, zorder=1)
    ax.add_patch(arc_right)
    
    # 골대 (오른쪽)
    ax.add_patch(patches.Rectangle(
        (FIELD_LENGTH, goal_y_start), goal_depth, goal_width,
        facecolor='#555555', edgecolor='white', linewidth=3, alpha=0.3, zorder=1
    ))
    ax.plot([FIELD_LENGTH, FIELD_LENGTH], [goal_y_start, goal_y_end], 
            color='white', linewidth=5, zorder=2)
    
    # 코너 아크 (반경 1m)
    corner_radius = 1
    # 왼쪽 하단
    ax.add_patch(patches.Arc((0, 0), corner_radius*2, corner_radius*2, 
                              angle=0, theta1=0, theta2=90, color=line_color, linewidth=2, zorder=1))
    # 왼쪽 상단
    ax.add_patch(patches.Arc((0, FIELD_WIDTH), corner_radius*2, corner_radius*2, 
                              angle=0, theta1=270, theta2=360, color=line_color, linewidth=2, zorder=1))
    # 오른쪽 하단
    ax.add_patch(patches.Arc((FIELD_LENGTH, 0), corner_radius*2, corner_radius*2, 
                              angle=0, theta1=90, theta2=180, color=line_color, linewidth=2, zorder=1))
    # 오른쪽 상단
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
        'other': 기타 이벤트
    """
    # Carry 계열 (선수 + 공 함께 이동)
    carry_types = ['Carry', 'Take-On', 'dribble', 'take_on']
    
    # Pass 계열 (공만 이동)
    pass_types = ['Pass', 'Cross', 'Shot', 'Goal', 'Pass_Corner', 'Pass_Freekick', 
                  'Shot_Freekick', 'Throw-In', 'Goal Kick', 
                  'pass', 'cross', 'shot', 'shot_freekick', 'freekick_short', 
                  'throw_in', 'goalkick', 'corner_short']
    
    if type_name in carry_types or spadl_type in carry_types:
        return 'carry'
    elif type_name in pass_types or spadl_type in pass_types:
        return 'pass'
    else:
        return 'other'


def get_event_color(type_name, spadl_type=None):
    """이벤트 타입에 맞는 색상을 반환합니다."""
    if type_name in EVENT_COLORS:
        return EVENT_COLORS[type_name]
    elif spadl_type and spadl_type in EVENT_COLORS:
        return EVENT_COLORS[spadl_type]
    return EVENT_COLORS['default']


# =========================================================
# 선수 및 공 그리기 함수
# =========================================================
def draw_player(ax, x, y, color='#3498db', size=12, number=None, name=None, is_active=False):
    """
    선수를 원형으로 그립니다.
    
    Args:
        ax: matplotlib axes
        x, y: 위치 좌표
        color: 유니폼 색상
        size: 선수 크기
        number: 등번호 (선택)
        name: 선수 이름 (선택)
        is_active: 활성 선수 여부 (하이라이트)
    """
    # 외곽 효과 (활성 선수일 때)
    if is_active:
        glow = plt.Circle((x, y), size * 0.18, color='yellow', alpha=0.5, zorder=4)
        ax.add_patch(glow)
    
    # 선수 원
    player = plt.Circle((x, y), size * 0.12, color=color, 
                         edgecolor='white', linewidth=1.5, zorder=5)
    ax.add_patch(player)
    
    # 등번호 표시 (선택)
    if number is not None:
        ax.text(x, y, str(number), ha='center', va='center', 
                fontsize=7, fontweight='bold', color='white', zorder=6)
    
    # 선수 이름 표시 (선택)
    if name is not None:
        ax.text(x, y - size * 0.2, name, ha='center', va='top', 
                fontsize=6, color='white', zorder=6)
    
    return player


def draw_ball(ax, x, y, size=0.8):
    """
    축구공을 그립니다.
    
    Args:
        ax: matplotlib axes
        x, y: 위치 좌표
        size: 공 크기
    """
    # 공 본체 (흰색 + 검정 패턴)
    ball = plt.Circle((x, y), size, color='white', edgecolor='black', 
                       linewidth=1.5, zorder=7)
    ax.add_patch(ball)
    
    # 오각형 패턴 (간단한 표현)
    pentagon = RegularPolygon((x, y), numVertices=5, radius=size * 0.5, 
                               color='black', zorder=8)
    ax.add_patch(pentagon)
    
    return ball


# =========================================================
# 시퀀스 애니메이션 생성 (이벤트 기반)
# =========================================================
def create_event_based_animation(seq_df, sequence_id, output_path, title="전술 패턴"):
    """
    이벤트 타입에 따른 시퀀스 애니메이션을 생성합니다.
    - Carry: 선수가 공과 함께 이동
    - Pass: 공만 목표 지점으로 이동, 선수는 제자리
    
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
    
    # 정규화된 좌표를 실제 규격으로 변환
    events = []
    for idx, row in seq.iterrows():
        event = {
            'start_x': row['start_x'] * FIELD_LENGTH,
            'start_y': row['start_y'] * FIELD_WIDTH,
            'end_x': row['end_x'] * FIELD_LENGTH if pd.notna(row['end_x']) else row['start_x'] * FIELD_LENGTH,
            'end_y': row['end_y'] * FIELD_WIDTH if pd.notna(row['end_y']) else row['start_y'] * FIELD_WIDTH,
            'type_name': row['type_name'],
            'spadl_type': row.get('spadl_type', ''),
            'player_name': row.get('player_name_ko', ''),
            'category': get_event_category(row['type_name'], row.get('spadl_type', '')),
            'color': get_event_color(row['type_name'], row.get('spadl_type', ''))
        }
        events.append(event)
    
    # Figure 설정
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')
    draw_pitch_real_scale(ax)
    
    # 제목 스타일
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', 
                 pad=15, fontfamily='Malgun Gothic')
    
    # 시퀀스 정보 표시
    info_text = f"시퀀스 ID: {sequence_id} | 이벤트 수: {len(events)}"
    ax.text(FIELD_LENGTH/2, -3, info_text, ha='center', fontsize=10, 
            color='#cccccc', fontfamily='Malgun Gothic')
    
    # 애니메이션 요소 초기화
    # 경로 라인 (지나온 경로)
    path_line, = ax.plot([], [], '-', color='yellow', linewidth=2, alpha=0.7, zorder=3)
    
    # 현재 이벤트 화살표 (동적으로 생성)
    arrow_patch = None
    
    # 선수 및 공 (초기 위치)
    if events:
        init_x, init_y = events[0]['start_x'], events[0]['start_y']
    else:
        init_x, init_y = FIELD_LENGTH / 2, FIELD_WIDTH / 2
    
    # 선수 마커 (원)
    player_circle = plt.Circle((init_x, init_y), 2.5, color='#3498db', 
                                 edgecolor='white', linewidth=2, zorder=5)
    ax.add_patch(player_circle)
    
    # 선수 활성 글로우
    player_glow = plt.Circle((init_x, init_y), 4, color='yellow', alpha=0.3, zorder=4)
    ax.add_patch(player_glow)
    
    # 공 마커
    ball_circle = plt.Circle((init_x, init_y), 1.2, color='white', 
                               edgecolor='black', linewidth=1.5, zorder=7)
    ax.add_patch(ball_circle)
    
    # 공 패턴
    ball_pattern = RegularPolygon((init_x, init_y), numVertices=5, radius=0.6, 
                                   color='black', zorder=8)
    ax.add_patch(ball_pattern)
    
    # 이벤트 레이블
    event_label = ax.text(init_x, init_y + 5, '', ha='center', fontsize=9, 
                          color='white', fontweight='bold', zorder=10,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='#333', alpha=0.8))
    
    # 시작점 마커
    ax.scatter([init_x], [init_y], color='lime', s=200, zorder=3, 
               edgecolor='white', linewidth=2, marker='o', label='시작점')
    
    # 경로 데이터 저장
    path_x = [init_x]
    path_y = [init_y]
    
    # 이벤트 인덱스별 프레임 계산
    total_frames = len(events) * TRANSITION_FRAMES + 10  # 마지막 유지 프레임 추가
    
    # 플로팅된 이벤트 포인트 저장
    event_markers = []
    
    def init():
        """애니메이션 초기화"""
        path_line.set_data([], [])
        return path_line, player_circle, player_glow, ball_circle, ball_pattern, event_label
    
    def animate(frame):
        """프레임별 애니메이션 업데이트"""
        nonlocal path_x, path_y, arrow_patch
        
        # 현재 이벤트 인덱스 및 보간 진행률 계산
        event_idx = min(frame // TRANSITION_FRAMES, len(events) - 1)
        progress = (frame % TRANSITION_FRAMES) / TRANSITION_FRAMES
        
        if event_idx >= len(events):
            return path_line, player_circle, player_glow, ball_circle, ball_pattern, event_label
        
        event = events[event_idx]
        category = event['category']
        
        # 시작점과 끝점
        sx, sy = event['start_x'], event['start_y']
        ex, ey = event['end_x'], event['end_y']
        
        # 현재 보간 위치
        if category == 'carry':
            # Carry: 선수와 공이 함께 이동
            curr_x = sx + (ex - sx) * progress
            curr_y = sy + (ey - sy) * progress
            
            # 선수 위치 업데이트
            player_circle.center = (curr_x, curr_y)
            player_glow.center = (curr_x, curr_y)
            
            # 공 위치 업데이트 (선수와 함께)
            ball_circle.center = (curr_x + 1.5, curr_y)  # 약간 앞에
            ball_pattern.xy = (curr_x + 1.5, curr_y)
            
        elif category == 'pass':
            # Pass: 선수는 시작점에 고정, 공만 이동
            player_circle.center = (sx, sy)
            player_glow.center = (sx, sy)
            
            # 공만 이동
            curr_x = sx + (ex - sx) * progress
            curr_y = sy + (ey - sy) * progress
            ball_circle.center = (curr_x, curr_y)
            ball_pattern.xy = (curr_x, curr_y)
            
        else:
            # 기타: 기본 이동
            curr_x = sx + (ex - sx) * progress
            curr_y = sy + (ey - sy) * progress
            player_circle.center = (curr_x, curr_y)
            player_glow.center = (curr_x, curr_y)
            ball_circle.center = (curr_x, curr_y)
            ball_pattern.xy = (curr_x, curr_y)
        
        # 이벤트 레이블 업데이트
        event_type = event['type_name']
        player_name = event['player_name'][:6] if event['player_name'] else ''
        label_text = f"{event_type}"
        if player_name:
            label_text = f"{player_name}\n{event_type}"
        event_label.set_text(label_text)
        event_label.set_position((curr_x, curr_y + 6))
        event_label.set_color(event['color'])
        
        # 경로 업데이트 (이벤트 완료 시)
        if progress >= 0.9 and len(path_x) <= event_idx + 1:
            path_x.append(ex)
            path_y.append(ey)
            path_line.set_data(path_x, path_y)
            
            # 이벤트 포인트 마커 추가
            marker = ax.scatter([ex], [ey], color=event['color'], s=80, 
                               zorder=4, edgecolor='white', linewidth=1, alpha=0.8)
            event_markers.append(marker)
        
        # 선수 색상 업데이트 (이벤트 타입별)
        player_circle.set_facecolor(event['color'])
        
        return path_line, player_circle, player_glow, ball_circle, ball_pattern, event_label
    
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
    
    Args:
        seq_df: 전체 시퀀스 데이터프레임
        sequence_id: 플롯할 시퀀스 ID
        output_path: 출력 파일 경로
        title: 플롯 제목
    """
    # 시퀀스 추출 및 정렬
    seq = seq_df[seq_df['sequence_id'] == sequence_id].sort_values('seq_position', ascending=False).reset_index(drop=True)
    
    if len(seq) == 0:
        print(f"    [경고] 시퀀스 {sequence_id}를 찾을 수 없습니다.")
        return None
    
    # Figure 설정
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')
    draw_pitch_real_scale(ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', 
                 pad=15, fontfamily='Malgun Gothic')
    
    # 이벤트별 플롯
    prev_x, prev_y = None, None
    
    for idx, row in seq.iterrows():
        # 좌표 변환
        x = row['start_x'] * FIELD_LENGTH
        y = row['start_y'] * FIELD_WIDTH
        end_x = row['end_x'] * FIELD_LENGTH if pd.notna(row['end_x']) else x
        end_y = row['end_y'] * FIELD_WIDTH if pd.notna(row['end_y']) else y
        
        # 이벤트 색상
        color = get_event_color(row['type_name'], row.get('spadl_type', ''))
        category = get_event_category(row['type_name'], row.get('spadl_type', ''))
        
        # 이전 이벤트와 연결 (경로 라인)
        if prev_x is not None:
            ax.plot([prev_x, x], [prev_y, y], '-', color='yellow', 
                   linewidth=1.5, alpha=0.6, zorder=2)
        
        # 이벤트 화살표 (시작 -> 끝)
        if category == 'pass':
            arrow_style = 'fancy,head_width=6,head_length=8'
            arrow = FancyArrowPatch(
                (x, y), (end_x, end_y),
                arrowstyle=arrow_style,
                color=color,
                linewidth=2,
                alpha=0.8,
                zorder=3,
                connectionstyle='arc3,rad=0.1'
            )
            ax.add_patch(arrow)
        
        # 이벤트 포인트
        if idx == 0:
            # 시작점 (특별 표시)
            ax.scatter([x], [y], color='lime', s=250, zorder=5, 
                      edgecolor='white', linewidth=2, marker='o')
        else:
            ax.scatter([x], [y], color=color, s=120, zorder=4, 
                      edgecolor='white', linewidth=1.5)
        
        # 이벤트 레이블
        label = f"{idx+1}. {row['type_name']}"
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=7, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
        
        prev_x, prev_y = end_x, end_y
    
    # 마지막 끝점 표시
    if len(seq) > 0:
        last_row = seq.iloc[-1]
        end_x = last_row['end_x'] * FIELD_LENGTH if pd.notna(last_row['end_x']) else last_row['start_x'] * FIELD_LENGTH
        end_y = last_row['end_y'] * FIELD_WIDTH if pd.notna(last_row['end_y']) else last_row['start_y'] * FIELD_WIDTH
        ax.scatter([end_x], [end_y], color='red', s=200, zorder=5, 
                  edgecolor='white', linewidth=2, marker='*')
    
    # 범례
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
                   markersize=12, label='시작점', linestyle='None'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                   markersize=15, label='종료점', linestyle='None'),
        plt.Line2D([0], [0], color='yellow', linewidth=2, label='이동 경로'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, 
              facecolor='#333', edgecolor='white', labelcolor='white')
    
    # 저장
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    
    return output_path


# =========================================================
# 클러스터별 대표 시퀀스 애니메이션
# =========================================================
def create_cluster_animations(seq_df, cluster_df, cluster_col, n_clusters=10, n_samples=3):
    """
    각 클러스터의 대표 시퀀스들을 애니메이션으로 생성합니다.
    
    Args:
        seq_df: 시퀀스 데이터프레임
        cluster_df: 클러스터 레이블 데이터프레임
        cluster_col: 클러스터 컬럼명
        n_clusters: 생성할 클러스터 수
        n_samples: 클러스터당 샘플 수
    """
    print(f"[애니메이션] {cluster_col} 클러스터별 애니메이션 생성 중...")
    
    # 클러스터 크기 순으로 정렬
    cluster_sizes = cluster_df[cluster_col].value_counts()
    top_clusters = [c for c in cluster_sizes.index if c != -1][:n_clusters]
    
    for cluster_id in top_clusters:
        seq_ids = cluster_df[cluster_df[cluster_col] == cluster_id]['sequence_id'].values[:n_samples]
        
        for i, seq_id in enumerate(seq_ids):
            # 애니메이션 생성
            output_path = f"{OUTPUT_DIR}{cluster_col}_cluster{cluster_id}_sample{i+1}.mp4"
            title = f"클러스터 {cluster_id} - 샘플 {i+1}"
            
            result = create_event_based_animation(seq_df, seq_id, output_path, title)
            if result:
                print(f"    -> 저장: {result}")
            
            # 정적 플롯도 생성
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
    print("deTACTer 전술 애니메이션 생성 (이벤트 기반)")
    print("=" * 60)
    
    # 데이터 로드
    print("[1/4] 데이터 로드 중...")
    seq_df = pd.read_csv(DATA_DIR + 'attack_sequences.csv', encoding='utf-8-sig')
    
    # 세부 분류를 위한 OPTICS 재실행
    print("[2/4] OPTICS 세부 분류 (min_samples=3, xi=0.03)...")
    dtw_matrix = np.load(DATA_DIR + 'dtw_distance_matrix.npy')
    
    optics = OPTICS(min_samples=3, xi=0.03, metric='precomputed')
    labels = optics.fit_predict(dtw_matrix)
    
    # 기존 클러스터 레이블 로드 및 업데이트
    cluster_df = pd.read_csv(DATA_DIR + 'cluster_labels.csv', encoding='utf-8-sig')
    cluster_df['optics_detailed'] = labels
    cluster_df.to_csv(DATA_DIR + 'cluster_labels.csv', index=False, encoding='utf-8-sig')
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"    -> {n_clusters}개 클러스터, {n_noise}개 노이즈")
    
    # 상위 5개 클러스터 애니메이션 생성
    print("[3/4] 클러스터별 애니메이션 생성 중...")
    create_cluster_animations(seq_df, cluster_df, 'optics_detailed', n_clusters=5, n_samples=2)
    
    # 단일 샘플 애니메이션 (데모용)
    print("[4/4] 데모 애니메이션 생성...")
    sample_seq = cluster_df[cluster_df['optics_detailed'] != -1]['sequence_id'].iloc[0]
    demo_path = create_event_based_animation(seq_df, sample_seq, OUTPUT_DIR + 'demo_sequence.mp4', "전술 패턴 데모")
    print(f"    -> 데모 저장: {demo_path}")
    
    # 데모 정적 플롯
    static_demo_path = plot_sequence_static(seq_df, sample_seq, OUTPUT_DIR + 'demo_sequence_static.png', "전술 패턴 데모 (정적)")
    print(f"    -> 데모 정적 플롯 저장: {static_demo_path}")
    
    print("=" * 60)
    print(f"애니메이션 저장 완료: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
