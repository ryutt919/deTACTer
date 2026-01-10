# =========================================================
# animate_tactics.py
# deTACTer 프로젝트용 전술 애니메이션 생성 모듈 (v2.1)
# =========================================================
# 실제 경기장 규격 (105m x 68m)에 맞춘 애니메이션 생성
# 모든 선수 추적 + 이벤트 타입별 애니메이션 (리시브 이벤트 제외)
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
# 설정 및 경로 로드 (v3.2)
# =========================================================
import yaml
import os

CONFIG_PATH = 'c:/Users/Public/Documents/DIK/deTACTer/config.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 버전 관리 (v3.5 소프트 코딩 적용)
VERSION = config['version']
BASE_DIR = 'c:/Users/Public/Documents/DIK/deTACTer'

# 입력 및 출력 경로 설정 (버전별 폴더)
DATA_DIR = f"{BASE_DIR}/data/refined/{VERSION}/"
OUTPUT_DIR = f"{BASE_DIR}/results/animations/{VERSION}/"

# 폴더 생성 보장
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 실제 경기장 규격 (미터)
FIELD_LENGTH = 105
FIELD_WIDTH = 68

# 애니메이션 설정 (v3.9: 실시간 타이밍 + 높은 FPS)
FPS = 30  # 초당 프레임 수 (부드러운 애니메이션)
FRAME_DURATION = 1.0 / FPS  # 각 프레임의 실제 시간 (초)

# 시각적 요소 크기 (미터 단위)
PLAYER_RADIUS = 1.5  # 선수 원 반경 (사용자 요청: 1.5m)
BALL_RADIUS = 1.5    # 공 반경

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

TEAM_COLORS = {
    'attacking': '#3498db',  # 파랑 계열 (공격팀)
    'defending': '#e74c3c'   # 빨강 계열 (수비팀)
}

# =========================================================
# 축구장 그리기 (동일)
# =========================================================
def draw_pitch_real_scale(ax, grass_stripes=True):
    # 배경색 (진한 녹색)
    ax.set_facecolor('#1a6b1a')
    
    if grass_stripes:
        stripe_width = FIELD_LENGTH / 12
        for i in range(12):
            color = '#228B22' if i % 2 == 0 else '#1e7b1e'
            rect = patches.Rectangle((i * stripe_width, 0), stripe_width, FIELD_WIDTH,
                                    facecolor=color, edgecolor='none', zorder=0)
            ax.add_patch(rect)
    
    line_color = 'white'
    line_width = 2.5
    ax.plot([0, FIELD_LENGTH], [0, 0], color=line_color, linewidth=line_width, zorder=1)
    ax.plot([0, FIELD_LENGTH], [FIELD_WIDTH, FIELD_WIDTH], color=line_color, linewidth=line_width, zorder=1)
    ax.plot([0, 0], [0, FIELD_WIDTH], color=line_color, linewidth=line_width, zorder=1)
    ax.plot([FIELD_LENGTH, FIELD_LENGTH], [0, FIELD_WIDTH], color=line_color, linewidth=line_width, zorder=1)
    ax.plot([FIELD_LENGTH/2, FIELD_LENGTH/2], [0, FIELD_WIDTH], color=line_color, linewidth=2, zorder=1)
    
    ax.add_patch(plt.Circle((FIELD_LENGTH/2, FIELD_WIDTH/2), 9.15, fill=False, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(plt.Circle((FIELD_LENGTH/2, FIELD_WIDTH/2), 0.3, fill=True, color=line_color, zorder=2))
    
    # 박스 및 기타 요소는 생략하지 않고 구현 (기존 코드 유지)
    ax.add_patch(patches.Rectangle((0, (FIELD_WIDTH - 40.32) / 2), 16.5, 40.32, fill=False, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(patches.Rectangle((0, (FIELD_WIDTH - 18.32) / 2), 5.5, 18.32, fill=False, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(plt.Circle((11, FIELD_WIDTH/2), 0.25, fill=True, color=line_color, zorder=2))
    ax.add_patch(patches.Arc((11, FIELD_WIDTH/2), 18.3, 18.3, angle=0, theta1=308, theta2=52, color=line_color, linewidth=2, zorder=1))
    
    ax.add_patch(patches.Rectangle((FIELD_LENGTH - 16.5, (FIELD_WIDTH - 40.32) / 2), 16.5, 40.32, fill=False, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(patches.Rectangle((FIELD_LENGTH - 5.5, (FIELD_WIDTH - 18.32) / 2), 5.5, 18.32, fill=False, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(plt.Circle((FIELD_LENGTH - 11, FIELD_WIDTH/2), 0.25, fill=True, color=line_color, zorder=2))
    ax.add_patch(patches.Arc((FIELD_LENGTH - 11, FIELD_WIDTH/2), 18.3, 18.3, angle=0, theta1=128, theta2=232, color=line_color, linewidth=2, zorder=1))
    
    # 골대
    ax.plot([0, 0], [(FIELD_WIDTH-7.32)/2, (FIELD_WIDTH+7.32)/2], color='white', linewidth=5, zorder=2)
    ax.plot([FIELD_LENGTH, FIELD_LENGTH], [(FIELD_WIDTH-7.32)/2, (FIELD_WIDTH+7.32)/2], color='white', linewidth=5, zorder=2)
    
    ax.set_xlim(-5, FIELD_LENGTH + 5)
    ax.set_ylim(-5, FIELD_WIDTH + 5)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax

# =========================================================
# 카테고리 분류 (리시브 제외 로직 포함 시 필터링됨)
# =========================================================
def get_event_category(type_name, spadl_type=None):
    carry_types = ['Carry', 'Take-On', 'dribble', 'take_on']
    pass_types = ['Pass', 'Cross', 'Shot', 'Goal', 'Pass_Corner', 'Pass_Freekick', 
                  'Shot_Freekick', 'Throw-In', 'Goal Kick', 'pass', 'cross', 'shot']
    
    if type_name in carry_types or spadl_type in carry_types:
        return 'carry'
    elif type_name in pass_types or spadl_type in pass_types:
        return 'pass'
    else:
        return 'other'

# =========================================================
# 선수별 이벤트 추출 (순간이동 방지 로직 개선)
# =========================================================
def extract_player_events(seq_df):
    player_events = {}
    for idx, row in seq_df.iterrows():
        player_id = row['player_id']
        if pd.isna(player_id): continue
        if player_id not in player_events: player_events[player_id] = []
        
        category = get_event_category(row['type_name'], row.get('spadl_type', ''))
        
        # 선수의 "종료 위치" 결정
        # Carry나 Take-On은 end_x/y로 이동하지만, Pass나 Shot은 start_x/y에 남음
        end_x = row['end_x'] * FIELD_LENGTH if category == 'carry' and pd.notna(row['end_x']) else row['start_x'] * FIELD_LENGTH
        end_y = row['end_y'] * FIELD_WIDTH if category == 'carry' and pd.notna(row['end_y']) else row['start_y'] * FIELD_WIDTH
        
        player_events[player_id].append({
            'event_idx': idx,
            'start_x': row['start_x'] * FIELD_LENGTH,
            'start_y': row['start_y'] * FIELD_WIDTH,
            'end_x': end_x,
            'end_y': end_y,
            'type_name': row['type_name'],
            'category': category,
            'player_name': row.get('player_name_ko', '')[:6] if pd.notna(row.get('player_name_ko', '')) else ''
        })
    return player_events

# =========================================================
# 선수 위치 보간 (v3.9: 중복 이동 방지)
# =========================================================
def get_player_position(player_id, player_events, current_event_idx, progress):
    """
    선수의 현재 위치를 계산합니다.
    - current_event_idx: 현재 진행 중인 이벤트 인덱스
    - progress: 현재 이벤트 내에서의 진행도 (0.0 ~ 1.0)
    """
    if player_id not in player_events: 
        return FIELD_LENGTH / 2, FIELD_WIDTH / 2, False
    
    events = player_events[player_id]
    event_indices = [e['event_idx'] for e in events]
    
    # 1. 현재 이벤트에 선수가 참여 중
    if current_event_idx in event_indices:
        event = next(e for e in events if e['event_idx'] == current_event_idx)
        
        if event['category'] == 'carry':
            # 드리블/캐리: 시작 위치에서 종료 위치로 부드럽게 이동
            x = event['start_x'] + (event['end_x'] - event['start_x']) * progress
            y = event['start_y'] + (event['end_y'] - event['start_y']) * progress
        else:
            # 패스/슛 등: 시작 위치에 고정 (공만 이동)
            x, y = event['start_x'], event['start_y']
        
        return x, y, True  # active=True
    
    # 2. 현재 이벤트에 선수가 참여하지 않음
    # 가장 최근에 참여한 이벤트의 종료 위치에 머물러 있음
    past_events = [e for e in events if e['event_idx'] < current_event_idx]
    
    if past_events:
        # 마지막으로 참여한 이벤트의 종료 위치
        last_event = past_events[-1]
        return last_event['end_x'], last_event['end_y'], False
    
    # 3. 아직 첫 이벤트 전
    future_events = [e for e in events if e['event_idx'] > current_event_idx]
    if future_events:
        # 첫 이벤트의 시작 위치에서 대기
        first_event = future_events[0]
        return first_event['start_x'], first_event['start_y'], False
    
    # 4. 이벤트가 없는 선수 (중앙에 배치)
    return FIELD_LENGTH / 2, FIELD_WIDTH / 2, False

# =========================================================
# 애니메이션 생성
# =========================================================
def create_event_based_animation(seq_df, sequence_id, output_path, title="전술 패턴"):
    seq = seq_df[seq_df['sequence_id'] == sequence_id].sort_values('seq_position', ascending=False).reset_index(drop=True)
    
    # [고도화] 리시브 이벤트 필터링 및 전환 로직
    processed_events = []
    for _, row in seq.iterrows():
        type_name = row['type_name']
        sx, sy = row['start_x'], row['start_y']
        ex, ey = row['end_x'], row['end_y']
        
        if type_name in ['Pass Received', 'Ball Received']:
            # 시작/종료 좌표 차이가 매우 작은 경우 (정적 리시브) -> 제외
            if abs(sx - ex) < 0.001 and abs(sy - ey) < 0.01: # y축도 유사도 체크
                continue
            else:
                # 좌표가 다르면 Carry로 전환하여 이동 정보 보존
                row['type_name'] = 'Carry'
                row['spadl_type'] = 'dribble'
                processed_events.append(row)
        else:
            processed_events.append(row)
    
    if not processed_events: return None
    seq = pd.DataFrame(processed_events)

    # [고도화] 동일한 선수의 연속된 Carry 이벤트 병합
    merged_events = []
    for _, row in seq.iterrows():
        if not merged_events:
            merged_events.append(row.to_dict())
            continue
            
        last = merged_events[-1]
        # 같은 선수이고, 현재와 이전이 모두 Carry(또는 변환된 Carry)이며, 위치가 이어짐
        if (row['player_id'] == last['player_id'] and 
            get_event_category(row['type_name'], row.get('spadl_type', '')) == 'carry' and 
            get_event_category(last['type_name'], last.get('spadl_type', '')) == 'carry'):
            
            # 병합: 이전 이벤트의 종료 좌표를 현재의 종료 좌표로 업데이트 (연속 동작)
            last['end_x'] = row['end_x']
            last['end_y'] = row['end_y']
            # time_seconds도 업데이트 (v3.9)
            last['time_seconds'] = row['time_seconds']
            # is_outcome 플래그 보존
            if row['is_outcome']: last['is_outcome'] = True
        else:
            merged_events.append(row.to_dict())
    
    merged_seq = pd.DataFrame(merged_events)

    # 필터링: 리시브(정적) 제외하되, '성과 이벤트'는 보존
    mask = merged_seq['type_name'].isin(['Pass Received', 'Ball Received']) & (~merged_seq['is_outcome'])
    filtered_seq = merged_seq[~mask].reset_index(drop=True)
    
    if len(filtered_seq) == 0: return None
    
    player_events = extract_player_events(filtered_seq)
    
    # [v3.9] 실시간 타이밍 계산
    events = []
    attacking_team_id = filtered_seq['team_id'].iloc[-1] # 마지막 성과를 낸 팀이 공격팀
    
    # 시작 시간 기준 설정
    start_time = filtered_seq['time_seconds'].iloc[0]
    
    for idx, row in filtered_seq.iterrows():
        is_attacking = row['team_id'] == attacking_team_id
        events.append({
            'player_id': row['player_id'],
            'team_id': row['team_id'],
            'is_attacking': is_attacking,
            'start_x': row['start_x'] * FIELD_LENGTH,
            'start_y': row['start_y'] * FIELD_WIDTH,
            'end_x': row['end_x'] * FIELD_LENGTH if pd.notna(row['end_x']) else row['start_x'] * FIELD_LENGTH,
            'end_y': row['end_y'] * FIELD_WIDTH if pd.notna(row['end_y']) else row['start_y'] * FIELD_WIDTH,
            'type_name': row['type_name'],
            'category': get_event_category(row['type_name'], row.get('spadl_type', '')),
            'color': get_event_color(row['type_name'], row.get('spadl_type', '')),
            'team_color': TEAM_COLORS['attacking'] if is_attacking else TEAM_COLORS['defending'],
            'player_name': row.get('player_name_ko', '')[:6] if pd.notna(row.get('player_name_ko', '')) else '',
            'team_name': row.get('team_name_ko', 'Unknown'),
            'time_seconds': row['time_seconds'] - start_time  # 상대 시간
        })
    
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')
    draw_pitch_real_scale(ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=15)
    
    # 요소들
    ball_path_line, = ax.plot([], [], '-', color='yellow', linewidth=1.5, alpha=0.3, zorder=3)
    ball_path_x, ball_path_y = [], []
    
    player_markers = {}
    attacking_team_id = filtered_seq['team_id'].iloc[-1]
    for i, (pid, evs) in enumerate(player_events.items()):
        # 해당 선수가 소속된 팀 판단
        is_att = any(e['team_id'] == attacking_team_id for e in evs if 'team_id' in e) # 안전장치
        # 만약 evs에 team_id가 없으면 filtered_seq에서 찾음
        if not any('team_id' in e for e in evs):
            player_row = filtered_seq[filtered_seq['player_id'] == pid].iloc[0]
            is_att = player_row['team_id'] == attacking_team_id
            
        color = TEAM_COLORS['attacking'] if is_att else TEAM_COLORS['defending']
        
        circ = plt.Circle((evs[0]['start_x'], evs[0]['start_y']), PLAYER_RADIUS, facecolor=color, edgecolor='white', linewidth=1.5, zorder=5)
        ax.add_patch(circ)
        lbl = ax.text(0, 0, evs[0]['player_name'], ha='center', fontsize=7, color='white', zorder=6,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#333', alpha=0.7, edgecolor='none'))
        player_markers[pid] = {'circle': circ, 'label': lbl}
        
    ball_circ = plt.Circle((0,0), BALL_RADIUS, facecolor='white', edgecolor='black', linewidth=2, zorder=7)
    ax.add_patch(ball_circ)
    ball_pattern = RegularPolygon((0,0), 5, radius=BALL_RADIUS*0.6, facecolor='black', zorder=8)
    ax.add_patch(ball_pattern)
    
    event_label = ax.text(FIELD_LENGTH/2, FIELD_WIDTH+3, '', ha='center', fontsize=11, color='white', fontweight='bold', zorder=10,
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='#333', alpha=0.9))
    
    # [v3.9] 실시간 타이밍 기반 프레임 계산
    # 전체 시퀀스 지속 시간 계산
    total_duration = events[-1]['time_seconds'] + 1.0  # 마지막 이벤트 후 1초 추가
    total_frames = int(total_duration * FPS)
    
    event_markers = []

    def animate(frame):
        nonlocal ball_path_x, ball_path_y
        
        # 현재 프레임의 실제 시간 계산
        current_time = frame * FRAME_DURATION
        
        # 현재 시간에 해당하는 이벤트 찾기
        curr_idx = 0
        for i, ev in enumerate(events):
            if ev['time_seconds'] <= current_time:
                curr_idx = i
            else:
                break
        
        curr_ev = events[curr_idx]
        
        # 다음 이벤트가 있으면 보간 진행도 계산
        if curr_idx < len(events) - 1:
            next_ev = events[curr_idx + 1]
            event_duration = next_ev['time_seconds'] - curr_ev['time_seconds']
            if event_duration > 0:
                prog = (current_time - curr_ev['time_seconds']) / event_duration
                prog = min(max(prog, 0.0), 1.0)
            else:
                prog = 1.0
        else:
            prog = 1.0
        
        # 선수 위치
        for pid, mks in player_markers.items():
            px, py, active = get_player_position(pid, player_events, curr_idx, prog)
            mks['circle'].center = (px, py)
            mks['label'].set_position((px, py - PLAYER_RADIUS - 0.8))
            mks['circle'].set_linewidth(3 if active else 1.5)
            mks['circle'].set_edgecolor('yellow' if active else 'white')
            
        # 공 위치
        if curr_ev['category'] == 'carry':
            # 드리블 시 선수 위치 추적
            px, py, _ = get_player_position(curr_ev['player_id'], player_events, curr_idx, prog)
            bx, by = px + 1.2, py
        else:
            bx = curr_ev['start_x'] + (curr_ev['end_x'] - curr_ev['start_x']) * prog
            by = curr_ev['start_y'] + (curr_ev['end_y'] - curr_ev['start_y']) * prog
            
        ball_circ.center = (bx, by)
        ball_pattern.xy = (bx, by)
        
        if len(ball_path_x) == 0 or (bx != ball_path_x[-1]):
            ball_path_x.append(bx); ball_path_y.append(by)
            ball_path_line.set_data(ball_path_x, ball_path_y)
            
        event_label.set_text(f"[{curr_ev['team_name']}] {curr_ev['player_name']}: {curr_ev['type_name']}")
        event_label.set_color(curr_ev['color'])
        
        if prog >= 0.9 and len(event_markers) <= curr_idx:
            # 과거 이벤트 지점 마커 조정 (사용자 요청: 노란색, 투명도 0.6)
            m = ax.scatter([curr_ev['end_x']], [curr_ev['end_y']], color='yellow', s=40, alpha=0.6, edgecolor='none')
            event_markers.append(m)
            
        return [ball_path_line, ball_circ, ball_pattern, event_label] + [m['circle'] for m in player_markers.values()]
    
    # MP4 저장 (imageio 사용)
    try:
        import imageio
        # 프레임 캡처
        frames = []
        print(f"    프레임 생성 중... (총 {total_frames}프레임)")
        
        for i in range(total_frames):
            # artists 업데이트
            artists = animate(i)
            
            # 강제 렌더링 - 모든 artist를 비애니메이션 모드로 전환
            if artists:
                for artist in artists:
                    if artist is not None and hasattr(artist, 'set_animated'):
                        artist.set_animated(False)
            
            # 캔버스 렌더링
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # 프레임 캡처
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)
            frames.append(frame[:, :, :3])  # RGB만 사용 (알파 채널 제거)
            
            if (i + 1) % 50 == 0:
                print(f"      진행: {i+1}/{total_frames} ({100*(i+1)/total_frames:.1f}%)")
        
        # MP4로 저장
        print(f"    MP4 인코딩 중...")
        imageio.mimsave(output_path, frames, fps=FPS, codec='libx264', quality=8)
        print(f"    ✓ MP4 저장 완료: {output_path}")
    except Exception as e:
        print(f"    ✗ MP4 저장 실패: {e}")
        import traceback
        traceback.print_exc()
        # GIF로 폴백
        output_path = output_path.replace('.mp4', '.gif')
        anim = FuncAnimation(fig, animate, frames=total_frames, interval=int(FRAME_DURATION * 1000), blit=False)
        anim.save(output_path, writer=PillowWriter(fps=10), dpi=100)
        print(f"    ✓ GIF 저장 완료: {output_path}")
    
    plt.close(fig)
    return output_path

def get_event_color(type_name, spadl_type=None):
    if type_name in EVENT_COLORS: return EVENT_COLORS[type_name]
    return EVENT_COLORS.get(spadl_type, EVENT_COLORS['default'])

# 정적 플롯 (동일)
def plot_sequence_static(seq_df, sequence_id, output_path, title="전술 패턴"):
    seq = seq_df[seq_df['sequence_id'] == sequence_id].sort_values('seq_position', ascending=False).reset_index(drop=True)
    
    # 동일한 필터링/전환 로직 적용
    processed = []
    for _, row in seq.iterrows():
        sx, sy, ex, ey = row['start_x'], row['start_y'], row['end_x'], row['end_y']
        if row['type_name'] in ['Pass Received', 'Ball Received']:
            if abs(sx - ex) < 0.001 and abs(sy - ey) < 0.01: continue
            row['type_name'] = 'Carry'
        processed.append(row)
    
    if not processed: return None
    seq = pd.DataFrame(processed)
    fig, ax = plt.subplots(figsize=(14, 9)); fig.patch.set_facecolor('#1a1a2e')
    draw_pitch_real_scale(ax)
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=15)
    attacking_team_id = seq['team_id'].iloc[-1]
    px, py = None, None
    for i, row in seq.iterrows():
        x, y = row['start_x']*105, row['start_y']*68
        ex, ey = (row['end_x']*105 if pd.notna(row['end_x']) else x), (row['end_y']*68 if pd.notna(row['end_y']) else y)
        
        # 팀 색상 적용
        is_att = row['team_id'] == attacking_team_id
        t_color = TEAM_COLORS['attacking'] if is_att else TEAM_COLORS['defending']
        
        c = get_event_color(row['type_name'])
        if px is not None: ax.plot([px, x], [py, y], '-', color='yellow', alpha=0.3)
        if get_event_category(row['type_name']) == 'pass':
            ax.add_patch(FancyArrowPatch((x,y), (ex,ey), arrowstyle='fancy', color=c, alpha=0.6))
        
        # 정적 플롯 마커 수정 (팀 색상 적용)
        ax.scatter([x], [y], color=t_color, s=100, alpha=0.7, edgecolor='white', linewidth=1) 
        ax.annotate(f"{i+1}. {row['type_name']}", (x,y), color='white', fontsize=8)
        px, py = ex, ey
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    return output_path

def create_cluster_animations(seq_df, cluster_df, cluster_col, n_clusters=5, n_samples=2):
    top_clusters = cluster_df[cluster_col].value_counts().index[:n_clusters]
    for cid in top_clusters:
        if cid == -1: continue
        sids = cluster_df[cluster_df[cluster_col] == cid]['sequence_id'].values[:n_samples]
        for i, sid in enumerate(sids):
            create_event_based_animation(seq_df, sid, f"{OUTPUT_DIR}{cluster_col}_c{cid}_s{i+1}.mp4")
            plot_sequence_static(seq_df, sid, f"{OUTPUT_DIR}{cluster_col}_c{cid}_s{i+1}.png")

def main():
    seq_df = pd.read_csv(DATA_DIR + 'attack_sequences.csv', encoding='utf-8-sig')
    dtw_matrix = np.load(DATA_DIR + 'dtw_distance_matrix.npy')
    optics = OPTICS(min_samples=3, xi=0.03, metric='precomputed').fit_predict(dtw_matrix)
    cluster_df = pd.read_csv(DATA_DIR + 'cluster_labels.csv', encoding='utf-8-sig')
    cluster_df['optics_detailed'] = optics
    create_cluster_animations(seq_df, cluster_df, 'optics_detailed')
    sample_sid = cluster_df[cluster_df['optics_detailed'] != -1]['sequence_id'].iloc[0]
    create_event_based_animation(seq_df, sample_sid, OUTPUT_DIR + 'demo_sequence.mp4')
    plot_sequence_static(seq_df, sample_sid, OUTPUT_DIR + 'demo_sequence.png')

if __name__ == "__main__":
    main()
