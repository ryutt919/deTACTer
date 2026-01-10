"""
MP4 프레임 변화 테스트 스크립트
각 프레임이 실제로 변화하는지 확인
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)

# 움직이는 원
circle = Circle((0, 3), 0.5, color='red')
ax.add_patch(circle)

frames = []
for i in range(30):
    # 원 위치 업데이트
    x = i * 10 / 30
    circle.center = (x, 3)
    
    # 렌더링
    fig.canvas.draw()
    
    # 프레임 캡처
    buf = fig.canvas.buffer_rgba()
    frame = np.asarray(buf)
    frames.append(frame[:, :, :3])
    
    print(f"Frame {i}: center=({x:.2f}, 3)")

# MP4 저장
import imageio
imageio.mimsave('test_animation.mp4', frames, fps=10)
print("Test MP4 saved: test_animation.mp4")

plt.close()
