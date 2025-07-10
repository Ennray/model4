import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# =====================
# 参数配置
# =====================
enemy_speed = 5           # 敌群速度
uav_speed = 8            # UAV速度
num_uavs = 5
uav_spacing = 5

# 敌群初始位置（右侧）
enemy_pos = np.array([50.0, 0.0])

# 敌群宽度（模拟散开）
enemy_width = 20.0

# UAV初始位置（左下方）
uav_positions = [np.array([-30.0, -20.0 - i * uav_spacing]) for i in range(num_uavs)]
turn_end = np.array([0.0, 20.0])  # UAV的预定转弯终点

# UAV状态标记（False=前飞阶段，True=跟随阶段）
uav_follow = [False for _ in range(num_uavs)]

# 存储轨迹
uav_histories = [[] for _ in range(num_uavs)]
enemy_history = []

frames = 200

# =====================
# 图形设置
# =====================
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-60, 60)
ax.set_ylim(-40, 60)
ax.set_aspect('equal')

enemy_rect = plt.Rectangle((enemy_pos[0]-enemy_width/2, enemy_pos[1]-5), enemy_width, 10,
                           edgecolor='red', facecolor='none', lw=2, label='enemy')
ax.add_patch(enemy_rect)

uav_dots = [ax.plot([], [], 'o', color='green', markersize=5)[0] for _ in range(num_uavs)]
uav_trails = [ax.plot([], [], '--', color='green', lw=1, alpha=0.5)[0] for _ in range(num_uavs)]

# =====================
# 更新函数
# =====================
def update(frame):
    global enemy_pos, enemy_rect

    # 更新敌群位置
    enemy_pos[:] = enemy_pos + np.array([-enemy_speed * 0.1, 0.0])
    enemy_history.append(enemy_pos.copy())

    # 更新敌群矩形
    enemy_rect.set_xy((enemy_pos[0] - enemy_width / 2, enemy_pos[1] - 5))

    for i, pos in enumerate(uav_positions):
        if not uav_follow[i]:
            # 飞向转弯终点
            direction = turn_end - pos
            dist = np.linalg.norm(direction)
            if dist > 1:
                dir_norm = direction / dist
                step = min(uav_speed * 0.1, dist)
                pos_new = pos + dir_norm * step
            else:
                uav_follow[i] = True
                pos_new = pos
        else:
            # 转弯后，飞向各自分散的尾后目标点
            enemy_center = np.array([enemy_pos[0], enemy_pos[1]])
            # 偏移逻辑：在敌群后方（x方向正），并在y方向上下散开
            base_offset_x = 15.0  # 后方横向偏移，正值表示在敌群后面（更右）
            base_offset_y = 10.0  # 上方基础偏移

            y_spacing = 5.0       # 无人机之间的纵向散开间距
            offset_x = base_offset_x
            offset_y = base_offset_y + (i - (num_uavs - 1)/2) * y_spacing

            target_pos = enemy_center + np.array([offset_x, offset_y])
            direction = target_pos - pos
            dist = np.linalg.norm(direction)
            if dist > 0.5:
                dir_norm = direction / dist
                step = min(uav_speed * 0.1, dist)
                pos_new = pos + dir_norm * step
            else:
                pos_new = pos

        uav_positions[i] = pos_new
        uav_histories[i].append(pos_new.copy())

        uav_dots[i].set_data([pos_new[0]], [pos_new[1]])
        trail_x = [p[0] for p in uav_histories[i]]
        trail_y = [p[1] for p in uav_histories[i]]
        uav_trails[i].set_data(trail_x, trail_y)

    return [enemy_rect] + uav_dots + uav_trails

# =====================
# 动画执行
# =====================
ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)

plt.legend()
plt.title("无人机迎头接敌、转弯、左后方分散包围敌群示意动画")
plt.show()

