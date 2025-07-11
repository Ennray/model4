# 无人机协同侦察任务 - 跟随定位模块

## 概述

本模块实现了多波次无人机对敌机群的协同侦察任务，包括无人机运动规划、位置追踪、转弯时机预测和3D可视化等功能。

## 主要功能

### 1. 运动规划算法
- **`calculate_distances_opposite`**: 计算无人机与敌机迎面飞行时的运动时间
- **`calculate_distance_same`**: 计算无人机与敌机同向飞行时的运动时间  
- **`calculate_move_dist`**: 计算无人机完成追击和转弯需要移动的距离

### 2. 位置计算和转换
- **`find_edge_points`**: 在三维点集中找出边界点的最值
- **`geodetic_to_ecef`**: 将经纬高坐标转换为ECEF坐标
- **`ecef_to_enu_vector`**: 将ECEF向量转换为ENU坐标系
- **`calculate_direction_vector`**: 计算两个经纬度点的ENU方向向量

### 3. 任务规划
- **`second_turning_position`**: 计算第二波次无人机转弯后的位置
- **`first_turning_position`**: 计算第一波次无人机转弯后的占位布局
- **`classify_uav_turning`**: 对第二波次无人机进行转弯策略分类
- **`predict_cross_time`**: 预测第一波无人机与敌群的交错时间

### 4. 实时追踪
- **`update_positions_geo`**: 实时更新敌我双方无人机的经纬度坐标
- **`get_uav_turning_points`**: 获取无人机转弯时的起点和终点位置

### 5. 可视化
- **`plot_moving_3D`**: 绘制敌我双方无人机的3D运动轨迹
- **`plot_initial_positions_with_centers`**: 绘制初始位置的3D分布图

### 6. 速度调整
- **`adjust_speed`**: 根据敌机速度计算无人机适当跟飞速度范围
- **`relative_position_difference`**: 计算两组无人机之间的相对位置差

## 核心算法

### 运动模型
系统采用三段式运动模型：
1. **加速阶段**: 从起始速度加速到最大速度
2. **匀速阶段**: 以最大速度飞行
3. **减速阶段**: 从最大速度减速到目标速度

### 坐标系统
- **WGS84**: 输入的经纬度坐标系统
- **ECEF**: 地心地固坐标系统（中间转换）
- **ENU**: 东北天局部坐标系统（计算使用）

### 决策逻辑
1. **距离判断**: 根据与基准点的距离决定转弯策略
2. **时机预测**: 通过相对运动预测最佳转弯时机
3. **位置分配**: 根据敌机群边界分配无人机占位

## 参数说明

### 速度参数 (uav_speeds)
- `[0]` 当前速度 (m/s)
- `[1]` 加速度 (m/s²)
- `[2]` 减速度 (m/s²) 
- `[3]` 最大速度 (m/s)
- `[4]` 转弯速度 (m/s)

### 几何参数
- `distance_follow`: 伴飞距离 (m)
- `detection_size`: 探测区域大小 [x, z] (m)
- `height`: 高度差 (m)
- `y_gap`: Y方向间距 (m)

### 阈值参数
- `base_distance_threshold`: 基准距离阈值 (m)
- `turn_distance_threshold`: 转弯距离阈值 (m)
- `dt`: 时间步长 (s)

## 数据格式

### 位置数据格式
```python
position = ["28:11:37.48W", "10:31:48.11N", "1029.40"]
# [经度（度:分:秒方向）, 纬度（度:分:秒方向）, 高度（米）]
```

### 决策输出格式
```python
decisions = [1, 0, 1, 0, ...]
# 1: 跟随第一波转弯
# 0: 独立转弯
```

## 使用示例

```python
# 创建坐标转换器
converter = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)

# 第二波次转弯决策
decisions = classify_uav_turning(second_points, base, converter, 1000)

# 预测交错时间
predicted_time, _, _ = predict_cross_time(base, enemy_center, 180, 240, converter, 500, 1)

# 计算转弯后位置
final_positions = second_turning_position(
    enemy_geo, enemy_center, second_points, base,
    240, [180, 80, 80, 300, 200], 15, 50, [200, 300], 500, 200
)

# 3D可视化
plot_moving_3D(enemy_geo, second_points, enemy_center, base, 240, 180, converter, 120, 1)
```

## 依赖模块

- `numpy`: 数值计算
- `matplotlib`: 图形绘制
- `math`: 数学运算
- `sympy`: 符号计算
- `GeodeticConverter`: 坐标转换（自定义模块）
- `velocity_recong`: 速度识别（自定义模块）

## 注意事项

1. **坐标系统**: 确保输入的经纬度格式正确（度:分:秒格式）
2. **单位统一**: 所有距离单位为米，时间单位为秒，速度单位为m/s
3. **参数合理性**: 确保速度、加速度等参数符合实际物理约束
4. **计算精度**: 符号计算可能较慢，可考虑数值解法优化

## 版本历史

- v1.0: 初始版本，包含基本功能
- v1.1: 代码重构，删除冗余代码，完善文档

## 作者

[您的姓名]  
2025年7月11日
