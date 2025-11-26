
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

import GeodeticConverter as GC
from geopy.distance import geodesic
import plotly.graph_objects as go

from test_interface_data import config as test_config
import math


# ====================================================== 工具函数 =======================================================

def _nudge_lat_deg(north_m: float) -> float:
    """把纬度往北挪 north_m 米对应的度数（1°≈111320 m）。"""
    return north_m / 111_320.0


# 坐标dms转degree，已经是degree那就不影响
def dms_to_degree(dms: Any) -> Tuple[float, float, float]:

    if isinstance(dms, (list, tuple)) and len(dms) == 3 and isinstance(dms[0], str):
        lat, lon, alt = GC.decimal_dms_to_degrees(dms)
        return float(lat), float(lon), float(alt)
    return float(dms[0]), float(dms[1]), float(dms[2])


# 坐标dms转local
def dms_to_local(converter: "GC.GeodeticToLocalConverter", dms: Any) -> np.ndarray:
    lat, lon, alt = dms_to_degree(dms)
    return np.asarray(converter.geodetic_to_local(lat, lon, alt), dtype=float)


# 建立全局坐标系
def build_global_converter(uav_center, enemy_center):
    uav_lat, uav_lon, uav_alt = dms_to_degree(uav_center)
    enemy_lat, enemy_lon, enemy_alt = dms_to_degree(enemy_center)
    converter = GC.GeodeticToLocalConverter(uav_lat, uav_lon, uav_alt, enemy_lat, enemy_lon, enemy_alt)
    R_local_to_ENU = converter.rotation_matrix
    R_ENU_to_local = R_local_to_ENU.T

    return converter, R_local_to_ENU, R_ENU_to_local

# 计算两点之间的距离
def diff_between_pos(lat1, lon1, lat2, lon2):
    diff_distance = geodesic((lat1, lon1), (lat2, lon2)).meters
    return diff_distance



# ================================================= 核心算法 ==================================================

# 判断是否有敌机中心，没有则自己计算
def judge_enemy_center(enemy_center, enemy_dms):
    if enemy_center is None:
        degs = np.array([dms_to_degree(x) for x in enemy_dms], float)  # [lat, lon, alt]
        lat0 = float(np.mean(degs[:, 0]))
        lon0 = float(np.mean(degs[:, 1]))
        alt0 = float(np.mean(degs[:, 2])) if degs.shape[1] == 3 else 0.0
        enemy_center = [lat0, lon0, alt0]

    return enemy_center


# 计算敌群/我机在n秒后的位置
def calculate_position(enemy_dms, enemy_speed, bearing, time, converter):
    next_enemy_dms = []

    print("bearing", bearing)
    for i, uav in enumerate(enemy_dms):
        enemy_lat, enemy_lon, enemy_alt = dms_to_degree(uav)
        next_enemy_lat, next_enemy_lon, next_enemy_alt = converter.calculate_destination_point(enemy_lat, enemy_lon, enemy_alt, bearing, time * enemy_speed)

        new_enemy_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(next_enemy_lat, next_enemy_lon, next_enemy_alt))
        next_enemy_dms.append(new_enemy_dms)

    return next_enemy_dms


# 计算C点理想位置（最小数量）- 网格支配集算法
def compute_C_points_least(
        enemy_dms: List[Any],
        n_rows: int,
        n_cols: int,
        converter: "GC.GeodeticToLocalConverter",
        high_distance: float,
        coverage_neighbors: int = 4,
) -> Tuple[List[Any], List[int]]:
    """
    使用网格支配集算法计算最少C点数量，确保100%覆盖且排布整齐。

    基于已知信息：每个C点最多覆盖5架敌机（自己+上下左右4个邻居）

    参数:
        enemy_dms: 敌机位置列表 [[lon, lat, alt], ...]
        n_rows: 敌机编队行数
        n_cols: 敌机编队列数
        converter: 坐标转换器
        high_distance: 抬升高度（米）
        coverage_neighbors: 覆盖邻居数（默认4，即上下左右）

    返回:
        (c_points, c_indices): C点列表和对应的敌机索引列表
    """
    if not enemy_dms:
        return [], []

    total = len(enemy_dms)

    # 1. 将敌机转换为局部坐标，计算相邻距离阈值
    enemy_local = []
    for dms in enemy_dms:
        local = dms_to_local(converter, dms)
        enemy_local.append(local)
    enemy_local = np.array(enemy_local)
    enemy_xy = enemy_local[:, :2]  # (N, 2)

    # 2. 计算网格中相邻敌机的平均距离（用于判断邻居）
    neighbor_distances = []
    for i in range(min(5, total)):  # 检查前几个敌机
        if i + 1 < total:
            dist = np.linalg.norm(enemy_xy[i] - enemy_xy[i + 1])
            neighbor_distances.append(dist)
        if i + n_cols < total:
            dist = np.linalg.norm(enemy_xy[i] - enemy_xy[i + n_cols])
            neighbor_distances.append(dist)

    if neighbor_distances:
        avg_neighbor_dist = np.mean(neighbor_distances)
        neighbor_threshold = avg_neighbor_dist * 1.3  # 容差30%
    else:
        neighbor_threshold = 1000.0  # 默认1000米

    print(f"相邻敌机距离阈值: {neighbor_threshold:.1f}m")

    # 3. 安全的索引映射函数
    def safe_index(r: int, c: int) -> int | None:
        idx = r * n_cols + c
        if 0 <= idx < total:
            return idx
        return None

    # 4. 计算某个位置能覆盖哪些敌机（自己+邻居）

    def get_coverage(center_idx: int) -> set:
        """返回以center_idx为中心的C点能覆盖的所有敌机索引"""
        covered = {center_idx}  # 首先覆盖自己
        center_pos = enemy_xy[center_idx]

        # 检查所有敌机，找出距离小于阈值的邻居
        for i in range(total):
            if i == center_idx:
                continue
            dist = np.linalg.norm(enemy_xy[i] - center_pos)
            if dist <= neighbor_threshold:
                covered.add(i)
        return covered

    # 5. 规整排布策略：每隔2个位置放置C点（网格支配集）
    # 模式：(0,0), (0,2), (0,4), ..., (2,0), (2,2), (2,4), ...
    selected_indices = []

    for r in range(0, n_rows, 2):  # 行：0, 2, 4, ...
        for c in range(0, n_cols, 2):  # 列：0, 2, 4, ...
            idx = safe_index(r, c)
            if idx is not None:
                selected_indices.append(idx)

    # 6. 验证覆盖率
    covered = set()
    for idx in selected_indices:
        covered.update(get_coverage(idx))

    initial_coverage = len(covered)
    print(f"初始规整排布覆盖: {initial_coverage}/{total} = {initial_coverage / total * 100:.1f}%")

    # 7. 贪心补充未覆盖的敌机
    uncovered = set(range(total)) - covered
    iteration = 0
    max_iterations = 100

    while uncovered and iteration < max_iterations:
        # 找到能覆盖最多未覆盖敌机的位置
        best_idx = None
        best_new_coverage = 0

        for idx in range(total):
            if idx in selected_indices:
                continue
            # 计算该位置能覆盖多少未覆盖敌机
            coverage = get_coverage(idx)
            new_coverage_count = len(coverage & uncovered)

            if new_coverage_count > best_new_coverage:
                best_new_coverage = new_coverage_count
                best_idx = idx

        if best_idx is not None and best_new_coverage > 0:
            selected_indices.append(best_idx)
            new_covered = get_coverage(best_idx)
            covered.update(new_covered)
            uncovered -= new_covered
            print(f"补充C点 #{best_idx}，新覆盖 {best_new_coverage} 架敌机，剩余未覆盖 {len(uncovered)}")
        else:
            # 无法找到能覆盖的点，可能网格不规则，直接补充
            if uncovered:
                idx = list(uncovered)[0]
                selected_indices.append(idx)
                new_covered = get_coverage(idx)
                covered.update(new_covered)
                uncovered -= new_covered
                print(f"强制补充C点 #{idx}")

        iteration += 1

    # 8. 按行列顺序排序（保持规整）
    def get_row_col(idx):
        return (idx // n_cols, idx % n_cols)
    selected_indices.sort(key=get_row_col)

    # 9. 生成C点列表（抬升高度）
    c_points = []
    for idx in selected_indices:
        dms = enemy_dms[idx]
        # 创建新的列表，不修改原始数据
        lon_str, lat_str, _ = dms[0], dms[1], dms[2]
        # 获取当前高度并抬升
        _, _, current_alt = dms_to_degree(dms)
        new_alt = current_alt + high_distance
        new_alt_str = str(new_alt) if isinstance(dms[2], str) else new_alt
        c_points.append([lon_str, lat_str, new_alt_str])

    # 10. 计算覆盖统计
    coverage_counts = {}  # 每架敌机被覆盖的次数
    for i in range(total):
        coverage_counts[i] = 0

    for idx in selected_indices:
        for covered_idx in get_coverage(idx):
            coverage_counts[covered_idx] += 1

    # 统计重复覆盖情况
    no_coverage = sum(1 for c in coverage_counts.values() if c == 0)
    single_coverage = sum(1 for c in coverage_counts.values() if c == 1)
    multi_coverage = sum(1 for c in coverage_counts.values() if c > 1)
    max_coverage = max(coverage_counts.values())
    avg_coverage = np.mean(list(coverage_counts.values()))

    print(f"\n网格支配集算法结果:")
    print(f"  - 敌机编队: {n_rows}行 × {n_cols}列 = {total}架")
    print(f"  - C点数量: {len(c_points)}")
    print(f"  - 覆盖率: {len(covered)}/{total} = {len(covered) / total * 100:.1f}%")
    print(f"  - 减少比例: {(1 - len(c_points) / total) * 100:.1f}%")
    print(f"  - 覆盖统计:")
    print(f"    · 未覆盖: {no_coverage} 架")
    print(f"    · 单次覆盖: {single_coverage} 架")
    print(f"    · 多次覆盖: {multi_coverage} 架")
    print(f"    · 最大覆盖次数: {max_coverage}")
    print(f"    · 平均覆盖次数: {avg_coverage:.2f}")
    print(f"  - 排布模式: 规整网格（每隔2行2列）+ 贪心补充")

    return c_points, selected_indices


# 计算C点理想位置（限制数量）- 最大覆盖贪心算法
def compute_C_points_limit(
        enemy_dms: List[Any],
        n_rows: int,
        n_cols: int,
        converter: "GC.GeodeticToLocalConverter",
        high_distance: float,
        max_c_points: int,
        coverage_neighbors: int = 4,
        custom_radius: float = None,  # 新增：自定义覆盖半径（米）
) -> Tuple[List[Any], List[int]]:
    """
    在限制C点数量的情况下，使用贪心算法最大化覆盖率。

    基于需求：350架敌机，最多使用117个C点（1:3覆盖比）

    参数:
        enemy_dms: 敌机位置列表 [[lon, lat, alt], ...]
        n_rows: 敌机编队行数
        n_cols: 敌机编队列数
        converter: 坐标转换器
        high_distance: 抬升高度（米）
        max_c_points: C点数量上限
        coverage_neighbors: 覆盖邻居数（默认4，即上下左右）

    返回:
        (c_points, c_indices): C点列表和对应的敌机索引列表
    """
    if not enemy_dms:
        return [], []

    total = len(enemy_dms)

    # 如果上限大于等于敌机数，直接调用least方法
    if max_c_points >= total:
        print(f"C点上限({max_c_points})大于敌机数({total})，使用compute_C_points_least")
        return compute_C_points_least(enemy_dms, n_rows, n_cols, converter, high_distance, coverage_neighbors)

    # 1. 将敌机转换为局部坐标，计算相邻距离阈值
    enemy_local = []
    for dms in enemy_dms:
        local = dms_to_local(converter, dms)
        enemy_local.append(local)
    enemy_local = np.array(enemy_local)
    enemy_xy = enemy_local[:, :2]  # (N, 2)

    # 2. 计算网格中相邻敌机的平均距离（用于判断邻居）
    neighbor_distances = []
    for i in range(min(10, total)):  # 检查前几个敌机
        if i + 1 < total:
            dist = np.linalg.norm(enemy_xy[i] - enemy_xy[i + 1])
            neighbor_distances.append(dist)
        if i + n_cols < total:
            dist = np.linalg.norm(enemy_xy[i] - enemy_xy[i + n_cols])
            neighbor_distances.append(dist)

    if neighbor_distances:
        avg_neighbor_dist = np.mean(neighbor_distances)

        # 如果指定了自定义覆盖半径，直接使用
        if custom_radius is not None:
            neighbor_threshold = custom_radius
            radius_multiplier = custom_radius / avg_neighbor_dist
            print(f"使用自定义覆盖半径: {custom_radius:.1f}m")
        else:
            # 根据coverage_neighbors动态计算覆盖半径
            # coverage_neighbors=4: 覆盖上下左右（1层）
            # coverage_neighbors=8: 覆盖周围8个（1层+对角）
            # coverage_neighbors=12: 覆盖2层
            # coverage_neighbors=20: 覆盖3层
            if coverage_neighbors <= 4:
                radius_multiplier = 1.5  # 覆盖直接相邻
            elif coverage_neighbors <= 8:
                radius_multiplier = 2.2  # 覆盖对角线（√2倍距离）
            elif coverage_neighbors <= 12:
                radius_multiplier = 2.8  # 覆盖2层
            elif coverage_neighbors <= 20:
                radius_multiplier = 3.5  # 覆盖3层
            else:
                radius_multiplier = 1.0 + coverage_neighbors / 10  # 更大范围

            neighbor_threshold = avg_neighbor_dist * radius_multiplier
    else:
        neighbor_threshold = custom_radius if custom_radius is not None else 1000.0
        radius_multiplier = neighbor_threshold / 100.0  # 估算值
        avg_neighbor_dist = 100.0  # 默认值

    print(f"覆盖目标邻居数: {coverage_neighbors}")
    print(f"相邻敌机平均距离: {avg_neighbor_dist:.1f}m")
    print(f"覆盖半径倍数: {radius_multiplier:.2f}")
    print(f"实际覆盖半径: {neighbor_threshold:.1f}m")

    # 3. 计算某个位置能覆盖哪些敌机（自己+邻居）
    def get_coverage(center_idx: int) -> set:
        """返回以center_idx为中心的C点能覆盖的所有敌机索引"""
        covered = {center_idx}  # 首先覆盖自己
        center_pos = enemy_xy[center_idx]

        # 检查所有敌机，找出距离小于阈值的邻居
        for i in range(total):
            if i == center_idx:
                continue
            dist = np.linalg.norm(enemy_xy[i] - center_pos)
            if dist <= neighbor_threshold:
                covered.add(i)

        return covered

    # 4. 预计算所有位置的覆盖情况（加速后续查询）
    all_coverage = {}
    for i in range(total):
        all_coverage[i] = get_coverage(i)

    print(f"预计算完成，每个位置平均覆盖: {np.mean([len(c) for c in all_coverage.values()]):.2f} 架敌机")

    # 5. 贪心算法：每次选择能覆盖最多未覆盖敌机的位置
    selected_indices = []
    covered = set()
    uncovered = set(range(total))

    print(f"\n开始贪心选择，目标C点数量: {max_c_points}")

    for iteration in range(max_c_points):
        if not uncovered:
            print(f"第{iteration}轮：所有敌机已覆盖，提前结束")
            break

        # 找到能覆盖最多未覆盖敌机的位置
        best_idx = None
        best_new_coverage = 0
        best_coverage_set = set()

        for idx in range(total):
            if idx in selected_indices:
                continue

            # 计算该位置能覆盖多少未覆盖敌机
            coverage = all_coverage[idx]
            new_coverage = coverage & uncovered
            new_coverage_count = len(new_coverage)

            if new_coverage_count > best_new_coverage:
                best_new_coverage = new_coverage_count
                best_idx = idx
                best_coverage_set = coverage

        if best_idx is not None:
            selected_indices.append(best_idx)
            covered.update(best_coverage_set)
            uncovered -= best_coverage_set

            if (iteration + 1) % 10 == 0 or iteration < 5:
                print(f"第{iteration+1}轮：选择C点#{best_idx}，新覆盖{best_new_coverage}架，累计覆盖{len(covered)}/{total} ({len(covered)/total*100:.1f}%)")
        else:
            print(f"第{iteration+1}轮：无法找到新的覆盖位置，提前结束")
            break

    # 6. 按行列顺序排序（保持一定规整性）
    def get_row_col(idx):
        return (idx // n_cols, idx % n_cols)

    selected_indices.sort(key=get_row_col)

    # 7. 生成C点列表（抬升高度）
    c_points = []
    for idx in selected_indices:
        dms = enemy_dms[idx]
        # 创建新的列表，不修改原始数据
        lon_str, lat_str, _ = dms[0], dms[1], dms[2]
        # 获取当前高度并抬升
        _, _, current_alt = dms_to_degree(dms)
        new_alt = current_alt + high_distance
        new_alt_str = str(new_alt) if isinstance(dms[2], str) else new_alt
        c_points.append([lon_str, lat_str, new_alt_str])

    # 8. 计算详细覆盖统计
    coverage_counts = {}  # 每架敌机被覆盖的次数
    for i in range(total):
        coverage_counts[i] = 0

    for idx in selected_indices:
        for covered_idx in all_coverage[idx]:
            coverage_counts[covered_idx] += 1

    # 统计重复覆盖情况
    no_coverage = sum(1 for c in coverage_counts.values() if c == 0)
    single_coverage = sum(1 for c in coverage_counts.values() if c == 1)
    multi_coverage = sum(1 for c in coverage_counts.values() if c > 1)
    max_coverage_times = max(coverage_counts.values()) if coverage_counts else 0
    avg_coverage = np.mean(list(coverage_counts.values()))

    # 计算实际覆盖比
    actual_ratio = len(covered) / len(selected_indices) if selected_indices else 0

    print(f"\n限制数量贪心算法结果:")
    print(f"  - 敌机编队: {n_rows}行 × {n_cols}列 = {total}架")
    print(f"  - C点上限: {max_c_points}")
    print(f"  - 实际C点: {len(c_points)}")
    print(f"  - 覆盖敌机: {len(covered)}/{total} = {len(covered)/total*100:.2f}%")
    print(f"  - 未覆盖: {no_coverage} 架敌机")
    print(f"  - 实际覆盖比: 1:{actual_ratio:.2f} (目标 1:3)")
    print(f"  - 覆盖统计:")
    print(f"    · 未覆盖: {no_coverage} 架")
    print(f"    · 单次覆盖: {single_coverage} 架")
    print(f"    · 多次覆盖: {multi_coverage} 架")
    print(f"    · 最大覆盖次数: {max_coverage_times}")
    print(f"    · 平均覆盖次数: {avg_coverage:.2f}")
    print(f"  - 算法: 贪心最大覆盖")

    # 9. 如果覆盖率不足，给出警告和建议
    coverage_rate = len(covered) / total
    if coverage_rate < 0.95:
        print(f"\n 警告：覆盖率仅 {coverage_rate*100:.2f}%，低于95%")
        print(f"   建议：")
        needed_for_95 = int(np.ceil(total * 0.95 / actual_ratio)) if actual_ratio > 0 else max_c_points
        needed_for_100 = int(np.ceil(total / actual_ratio)) if actual_ratio > 0 else max_c_points
        print(f"   - 达到95%覆盖需要约 {needed_for_95} 个C点")
        print(f"   - 达到100%覆盖需要约 {needed_for_100} 个C点")
    elif coverage_rate < 1.0:
        print(f"\n 覆盖率 {coverage_rate*100:.2f}%，接近目标但仍有 {no_coverage} 架敌机未覆盖")
    else:
        print(f"\n 已实现100%覆盖！")

    return c_points, selected_indices


# 计算C点理想位置，默认N为按照纬度也就是行来进行排布，E是按经度也就是列来排布
def compute_C_points(
        enemy_dms: List[Any],
        n_rows: int,
        n_cols: int,
        high_distance: float,
        mode: str = "row",
) -> Tuple[List[Any], List[int]]:

    total = len(enemy_dms)

    # 小工具：安全的 (r, c) → index 映射，超出范围返回 None
    def safe_index(r: int, c: int) -> int | None:
        idx = r * n_cols + c
        if 0 <= idx < total:
            return idx
        return None

    # 1) 决定要选哪些“行/列索引”（0-based）
    mode_low = mode.lower()
    if mode_low == "row":
        # 选行：1,3,5,...；如果是奇数行数，再补上最后一行
        if n_rows % 2 == 0:
            selected_rows = list(range(1, n_rows, 2))
        else:
            selected_rows = list(range(1, n_rows, 2)) + [n_rows - 1]

        c_indices: List[int] = []
        for r in selected_rows:
            # 对这一行的所有列尝试取 index，但只保留真实存在的敌机
            for c in range(n_cols):
                idx = safe_index(r, c)
                if idx is None:
                    # 这一行后面已经没有敌机了，直接跳出这一行
                    break
                c_indices.append(idx)

    elif mode_low == "col":
        # 选列：1,3,5,...；如果是奇数列数，再补上最后一列
        if n_cols % 2 == 0:
            selected_cols = list(range(1, n_cols, 2))
        else:
            selected_cols = list(range(1, n_cols, 2)) + [n_cols - 1]

        c_indices: List[int] = []
        for r in range(n_rows):
            for c in selected_cols:
                idx = safe_index(r, c)
                if idx is None:
                    # 这一行在这个列上没有敌机，就跳过（但这一行的其他列还可能有）
                    continue
                c_indices.append(idx)
    else:
        raise ValueError(f"mode必须是'row'或'col'，当前是：{mode}")

    # 防止空结果
    if not c_indices:
        return [], []

    # 2) 先根据 c_indices 从 enemy_dms 里挑出“原始点”（只读，不改）
    raw_c_points = [enemy_dms[i] for i in c_indices]

    # 3) 用原始 C 点的高度算最大值（不改原数组）
    max_alt = 0.0
    for point in raw_c_points:
        max_alt = max(max_alt, float(point[2]))
    print("海拔：", max_alt)

    # 4) 构造一个“新”的 C 点列表，只修改高度，不动 enemy_dms
    new_alt_str = str(max_alt + high_distance)
    c_points: List[Any] = []
    for lon_str, lat_str, _ in raw_c_points:
        # 这里创建新的 [lon, lat, alt]，不引用原 list
        c_points.append([lon_str, lat_str, new_alt_str])

    return c_points, c_indices
# 反推B点
def compute_B_point(A_dms, C_dms, bearing_deg,
                    uav_start_speed, uav_end_speed, time_to_interference, time_BC,
                    acc_max, uav_max_speed, converter):

    B_points = []
    # 对于每一架无人机，根据其A点和C点来确定B点
    for i, uav in enumerate(A_dms):
        latA, lonA, altA = dms_to_degree(uav)
        latC, lonC, altC = dms_to_degree(C_dms[i])


        distance_AC = converter.calculate_spherical_distance(latA, lonA, altA, latC, lonC, altC)

        # AB的时间，就等于总时间-BC时间
        time_AB = time_to_interference - time_BC

        # 计算VB
        uav_speed_B = (2 * distance_AC - uav_start_speed * time_AB - uav_end_speed * time_BC) / time_to_interference

        # 计算加速度
        a1 = (uav_speed_B - uav_start_speed) / time_AB
        a2 = (uav_end_speed - uav_speed_B) / time_BC

        # 计算AB距离
        distance_AB = 0.5 * (uav_start_speed + uav_speed_B) / time_AB

        # 沿着bearing反向求B点
        latB, lonB, altB = converter.calculate_destination_point(latC, lonC, altC, bearing_deg, distance_AB, 0)
        pointB = converter.local_to_geodetic_dms(converter.geodetic_to_local(latB, lonB, altB))
        B_points.append([pointB,uav_speed_B])

    return B_points


# 由相邻距离判断行数列数
def calculate_formation(enemy_position, threshold_distance=1000):

    deg_positions = [dms_to_degree(p) for p in enemy_position]

    num_rows = 1  # 初始化行数为1
    drones_per_row = [1]  # 第一行默认有1架无人机
    prev_lat, prev_lon, _= deg_positions[0]

    for i in range(1, len(deg_positions)):
        curr_lat, curr_lon, _ = deg_positions[i]

        # 计算相邻两个点之间的距离
        distance = diff_between_pos(prev_lat, prev_lon, curr_lat, curr_lon)

        if distance > threshold_distance:
            # 如果距离大于阈值，则换行
            num_rows += 1
            drones_per_row.append(1)  # 新的一行开始，第一架无人机
        else:
            # 如果距离小于阈值，则当前行增加无人机
            drones_per_row[-1] += 1

        prev_lat, prev_lon = curr_lat, curr_lon  # 更新前一个点位为当前点位

    return num_rows, max(drones_per_row)



def layered_computation(enemy_positions, height_tolerance=30.0):

    # 对敌群坐标进行排序，按照海拔高度进行排序
    sorted_positions = sorted(enemy_positions, key=lambda pos: pos[2])

    # 按高度差简单分类，差值<=height_tolerance认为是同一层
    layers_dms = []
    current_layer = [sorted_positions[0]]
    base_height = sorted_positions[0][2]

    for pos in sorted_positions[1:]:
        h = pos[2]
        if abs(h - base_height) <= height_tolerance:
            # 和当前层高度差不多，视为同一层
            current_layer.append(pos)
        else:
            # 开启新的一层
            layers_dms.append(current_layer)
            current_layer = [pos]
            base_height = h
    # 最后一层
    layers_dms.append(current_layer)

    return layers_dms




# ================= I/O 与可视化 =================

def load_enemy_dms(json_path: Path) -> List[Any]:
    """
    读取敌机经纬高列表。兼容两种结构：
    1) {"enemy_dms": [[lat, lon, alt], ...]}
    2) [[lat, lon, alt], ...]
    """
    obj = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "enemy_dms" in obj:
        data = obj["enemy_dms"]
    else:
        data = obj
    if not isinstance(data, list) or not data:
        raise ValueError("enemy_dms is empty or invalid.")
    return data


def save_c_points(c_points: List[List[float]], out_path: Path) -> None:
    out = {"c_points_dms": c_points}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_lonlat_points(enemy_dms: List[Any], c_points: List[List[float]], out_png: Path) -> None:
    """
    敌机与 C 点都用散点（点样式），经纬度轴使用等比例（same step length），
    两轴取相同的可视范围并对齐刻度步长，便于直观对比。
    """
    import matplotlib.ticker as mticker
    enemy_deg = np.array([dms_to_degree(x) for x in enemy_dms], float)  # [lat, lon, alt]
    c_deg     = np.array([dms_to_degree(x) for x in c_points], float)

    lat_e, lon_e = enemy_deg[:, 0], enemy_deg[:, 1]
    lat_c, lon_c = c_deg[:, 0], c_deg[:, 1]

    # 合并范围
    xmin = float(min(lon_e.min(), lon_c.min()))
    xmax = float(max(lon_e.max(), lon_c.max()))
    ymin = float(min(lat_e.min(), lat_c.min()))
    ymax = float(max(lat_e.max(), lat_c.max()))

    # 两轴用相同的跨度（span）与中心，保证“同样长度的步长”
    cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
    span = max(xmax - xmin, ymax - ymin)
    pad = 0.05 * span if span > 0 else 1e-4
    span = span + 2 * pad  # 外扩 5%

    # 画图
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(lon_e, lat_e, s=10, marker='.', label="Enemy")
    ax.scatter(lon_c, lat_c, s=12, marker='.', label="C")

    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_aspect('equal', adjustable='box')  # 等比例（一度经 == 一度纬 的视觉步长）

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Enemy vs C points (lat-lon) — equal step length")

    # 让两轴使用同一个刻度“步长”
    target_ticks = 6
    raw_step = span / target_ticks

    def nice_step(x: float) -> float:
        # 把步长变成 1/2/5*10^k 的“整洁值”
        import math
        if x <= 0:
            return 1.0
        exp = math.floor(np.log10(x))
        base = x / (10 ** exp)
        if base <= 1:
            n = 1
        elif base <= 2:
            n = 2
        elif base <= 5:
            n = 5
        else:
            n = 10
        return n * (10 ** exp)

    step = nice_step(raw_step)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 核心函数，负责整个算法的调度并给出结果
def verify_coverage_and_plot(
    enemy_dms: List[Any],
    time_to_interference: float,
    time_BC: float,
    enemy_speed: float,
    uav_start_speed: float,
    uav_max_speed: float,
    uav_end_speed: float,
    acc_max: float,
    bearing_deg: float,
    high_distance: float,
    uav_center: Optional[Any] = None,
    enemy_center: Optional[Any] = None,
    interference_mode: str = "normal",  # 干扰模式 "normal"(默认), "least"(最少), "limit"(限制)
    max_c_points: Optional[int] = None,  # C点数量上限（limit模式）
    coverage_neighbors: int = 4,  # 覆盖邻居数（least/limit模式）
    custom_radius: Optional[float] = None,  #自定义覆盖半径（limit模式，单位：米）
    mode: str = "col",  # normal模式下的行列选择："row"或"col"
) -> Dict[str, Any]:
    """
    干扰算法主函数，支持三种模式：

    1. normal模式：隔行/列选择（约50%覆盖）
       - 参数：mode="row"或"col"

    2. least模式：网格支配集算法（约25-30%覆盖，100%覆盖率）
       - 参数：coverage_neighbors（控制覆盖范围）

    3. limit模式：限制数量最大覆盖（指定C点数量上限）
       - 参数：max_c_points, coverage_neighbors, custom_radius
       - 智能判断：如果提供custom_radius或max_c_points，自动切换到limit模式
    """

    lat_u, lon_u, alt_u = dms_to_degree(uav_center)
    uav_center = [lat_u, lon_u, alt_u]

    # 先判断敌群中心是否给出，如果没有，则计算中心
    enemy_center = judge_enemy_center(enemy_center, enemy_dms)

    # 建立全局坐标系
    converter, _, _ = build_global_converter(uav_center, enemy_center)

    # 计算敌机行数和列数
    row, line = calculate_formation(enemy_dms, threshold_distance=1000)

    # 假设time秒后到达C点，此时敌群的位置
    interfence_enemy_dms = calculate_position(enemy_dms, enemy_speed, bearing_deg, time_to_interference, converter)

    # 智能模式判断：如果提供了custom_radius或max_c_points，自动切换到limit模式
    if custom_radius is not None or max_c_points is not None:
        if interference_mode == "normal":
            interference_mode = "limit"
            print(f"检测到custom_radius或max_c_points参数，自动切换到limit模式")

    # 根据模式调用不同的函数
    if interference_mode == "least":
        print(f"\n使用模式: LEAST（最少C点，100%覆盖）")
        c_points, c_indices = compute_C_points_least(
            interfence_enemy_dms,
            n_rows=row,
            n_cols=line,
            converter=converter,
            high_distance=high_distance,
            coverage_neighbors=coverage_neighbors,
        )

    elif interference_mode == "limit":
        print(f"\n使用模式: LIMIT（限制数量，最大覆盖）")
        if max_c_points is None:
            # 默认值：总敌机数的1/3
            max_c_points = len(interfence_enemy_dms) // 3
            print(f"未指定max_c_points，使用默认值: {max_c_points} (敌机数的1/3)")

        c_points, c_indices = compute_C_points_limit(
            interfence_enemy_dms,
            n_rows=row,
            n_cols=line,
            converter=converter,
            high_distance=high_distance,
            max_c_points=max_c_points,
            coverage_neighbors=coverage_neighbors,
            custom_radius=custom_radius,
        )

    else:  # normal模式
        print(f"\n使用模式: NORMAL（隔{mode}选择）")
        c_points, c_indices = compute_C_points(
            interfence_enemy_dms,
            n_rows=row,
            n_cols=line,
            high_distance=high_distance,
            mode=mode,
        )

    print(f"\n最终选中的C点数量：{len(c_points)}")

    # 可视化
    bearing_reverse = (bearing_deg + 180) % 360

    # # 根据C点倒推B点
    # b_points = compute_B_point(
    #     uav_dms,
    #     c_points,
    #     bearing_reverse,
    #     uav_start_speed,
    #     uav_end_speed,
    #     time_to_interference,
    #     time_BC,
    #     acc_max,
    #     uav_max_speed,
    #     converter,
    # )

    lons, lats, alts = [], [], []
    for dms in c_points:
        lon, lat, alt = dms_to_degree(dms)
        lons.append(lon)
        lats.append(lat)
        alts.append(alt)

    enemy_lons, enemy_lats, enemy_alts = [], [], []
    for dms in interfence_enemy_dms:
        lon, lat, alt = GC.decimal_dms_to_degrees(dms)
        enemy_lons.append(lon)
        enemy_lats.append(lat)
        enemy_alts.append(alt)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(enemy_lons, enemy_lats, enemy_alts,
               s=5, c='gray', alpha=0.6, label='Enemies')

    ax.scatter(lons, lats, alts,
               s=5, c='red', marker='^', label='C points')

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(f"Enemy Formation & C Points ({interference_mode.upper()} mode)")
    ax.legend()

    plt.show()

    return {
        "c_points": c_points,
        "c_indices": c_indices,
        "mode": interference_mode,
        "c_count": len(c_points),
    }


# ==================================================== 输入输出 =========================================================
# 获取输入接口信息
def api_main(config: Dict[str, Any]) -> dict:

    if "enemy_dms" in config:
        enemy_dms = config["enemy_dms"]
    else:
        raise ValueError("config 中缺少enemy_dms字段")


    # 获取输入参数
    uav_center = config.get("uav_center", None)
    enemy_center = config.get("enemy_center", None)
    enemy_speed = config.get("enemy_speed")
    bearing_deg = config.get("bearing")
    time_to_interference = config.get("time_to_interference")
    uav_start_speed = config.get("uav_start_speed")
    uav_max_speed = config.get("uav_max_speed")
    acc_max = config.get("acc_max")
    high_distance = config.get("high_distance")

    # 新增：干扰模式相关参数
    interference_mode = config.get("interference_mode", "normal")
    max_c_points = config.get("max_c_points", None)
    coverage_neighbors = config.get("coverage_neighbors", 4)
    custom_radius = config.get("custom_radius", None)
    mode = config.get("mode", "col")

    # 设定的参数
    time_BC = 2
    uav_end_speed = config.get("enemy_speed")

    # 直接调用最核心函数，得到C点
    result = verify_coverage_and_plot(
        enemy_dms,
        time_to_interference,
        time_BC,
        enemy_speed,
        uav_start_speed,
        uav_max_speed,
        uav_end_speed,
        acc_max,
        bearing_deg,
        high_distance,
        uav_center=uav_center,
        enemy_center=enemy_center,
        interference_mode=interference_mode,
        max_c_points=max_c_points,
        coverage_neighbors=coverage_neighbors,
        custom_radius=custom_radius,
        mode=mode,
    )

    return {
        "C_points": result["c_points"],
        "mode": result["mode"],
        "c_count": result["c_count"],
    }

def run_port(config: dict | None = None) -> dict:


    # 如果外面没传 config，就用 test_execute_main 里的那一份
    if config is None:
        config = test_config

    if "enemy_dms" in config:
        enemy_dms = config["enemy_dms"]


    # 获取输入参数
    uav_center = config.get("uav_center", None)
    enemy_center = config.get("enemy_center", None)
    enemy_speed = config.get("enemy_speed")
    bearing_deg = config.get("bearing")
    time_to_interference = config.get("time_to_interference")
    uav_start_speed = config.get("uav_start_speed")
    uav_max_speed = config.get("uav_max_speed")
    acc_max = config.get("acc_max")
    high_distance = config.get("high_distance")

    # 新增：干扰模式相关参数
    interference_mode = config.get("interference_mode", "normal")
    max_c_points = config.get("max_c_points", None)
    coverage_neighbors = config.get("coverage_neighbors", 4)
    custom_radius = config.get("custom_radius", None)
    mode = config.get("mode", "col")

    # 设定的参数
    time_BC = 2
    uav_end_speed = config.get("enemy_speed")

    # 直接调用最核心函数，得到C点
    result = verify_coverage_and_plot(
        enemy_dms,
        time_to_interference,
        time_BC,
        enemy_speed,
        uav_start_speed,
        uav_max_speed,
        uav_end_speed,
        acc_max,
        bearing_deg,
        high_distance,
        uav_center=uav_center,
        enemy_center=enemy_center,
        interference_mode=interference_mode,
        max_c_points=max_c_points,
        coverage_neighbors=coverage_neighbors,
        custom_radius=custom_radius,
        mode=mode,
    )

    return {
        "C_points": result["c_points"],
        "mode": result["mode"],
        "c_count": result["c_count"],
    }


if __name__ == "__main__":
    run_port()

