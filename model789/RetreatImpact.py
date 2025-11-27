import json
import os
import math
import numpy as np
from collections import defaultdict, Counter

# 默认始终匀速向西平飞，角度为无人机飞行时朝向的平面即向前下飞是270，左下为315，右下为225
# ==============================
# 🔧 配置参数
# ==============================
DIVE_DURATION_SECONDS = 3.0  # 俯冲时间
LEVEL_DURATION_SECONDS = 10.0  # 平飞时间（用于计算平飞终点位置）
EARTH_RADIUS_METERS = 6378137.0  # 赤道半径（WGS84）

# ===== 速度与俯冲角映射 =====
SPEED_TO_PITCH = {
    150: -60.0,
    120: -30.0,
    100: 0.0
}
INITIAL_SPEED = 100
EVAC_SPEED_STEEP = 150
EVAC_SPEED_SHALLOW = 120
HEADING_WEST = 270
HEADING_EAST = 90
HEADING_SOUTH = 180
HEADING_SW = 225
HEADING_NW = 315

# 脱离意图类型
INTENT_FORWARD_LEFT = "forward_left"
INTENT_FORWARD_RIGHT = "forward_right"
INTENT_LEVEL = "level"  # 平飞

SUPPORTED_FORMATIONS = {
    "formation_1_25x14",
    "formation_2_5x10x7_front_high",
    "formation_2_5x10x7_front_low",
    "formation_3_1x50x7_front_high",
    "formation_3_1x50x7_front_low"
}

# ===== 工具函数：根据航向、速度、时间计算新经纬度（简化模型）=====
def calculate_new_lon(lat_deg, lon_deg, heading_deg, speed_mps, duration_sec):
    """ 简化：仅处理东西向位移（heading=270 或 90 时最准），其他方向近似。 返回 (new_lat, new_lon)，此处 lat 不变，仅调整 lon。 """
    distance = speed_mps * duration_sec  # 米
    if heading_deg == 270:
        sign = -1  # 向西，经度减小
    elif heading_deg == 90:
        sign = 1  # 向东
    else:
        # 投影到东西方向
        east_component = math.cos(math.radians(heading_deg - 90))  # 90°是正东
        sign = 1 if east_component > 0 else -1
        distance = abs(distance * east_component)
    lat_rad = math.radians(lat_deg)
    meters_per_degree_lon = 111320 * math.cos(lat_rad)  # 近似
    delta_lon = sign * distance / meters_per_degree_lon
    return lat_deg, lon_deg + delta_lon

def circular_mean(angles_deg):
    """计算角度列表（度）的圆形平均"""
    angles_rad = np.radians(angles_deg)
    mean_cos = np.mean(np.cos(angles_rad))
    mean_sin = np.mean(np.sin(angles_rad))
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = np.degrees(mean_angle_rad) % 360
    return mean_angle_deg

def project_onto_heading(lat, lon, ref_lat, ref_lon, heading_deg):
    """ 将 (lat, lon) 投影到以 (ref_lat, ref_lon) 为原点、heading_deg 为正方向的轴上。 返回沿该轴的标量距离（单位：米，近似）。 """
    # 将经纬度差转为米（局部平面近似）
    dlat = (lat - ref_lat) * 111000  # 1 deg lat ≈ 111 km
    dlon = (lon - ref_lon) * 111000 * np.cos(np.radians(ref_lat))  # 经度缩放
    # 航向角：0=北，90=东 → 转为数学角（x=east, y=north）
    # 主航向的单位向量（x=east, y=north）
    theta = np.radians(heading_deg)
    u_east = np.sin(theta)  # 东分量
    u_north = np.cos(theta)  # 北分量
    # 投影 = dot([dlon, dlat], [u_east, u_north])
    projection = dlon * u_east + dlat * u_north
    return projection

# ===== 新增：方向敏感的轴值计算（用于策略三）=====
def get_along_axis_value(lat, lon, ref_lat, ref_lon, heading_deg):
    """返回沿主航向的标量值（越大越靠前），对正交航向使用精确坐标比较"""
    hdg = heading_deg % 360
    if hdg == 0:      # 北
        return lat
    elif hdg == 180:  # 南
        return -lat
    elif hdg == 90:   # 东
        return lon
    elif hdg == 270:  # 西
        return -lon
    else:
        return project_onto_heading(lat, lon, ref_lat, ref_lon, heading_deg)

def get_cross_axis_value(lat, lon, ref_lat, ref_lon, heading_deg):
    """返回垂直于主航向的左侧轴值（越大越靠左）"""
    left_hdg = (heading_deg - 90) % 360
    return get_along_axis_value(lat, lon, ref_lat, ref_lon, left_hdg)

# ===== 策略函数（扩展返回经纬度）=====

def apply_strategy_1(drones):
    if not drones:
        return []
    # --- Step A: 获取所有初始航向（从顶层字段！）---
    headings = []
    for d in drones:
        hdg = d.get("Heading", HEADING_WEST)
        headings.append(hdg)
    reference_heading = circular_mean(headings)
    # --- Step B: 计算质心 ---
    lats = np.array([d["Location"][0] for d in drones])
    lons = np.array([d["Location"][1] for d in drones])
    ref_lat = np.mean(lats)
    ref_lon = np.mean(lons)
    # --- Step C: 投影到主航向轴 ---
    projections = []
    for d in drones:
        lat, lon = d["Location"][0], d["Location"][1]
        proj = project_onto_heading(lat, lon, ref_lat, ref_lon, reference_heading)
        projections.append((proj, d))
    projections.sort(key=lambda x: x[0], reverse=True)  # 大者在前
    n = len(projections)
    mid_idx = n // 2
    front_half = [item[1] for item in projections[:mid_idx]]
    rear_half = [item[1] for item in projections[mid_idx:]]
    results = []
    # 前一半：forward_down
    for d in front_half:
        lat, lon, alt = d["Location"]
        hdg = d.get("Heading", reference_heading)
        speed = d.get("Speed", INITIAL_SPEED)
        results.append((d["PlatformName"], lat, lon, alt, hdg, speed, "forward_down"))
    # 后一半：分左右
    if rear_half:
        left_axis = (reference_heading - 90) % 360
        left_projections = []
        for d in rear_half:
            lat, lon = d["Location"][0], d["Location"][1]
            proj_left = project_onto_heading(lat, lon, ref_lat, ref_lon, left_axis)
            left_projections.append((proj_left, d))
        left_projections.sort(key=lambda x: x[0], reverse=True)
        rear_n = len(left_projections)
        rear_mid = rear_n // 2
        for i, (_, d) in enumerate(left_projections):
            lat, lon, alt = d["Location"]
            hdg = d.get("Heading", reference_heading)
            speed = d.get("Speed", INITIAL_SPEED)
            intent = "forward_left" if i < rear_mid else "forward_right"
            results.append((d["PlatformName"], lat, lon, alt, hdg, speed, intent))
    return results

def apply_strategy_2(drones):
    if not drones:
        return []
    H = drones[0].get("Heading", HEADING_EAST)
    left_axis = (H - 90) % 360
    lats = np.array([d["Location"][0] for d in drones])
    lons = np.array([d["Location"][1] for d in drones])
    ref_lat = np.mean(lats)
    ref_lon = np.mean(lons)
    projections = []
    for d in drones:
        lat, lon = d["Location"][0], d["Location"][1]
        proj = project_onto_heading(lat, lon, ref_lat, ref_lon, left_axis)
        projections.append((proj, d))
    projections.sort(key=lambda x: x[0], reverse=True)
    n = len(projections)
    mid = n // 2
    results = []
    for i, (_, d) in enumerate(projections):
        lat, lon, alt = d["Location"]
        heading = d.get("Heading", H)
        speed = d.get("Speed", INITIAL_SPEED)
        intent = "forward_left" if i < mid else "forward_right"
        results.append((d["PlatformName"], lat, lon, alt, heading, speed, intent))
    return results

# ===== 策略三：使用精确轴值避免浮点误差 =====
def apply_strategy_3(drones):
    """策略3修正版：按实际高度分层，仅当层高严格递增时应用撤离逻辑"""
    if not drones:
        return []

    # === 1. 按高度分组（精确值）===
    from collections import defaultdict
    alt_groups = defaultdict(list)
    for d in drones:
        alt = d["Location"][2]
        alt_groups[alt].append(d)

    # === 2. 按高度升序排列组 ===
    sorted_alts = sorted(alt_groups.keys())
    layers = [alt_groups[alt] for alt in sorted_alts]

    # === 3. 检查是否严格递增（至少两层才需检查）===
    if len(layers) > 1:
        for i in range(len(layers) - 1):
            if sorted_alts[i] >= sorted_alts[i + 1]:
                # 不满足递增，fallback：全部 forward_down
                results = []
                GLOBAL_HEADING = Counter([d.get("Heading", HEADING_EAST) for d in drones]).most_common(1)[0][0]
                for d in drones:
                    lat, lon, alt = d["Location"]
                    H = d.get("Heading", GLOBAL_HEADING)
                    V = d.get("Speed", INITIAL_SPEED)
                    results.append((d["PlatformName"], lat, lon, alt, H, V, "forward_down"))
                return results

    # === 4. 获取全局主航向（众数）===
    headings = [d.get("Heading", HEADING_EAST) for d in drones]
    GLOBAL_HEADING = Counter(headings).most_common(1)[0][0]

    # === 5. 对每层独立处理 ===
    results = []
    for layer in layers:
        n = len(layer)
        if n == 1:
            d = layer[0]
            lat, lon, alt = d["Location"]
            H = d.get("Heading", GLOBAL_HEADING)
            V = d.get("Speed", INITIAL_SPEED)
            results.append((d["PlatformName"], lat, lon, alt, H, V, "forward_down"))
            continue

        # 计算本层质心（用于投影）
        lats = np.array([d["Location"][0] for d in layer])
        lons = np.array([d["Location"][1] for d in layer])
        ref_lat = float(np.mean(lats))
        ref_lon = float(np.mean(lons))

        # 沿主航向排序（前 -> 后）
        along_list = []
        for d in layer:
            lat, lon = d["Location"][0], d["Location"][1]
            val = get_along_axis_value(lat, lon, ref_lat, ref_lon, GLOBAL_HEADING)
            along_list.append((val, d))
        along_list.sort(key=lambda x: x[0], reverse=True)  # 大者在前

        mid_idx = n // 2
        front_half = [item[1] for item in along_list[:mid_idx]]
        rear_half = [item[1] for item in along_list[mid_idx:]]

        # 前一半：forward_down
        for d in front_half:
            lat, lon, alt = d["Location"]
            H = d.get("Heading", GLOBAL_HEADING)
            V = d.get("Speed", INITIAL_SPEED)
            results.append((d["PlatformName"], lat, lon, alt, H, V, "forward_down"))

        # 后一半：分左右
        if rear_half:
            cross_list = []
            for d in rear_half:
                lat, lon = d["Location"][0], d["Location"][1]
                val = get_cross_axis_value(lat, lon, ref_lat, ref_lon, GLOBAL_HEADING)
                cross_list.append((val, d))
            cross_list.sort(key=lambda x: x[0], reverse=True)  # 大者更靠左
            rear_n = len(cross_list)
            rear_mid = rear_n // 2
            for i, (_, d) in enumerate(cross_list):
                lat, lon, alt = d["Location"]
                H = d.get("Heading", GLOBAL_HEADING)
                V = d.get("Speed", INITIAL_SPEED)
                intent = "forward_left" if i < rear_mid else "forward_right"
                results.append((d["PlatformName"], lat, lon, alt, H, V, intent))

    return results

def apply_strategy_4(drones):
    """ 策略4：按纬度（南北）分左右。 - 北侧一半执行 forward_left - 南侧一半执行 forward_right - 航向从输入数据中动态获取。 """
    if not drones:
        return []
    sorted_drones = sorted(drones, key=lambda x: x["Location"][0], reverse=True)
    n = len(sorted_drones)
    mid = n // 2
    results = []
    for i, d in enumerate(sorted_drones):
        lat, lon, alt = d["Location"]
        H = d.get("Heading", HEADING_EAST)
        V = d.get("Speed", INITIAL_SPEED)
        intent = INTENT_FORWARD_LEFT if i < mid else INTENT_FORWARD_RIGHT
        results.append((d["PlatformName"], lat, lon, alt, H, V, intent))
    return results

def apply_strategy_5(drones):
    """ 策略5：所有无人机继续平飞。 - 保持原有航向和速度。 - 意图为 level。 """
    if not drones:
        return []
    results = []
    for d in drones:
        lat, lon, alt = d["Location"]
        H = d.get("Heading", HEADING_EAST)
        V = d.get("Speed", INITIAL_SPEED)
        results.append((d["PlatformName"], lat, lon, alt, H, V, INTENT_LEVEL))
    return results

# 假设这里有一个辅助函数 calculate_new_position 来根据速度、航向等参数计算新位置
def calculate_new_position(lat, lon, heading, speed_mps, duration_sec):
    R = 6378137.0  # Earth radius in meters
    distance = speed_mps * duration_sec  # meters
    if distance == 0:
        return lat, lon
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    heading_rad = math.radians(heading)
    angular_distance = distance / R
    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(angular_distance) +
        math.cos(lat_rad) * math.sin(angular_distance) * math.cos(heading_rad)
    )
    new_lon_rad = lon_rad + math.atan2(
        math.sin(heading_rad) * math.sin(angular_distance) * math.cos(lat_rad),
        math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat_rad)
    )
    new_lat = math.degrees(new_lat_rad)
    new_lon = (math.degrees(new_lon_rad) + 540) % 360 - 180
    return new_lat, new_lon

def apply_evacuation_strategy(drones, formation_name):
    if formation_name == "formation_1_25x14":
        return apply_strategy_1(drones)
    elif formation_name == "formation_2_5x10x7_front_high":
        return apply_strategy_2(drones)
    elif formation_name == "formation_2_5x10x7_front_low":
        return apply_strategy_3(drones)  # 使用优化后的策略三
    elif formation_name == "formation_3_1x50x7_front_high":
        return apply_strategy_4(drones)
    elif formation_name == "formation_3_1x50x7_front_low":
        return apply_strategy_5(drones)
    else:
        raise ValueError(f"不支持的阵型: {formation_name}")

# --- DMS 转十进制度 ---
def _dms_to_dd(dms_str):
    dms_str = dms_str.strip()
    if not dms_str:
        raise ValueError("空的 DMS 字符串")
    direction = dms_str[-1].upper()
    parts = dms_str[:-1].split(':')
    deg = float(parts[0])
    min_ = float(parts[1]) if len(parts) > 1 else 0.0
    sec = float(parts[2]) if len(parts) > 2 else 0.0
    dd = deg + min_ / 60.0 + sec / 3600.0
    return -dd if direction in ('S', 'W') else dd

# --- 十进制度转 DMS 字符串 ---
def _dd_to_dms(dd, is_latitude=True):
    if is_latitude:
        if dd >= 0:
            hemi = 'N'
        else:
            hemi = 'S'
    else:
        if dd >= 0:
            hemi = 'E'
        else:
            hemi = 'W'
    dd = abs(dd)
    deg = int(dd)
    minutes_full = (dd - deg) * 60
    min_ = int(minutes_full)
    sec = (minutes_full - min_) * 60
    # 保留两位小数，与输入风格一致
    sec_str = f"{sec:.2f}".rstrip('0').rstrip('.') if '.' in f"{sec:.2f}" else f"{sec:.2f}"
    # 确保至少显示两位小数（如 10.10）
    if '.' not in sec_str:
        sec_str += ".00"
    elif len(sec_str.split('.')[1]) == 1:
        sec_str += "0"
    return f"{deg}:{min_}:{sec_str}{hemi}"


# ======== 标准化 API 入口：整合原 main 逻辑 =========
def api_main(config: dict) -> dict:
    """
    专用于「算法2-转弯」场景的 API。
    输入：算法2 JSON 格式，必须包含 'formation' 字段
    输出：航路点 Lat/Lon 为 DMS 字符串，如 "29:46:10.11N"
    """
    # --- 解析输入 ---
    try:
        uav_speed = config["uav_speed"]
        uav_direction = config["uav_direction"]
        formation = config.get("formation")  # ← 新增：从 config 读取阵型
        uav_positions = config["uav_position"]
    except KeyError as e:
        raise ValueError(f"缺少必要字段: {e}")

    # 验证 formation
    if formation is None:
        raise ValueError("缺少 'formation' 字段，请指定阵型（如 'formation_1_25x14'）")
    if formation not in SUPPORTED_FORMATIONS:
        raise ValueError(f"未知阵型: {formation}。支持的阵型: {sorted(SUPPORTED_FORMATIONS)}")

    if not isinstance(uav_positions, list):
        raise ValueError("'uav_position' 必须是列表")

    # 构建平台列表
    platforms = []
    for i, pos in enumerate(uav_positions):
        if len(pos) != 3:
            raise ValueError(f"位置项长度错误（应为3）: {pos}")
        lon_dms, lat_dms, alt_raw = pos
        try:
            lon = _dms_to_dd(lon_dms)
            lat = _dms_to_dd(lat_dms)
            alt = float(alt_raw)
        except Exception as e:
            raise ValueError(f"解析第{i+1}架无人机位置失败 ({pos}): {e}")
        platforms.append({
            "PlatformName": f"uav_{i+1:03d}",
            "Tactic": "撞击",
            "Formation": formation,  # ← 使用传入的阵型
            "Location": [lat, lon, alt],
            "Heading": uav_direction,
            "Speed": uav_speed
        })

    if not platforms:
        return {"MsgType": "UpdateEntityRoute", "Platforms": []}

    # --- 调用策略（使用 config 中的 formation）---
    filtered = [p for p in platforms if p.get("Tactic") == "撞击"]
    try:
        intent_tuples = apply_evacuation_strategy(filtered, formation)  # ← 关键：传入动态 formation
    except Exception as e:
        raise RuntimeError(f"策略执行失败: {e}")

    if not intent_tuples:
        return {"MsgType": "UpdateEntityRoute", "Platforms": []}

    sorted_intents = sorted(intent_tuples, key=lambda x: x[0])

    # --- 生成航路（Lat/Lon 转为 DMS 字符串）---
    route_platforms = []
    for idx, (orig_name, lat0, lon0, alt0, H, V0, intent) in enumerate(sorted_intents, start=1):
        new_name = f"ucav_{idx:03d}"
        if intent == "level":
            wp1 = {
                "Lat": _dd_to_dms(lat0, is_latitude=True),
                "Lon": _dd_to_dms(lon0, is_latitude=False),
                "Alt": alt0,
                "Speed": V0,
                "Heading": H
            }
            lat1, lon1 = calculate_new_position(lat0, lon0, H, V0, DIVE_DURATION_SECONDS)
            lat2, lon2 = calculate_new_position(lat1, lon1, H, V0, LEVEL_DURATION_SECONDS)
            wp2 = {
                "Lat": _dd_to_dms(lat1, is_latitude=True),
                "Lon": _dd_to_dms(lon1, is_latitude=False),
                "Alt": round(alt0, 1),
                "Speed": V0,
                "Heading": H
            }
            wp3 = {
                "Lat": _dd_to_dms(lat2, is_latitude=True),
                "Lon": _dd_to_dms(lon2, is_latitude=False),
                "Alt": round(alt0, 1),
                "Speed": V0,
                "Heading": H
            }
        else:
            V_dive = V0 + 20
            alt1 = max(0.0, alt0 - 300.0)
            if intent == "forward_left":
                dive_hdg = (H - 45) % 360
            elif intent == "forward_right":
                dive_hdg = (H + 45) % 360
            elif intent == "forward_down":
                dive_hdg = H
            else:
                dive_hdg = H
            wp1 = {
                "Lat": _dd_to_dms(lat0, is_latitude=True),
                "Lon": _dd_to_dms(lon0, is_latitude=False),
                "Alt": alt0,
                "Speed": V_dive,
                "Heading": dive_hdg
            }
            lat1, lon1 = calculate_new_position(lat0, lon0, dive_hdg, V_dive, DIVE_DURATION_SECONDS)
            lat2, lon2 = calculate_new_position(lat1, lon1, dive_hdg, V_dive, LEVEL_DURATION_SECONDS)
            wp2 = {
                "Lat": _dd_to_dms(lat1, is_latitude=True),
                "Lon": _dd_to_dms(lon1, is_latitude=False),
                "Alt": round(alt1, 1),
                "Speed": V_dive,
                "Heading": dive_hdg
            }
            wp3 = {
                "Lat": _dd_to_dms(lat2, is_latitude=True),
                "Lon": _dd_to_dms(lon2, is_latitude=False),
                "Alt": round(alt1, 1),
                "Speed": V_dive,
                "Heading": dive_hdg
            }
        route_platforms.append({
            "PlatformName": new_name,
            "Waypoints": [wp1, wp2, wp3]
        })

    return {"MsgType": "UpdateEntityRoute", "Platforms": route_platforms}

# ==============================
# ▶️ 启动
# ==============================
# if __name__ == "__main__":
#     run_port()


