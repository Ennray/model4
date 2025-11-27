import json
import os
import math

# ==============================
#  常量配置
# ==============================
EARTH_RADIUS_METERS = 6378137.0
FLIGHT_DURATION_SECONDS = 10.0
DEFAULT_SPEED = 150.0  # m/s

# 阵型 → 策略描述
FORMATION_TO_STRATEGY = {
    "formation_1_25x14": "向前上方脱离",
    "formation_2_5x10x7_front_high": "向前左、前右加速脱离",
    "formation_2_5x10x7_front_low": "向前上方脱离",
    "formation_3_1x50x7_front_low": "继续平飞",
    "formation_3_1x50x7_front_high": "向左下、右下加速脱离"
}

#  精细化策略参数（必须在 process_platforms 之前定义！）
EVADE_STRATEGIES = {
    "向前上方脱离": {"alt_change": 300.0, "speed_delta": 0.0, "heading_mode": "keep"},
    "向前左、前右加速脱离": {"alt_change": 0.0, "speed_delta": 20.0, "heading_mode": "left_right"},
    "继续平飞": {"alt_change": 0.0, "speed_delta": 0.0, "heading_mode": "keep"},
    "向左下、右下加速脱离": {"alt_change": -300.0, "speed_delta": 20.0, "heading_mode": "left_right"}
}

# ==============================
#  工具函数
# ==============================
def calculate_new_position(lat, lon, heading_deg, speed_mps, duration_sec):
    # 将航向转换为弧度（0°=北，顺时针）
    heading_rad = math.radians(heading_deg)
    # 北向分量（dy），东向分量（dx）
    dy = speed_mps * duration_sec * math.cos(heading_rad)  # 北为正
    dx = speed_mps * duration_sec * math.sin(heading_rad)  # 东为正

    # 纬度变化（1米 ≈ 1/111320 度）
    lat_new = lat + (dy / 111320.0)
    # 经度变化（需除以 cos(lat)）
    lon_new = lon + (dx / (111320.0 * math.cos(math.radians(lat))))

    return lat_new, lon_new

def dms_to_dd(dms_str):
    """将 DMS 字符串（如 '120:38:16.97E'）转换为十进制度"""
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


def dd_to_dms(dd, is_latitude=True):
    """将十进制度转换为 DMS 字符串，如 '29:46:10.11N'"""
    if is_latitude:
        hemi = 'N' if dd >= 0 else 'S'
    else:
        hemi = 'E' if dd >= 0 else 'W'
    dd = abs(dd)
    deg = int(dd)
    minutes_full = (dd - deg) * 60
    min_ = int(minutes_full)
    sec = (minutes_full - min_) * 60
    # 格式化秒为两位小数（补零）
    sec_str = f"{sec:.2f}"
    return f"{deg}:{min_}:{sec_str}{hemi}"

# ==============================
#  核心处理函数
# ==============================
def process_platforms(platforms, evasion_strategy):
    processed = []
    duration = FLIGHT_DURATION_SECONDS
    DEFAULT_SPEED_FALLBACK = 150.0  # 仅用于缺失 Speed 时的回退

    strategy_config = EVADE_STRATEGIES.get(evasion_strategy)
    if not strategy_config:
        raise ValueError(f"未知策略: {evasion_strategy}")

    for idx, p in enumerate(platforms):
        name = p["PlatformName"]
        loc = p["Location"]
        lat0, lon0, alt0 = loc[0], loc[1], loc[2]

        # 从输入获取初始状态
        initial_heading = float(p.get("Heading", 0.0))
        initial_speed = float(p.get("Speed", DEFAULT_SPEED_FALLBACK))

        # 确定新航向
        if strategy_config["heading_mode"] == "keep":
            new_heading = initial_heading
        elif strategy_config["heading_mode"] == "left_right":
            if idx % 2 == 0:
                new_heading = (initial_heading - 45.0) % 360.0
            else:
                new_heading = (initial_heading + 45.0) % 360.0
        else:
            new_heading = initial_heading

        # 计算新速度和高度
        new_speed = initial_speed + strategy_config["speed_delta"]
        new_alt = alt0 + strategy_config["alt_change"]
        # 计算新位置
        lat1, lon1 = calculate_new_position(lat0, lon0, new_heading, new_speed, duration)
        #  第一航点：原始状态（含原始 Heading 和 Speed）
        wp1 = {
            "Lat": lat0,
            "Lon": lon0,
            "Alt": alt0,
            "Speed": round(new_speed, 1),
            "Heading": round(new_heading, 1)
        }


        #  第二航点：新状态
        wp2 = {
            "Lat": round(lat1, 12),
            "Lon": round(lon1, 12),
            "Alt": round(new_alt, 1),
            "Speed": round(new_speed, 1),
            "Heading": round(new_heading, 1)
        }

        processed.append({
            "PlatformName": name,
            "Waypoints": [wp1, wp2]
        })

    return processed


# ======== 标准化 API 入口：整合原 main 逻辑 =========
def api_main(config: dict) -> dict:
    """
    输入必须为算法2格式：
    {
      "uav_speed": 120,
      "uav_direction": 90,
      "formation": "formation_2_5x10x7_front_low",  # ← 新增字段！
      "uav_position": [ ["120:38:16.97E", "29:46:10.11N", "5000"], ... ]
    }
    输出航路点 Lat/Lon 为 DMS 字符串。
    """
    # --- 解析输入 ---
    try:
        uav_speed = config["uav_speed"]
        uav_direction = config["uav_direction"]
        formation = config.get("formation")  # ← 新增：读取阵型
        uav_positions = config["uav_position"]
    except KeyError as e:
        raise ValueError(f"缺少必要字段: {e}")

    if formation is None:
        raise ValueError("缺少 'formation' 字段，请指定阵型（如 'formation_2_5x10x7_front_low'）")

    if formation not in FORMATION_TO_STRATEGY:
        raise ValueError(f"未知阵型: {formation}。支持的阵型: {list(FORMATION_TO_STRATEGY.keys())}")

    platforms = []
    for i, pos in enumerate(uav_positions):
        if len(pos) != 3:
            raise ValueError(f"位置项长度错误: {pos}")
        lon_dms, lat_dms, alt_raw = pos
        try:
            lon = dms_to_dd(lon_dms)
            lat = dms_to_dd(lat_dms)
            alt = float(alt_raw)
        except Exception as e:
            raise ValueError(f"解析第{i+1}架无人机位置失败 ({pos}): {e}")
        platforms.append({
            "PlatformName": f"uav_{i+1:03d}",
            "Tactic": "投弹",
            "Formation": formation,  # ← 使用传入的阵型
            "Location": [lat, lon, alt],
            "Heading": uav_direction,
            "Speed": uav_speed
        })

    if not platforms:
        return {"MsgType": "UpdateEntityRoute", "Platforms": []}

    # --- 应用策略（根据传入的 formation）---
    strategy_desc = FORMATION_TO_STRATEGY[formation]  # ← 动态获取策略
    filtered = [p for p in platforms if p.get("Tactic") == "投弹"]
    if not filtered:
        return {"MsgType": "UpdateEntityRoute", "Platforms": []}

    # 使用已有的 process_platforms 生成航路点（内部是十进制度）
    processed_platforms = process_platforms(filtered, strategy_desc)

    # 转换 Lat/Lon 为 DMS 字符串
    route_platforms = []
    for idx, item in enumerate(processed_platforms, start=1):
        name = f"ucav_{idx:03d}"
        waypoints = []
        for wp in item["Waypoints"]:
            wp_dms = {
                "Lat": dd_to_dms(wp["Lat"], is_latitude=True),
                "Lon": dd_to_dms(wp["Lon"], is_latitude=False),
                "Alt": wp["Alt"],
                "Speed": wp["Speed"],
                "Heading": wp["Heading"]
            }
            waypoints.append(wp_dms)
        route_platforms.append({
            "PlatformName": name,
            "Waypoints": waypoints
        })

    return {"MsgType": "UpdateEntityRoute", "Platforms": route_platforms}

# ==============================
# ▶️ 启动
# ==============================
# if __name__ == "__main__":
#     run_port()


