import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# 🖥️ Mac系统字体设置
# ==============================
def setup_mac_fonts():
    """设置Mac系统下的字体"""
    try:
        plt.rcParams['font.sans-serif'] = [
            'Arial Unicode MS',  # Mac常用Unicode字体
            'Helvetica',         # Mac系统字体
            'DejaVu Sans',       # 跨平台字体
            'Verdana'            # 备选字体
        ]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("✅ Font setup completed")
    except Exception as e:
        print(f"⚠️ Font setup failed: {e}")

# 初始化字体设置
setup_mac_fonts()

# ==============================
#  常量配置
# ==============================
EARTH_RADIUS_METERS = 6378137.0
FLIGHT_DURATION_SECONDS = 10.0
DEFAULT_SPEED = 150.0  # m/s

# 阵型 → 策略描述
FORMATION_TO_STRATEGY = {
    "formation_1_25x14": "Forward upward escape",
    "formation_2_5x10x7_front_high": "Forward left/right acceleration escape",
    "formation_2_5x10x7_front_low": "Forward upward escape",
    "formation_3_1x50x7_front_low": "Continue level flight",
    "formation_3_1x50x7_front_high": "Left/right downward acceleration escape"
}

#  精细化策略参数（必须在 process_platforms 之前定义！）
EVADE_STRATEGIES = {
    "Forward upward escape": {"alt_change": 300.0, "speed_delta": 0.0, "heading_mode": "keep"},
    "Forward left/right acceleration escape": {"alt_change": 0.0, "speed_delta": 20.0, "heading_mode": "left_right"},
    "Continue level flight": {"alt_change": 0.0, "speed_delta": 0.0, "heading_mode": "keep"},
    "Left/right downward acceleration escape": {"alt_change": -300.0, "speed_delta": 20.0, "heading_mode": "left_right"}
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
# 🎯 可视化模块
# ==============================

class UAVBombVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_2d = self.fig.add_subplot(122)
        self.fig.suptitle('UAV Bombing Route Visualization')
        
    def plot_initial_positions(self, drones, formation_name):
        """绘制初始位置"""
        self.ax_3d.clear()
        self.ax_2d.clear()
        
        # 提取位置数据
        lats = [d["Location"][0] for d in drones]
        lons = [d["Location"][1] for d in drones]
        alts = [d["Location"][2] for d in drones]
        names = [d["PlatformName"] for d in drones]
        
        # 3D 散点图
        scatter_3d = self.ax_3d.scatter(lons, lats, alts, c=alts, cmap='viridis', s=50)
        for i, name in enumerate(names):
            self.ax_3d.text(lons[i], lats[i], alts[i], f' {name}', fontsize=8)
        
        self.ax_3d.set_xlabel('Longitude')
        self.ax_3d.set_ylabel('Latitude')
        self.ax_3d.set_zlabel('Altitude (m)')
        self.ax_3d.set_title(f'3D View - {formation_name}')
        
        # 2D 俯视图
        scatter_2d = self.ax_2d.scatter(lons, lats, c=alts, cmap='viridis', s=50)
        for i, name in enumerate(names):
            self.ax_2d.text(lons[i], lats[i], f' {name}', fontsize=8)
        
        self.ax_2d.set_xlabel('Longitude')
        self.ax_2d.set_ylabel('Latitude')
        self.ax_2d.set_title(f'2D Top View - {formation_name}')
        plt.colorbar(scatter_2d, ax=self.ax_2d, label='Altitude (m)')
        
        plt.tight_layout()
        
    def plot_bombing_routes(self, route_data, formation_name):
        """绘制投弹航线"""
        self.ax_3d.clear()
        self.ax_2d.clear()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(route_data)))
        
        # 3D 航线图
        for i, platform in enumerate(route_data):
            waypoints = platform["Waypoints"]
            lons = [dms_to_dd(wp["Lon"]) for wp in waypoints]
            lats = [dms_to_dd(wp["Lat"]) for wp in waypoints]
            alts = [wp["Alt"] for wp in waypoints]
            
            # 绘制航线
            self.ax_3d.plot(lons, lats, alts, 'o-', color=colors[i], linewidth=2, markersize=6)
            self.ax_3d.text(lons[0], lats[0], alts[0], f' {platform["PlatformName"]}', fontsize=8)
            
            # 标注策略类型
            strategy = FORMATION_TO_STRATEGY.get(formation_name, "Unknown")
            self.ax_3d.text(lons[-1], lats[-1], alts[-1], f' {strategy}', fontsize=8)
        
        self.ax_3d.set_xlabel('Longitude')
        self.ax_3d.set_ylabel('Latitude')
        self.ax_3d.set_zlabel('Altitude (m)')
        self.ax_3d.set_title('3D Bombing Routes')
        
        # 2D 航线图
        for i, platform in enumerate(route_data):
            waypoints = platform["Waypoints"]
            lons = [dms_to_dd(wp["Lon"]) for wp in waypoints]
            lats = [dms_to_dd(wp["Lat"]) for wp in waypoints]
            
            self.ax_2d.plot(lons, lats, 'o-', color=colors[i], linewidth=2, markersize=6, 
                           label=platform["PlatformName"])
            
            # 添加箭头表示航向
            for j in range(len(waypoints)-1):
                dx = lons[j+1] - lons[j]
                dy = lats[j+1] - lats[j]
                self.ax_2d.arrow(lons[j], lats[j], dx*0.7, dy*0.7, 
                               head_width=0.0005, head_length=0.001, 
                               fc=colors[i], ec=colors[i], alpha=0.6)
        
        self.ax_2d.set_xlabel('Longitude')
        self.ax_2d.set_ylabel('Latitude')
        self.ax_2d.set_title('2D Bombing Routes')
        self.ax_2d.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
    def show(self):
        """显示图形"""
        plt.show()

# ==============================
# 📊 增强的API主函数（加入可视化）
# ==============================

def api_main_with_visualization(config: dict, show_visualization: bool = True) -> dict:
    """
    增强版API主函数，支持可视化
    """
    # 调用原有逻辑
    result = api_main(config)
    
    # 可视化
    if show_visualization and result["Platforms"]:
        visualizer = UAVBombVisualizer()
        
        # 重建无人机数据用于可视化初始位置
        uav_positions = config["uav_position"]
        formation = config["formation"]
        drones = []
        for i, pos in enumerate(uav_positions):
            lon_dms, lat_dms, alt_raw = pos
            lon = dms_to_dd(lon_dms)
            lat = dms_to_dd(lat_dms)
            alt = float(alt_raw)
            drones.append({
                "PlatformName": f"uav_{i+1:03d}",
                "Location": [lat, lon, alt],
                "Heading": config["uav_direction"],
                "Speed": config["uav_speed"]
            })
        
        # 绘制初始位置
        visualizer.plot_initial_positions(drones, formation)
        
        # 绘制投弹航线
        visualizer.plot_bombing_routes(result["Platforms"], formation)
        
        visualizer.show()
    
    return result

# ==============================
# 🎯 测试函数
# ==============================

def test_with_sample_data():
    """使用样例数据测试可视化"""
    sample_config = {
        "uav_speed": 150,
        "uav_direction": 90,
        "formation": "formation_2_5x10x7_front_high",
        "uav_position": [
            ["116:23:30.00E", "39:54:20.00N", "5000"],
            ["116:23:31.00E", "39:54:21.00N", "5000"],
            ["116:23:29.00E", "39:54:19.00N", "5000"],
            ["116:23:32.00E", "39:54:22.00N", "5000"],
            ["116:23:28.00E", "39:54:18.00N", "5000"]
        ]
    }
    
    print("Generating bombing strategy and visualization...")
    result = api_main_with_visualization(sample_config, show_visualization=True)
    
    # 打印结果摘要
    print(f"Generated routes for {len(result['Platforms'])} platforms:")
    for platform in result["Platforms"]:
        print(f"  {platform['PlatformName']}: {len(platform['Waypoints'])} waypoints")
        
    return result

# ==============================
# ▶️ 主程序入口
# ==============================

if __name__ == "__main__":
    # 测试可视化功能
    test_with_sample_data()