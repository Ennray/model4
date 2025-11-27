
import matplotlib.pyplot as plt
import math
import GeodeticConverter
import plotly.graph_objects as go
from math import sin, cos, radians, degrees, atan2, pi
from geopy.distance import distance
from geopy.point import Point

from typing import List, Tuple, Any, Optional, Dict


# 构建坐标系
def builtconverter(uav_center, enemy_center):
    enemy_center_lat, enemy_center_lon, enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    uav_center_lat, uav_center_lon, uav_center_alt = GeodeticConverter.decimal_dms_to_degrees(uav_center)
    converter = GeodeticConverter.GeodeticToLocalConverter(uav_center_lat, uav_center_lon, uav_center_alt,
                                                           enemy_center_lat, enemy_center_lon, enemy_center_alt)
    return converter


# 计算转弯后的位置（有转弯度数）
def calculate_turning_position_with_bearing(lat, lon, alt, turning_radius, angle_deg, bearing_deg, direction='left'):

    # 计算圆心方向（偏移航向±90度）
    offset_bearing = (bearing_deg + (90 if direction == 'left' else -90)) % 360

    # 计算圆心点
    center_point = distance(meters=turning_radius).destination(Point(lat, lon), bearing=offset_bearing)

    # 计算终点方向：从圆心开始，按方向旋转角度
    arc_end_bearing = (offset_bearing + (angle_deg if direction == 'left' else -angle_deg)) % 360

    # 沿圆弧从圆心出发，回到圆周上
    end_point = distance(meters=turning_radius).destination(center_point, bearing=arc_end_bearing)

    # print("无人机方向，偏移", {bearing_deg}, offset_bearing)

    return end_point.latitude, end_point.longitude, alt


# 转弯之后的位置
def after_turn_position(turning_radius, uav_meet_dms, angle_deg, bearing_deg, uav_center, enemy_center, direction='right'):
    converter = builtconverter(uav_center, enemy_center)
    # 先转度数
    uav_meet_lat, uav_meet_lon, uav_meet_alt = GeodeticConverter.decimal_dms_to_degrees(uav_meet_dms)

    # 角度制转弧度制
    angle_rad = math.radians(angle_deg)

    # 计算转弯后无人机的位置
    after_turning_uav_lat, after_turning_uav_lon, after_turning_uav_alt = calculate_turning_position_with_bearing(
        uav_meet_lat, uav_meet_lon, uav_meet_alt, turning_radius, angle_rad, bearing_deg, direction)

    # 角度转local再转经纬度，得到转弯某角度后的经纬度坐标
    after_turning_uav_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(after_turning_uav_lat, after_turning_uav_lon, after_turning_uav_alt))

    # print(
    #     f"转弯后无人机的位置: 纬度 = {after_turning_uav_lat}, 经度 = {after_turning_uav_lon}, 高度 = {after_turning_uav_alt}")
    return after_turning_uav_dms



# 2.转弯路径生成算法(默认敌我方向相向)
# 2.1 转弯起始点经纬高
def turning_start_position(enemy_center, enemy_speed, uav_position, uav_speed, uav_direction, uav_center):
    converter = builtconverter(uav_center, enemy_center)
    # 敌我中心坐标转换
    enemy_center_lat, enemy_center_lon, enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    uav_center_lat, uav_center_lon, uav_center_alt = GeodeticConverter.decimal_dms_to_degrees(uav_center)

    # 中心点相遇时间（现在是按照中心计算，可以修改为根据前沿计算，首先需要确定位置信息）
    # 利用球面坐标系计算无人机与敌机之间的初始距离
    distance_between_uav_enemy = converter.calculate_spherical_distance(uav_center_lat, uav_center_lon, uav_center_alt, enemy_center_lat, enemy_center_lon, enemy_center_alt)
    meet_time =  distance_between_uav_enemy / (enemy_speed + uav_speed)
    uav_move_distance = meet_time * uav_speed #全程匀速

    # 我方中心位置更新（转弯之前）
    new_center_lat, new_center_lon, new_center_alt = converter.calculate_destination_point(
        uav_center_lat, uav_center_lon, uav_center_alt, uav_direction, uav_move_distance, 0
    )

    # 转弯之前中心位置——分量转local再转dms
    new_center_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(new_center_lat, new_center_lon, new_center_alt))

    # 我方转弯起始点位置
    turn_start_pos = []
    turn_start_pos_dms = []
    for i in range(len(uav_position)):
        each_uav_lat, each_uav_lon, each_uav_alt = GeodeticConverter.decimal_dms_to_degrees(uav_position[i])
        dist = uav_speed * meet_time  # 距离 = 速度 × 时间
        turn_start_lat, turn_start_lon, turn_start_alt = converter.calculate_destination_point(
            each_uav_lat, each_uav_lon, each_uav_alt, uav_direction, dist, 0
    )
        turn_start_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(turn_start_lat, turn_start_lon, turn_start_alt))
        turn_start_pos.append([turn_start_lat, turn_start_lon, turn_start_alt])
        turn_start_pos_dms.append(turn_start_dms)
    return turn_start_pos, turn_start_pos_dms, new_center_dms


# 米转换为经纬
def en_to_ll(lat, lon, east_m, north_m):
    p1 = distance(meters=north_m).destination(Point(lat, lon), bearing=0.0)   # 北
    p2 = distance(meters=east_m ).destination(p1, bearing=90.0) # 东
    return p2.latitude, p2.longitude


# 经纬转换为米
def ll_to_en(lat0, lon0, lat1, lon1):
    # 北向量
    north_m = distance(Point(lat0, lon0), Point(lat1, lon0)).meters
    if lat1 < lat0: north_m = -north_m
    # 东向量
    east_m  = distance(Point(lat0, lon0), Point(lat0, lon1)).meters
    if lon1 < lon0: east_m = -east_m
    return east_m, north_m


# 自身方向转换为世界方位
def body_to_world(x_forward, y_right, bearing_deg):
    ψ = radians(bearing_deg)
    east  =  y_right * cos(ψ) + x_forward * sin(ψ)
    north = -y_right * sin(ψ) + x_forward * cos(ψ)
    return east, north


# 世界方位转换为自身方向
def world_to_body(east, north, bearing_deg):
    ψ = radians(bearing_deg)
    x_forward =  east * sin(ψ) + north * cos(ψ)
    y_right   =  east * cos(ψ) - north * sin(ψ)
    return x_forward, y_right

# 东、北方向位移转换为角度
def bearing_from_en(east, north):
    degree_en = (degrees(atan2(east, north)) + 360) % 360
    return degree_en


# 2.2 转弯结束点经纬高
def turning_over_position(enemy_center, enemy_speed, uav_position, uav_speed, uav_direction, uav_center, radius):
    converter = builtconverter(uav_center, enemy_center)
    angle_deg = 180 #角度值数值
    turn_start_pos, turn_start_pos_dms, new_center_dms = turning_start_position(enemy_center, enemy_speed, uav_position, uav_speed, uav_direction, uav_center)

    #转弯前中心点位置与航向
    before_center_lat, before_center_lon, before_center_alt = GeodeticConverter.decimal_dms_to_degrees(new_center_dms)
    heading0 = float(uav_direction) #我方航向，用于三角函数计算

    #计算每架无人机的转前相对位移情况
    rel_offsets = []
    for (lat, lon, alt) in turn_start_pos:
        # EN--位移（米）
        e, n = ll_to_en(before_center_lat, before_center_lon, lat, lon)
        # 转到机体坐标（相对位移）
        x_fwd, y_right = world_to_body(e, n, heading0)
        z_up = float(alt) - before_center_alt
        rel_offsets.append((x_fwd, y_right, z_up))
    print("转弯之前的offset：", rel_offsets)
    #转弯之后的中心点位置与新航向
    after_turn_center = after_turn_position(radius, new_center_dms, angle_deg, uav_direction, uav_center, enemy_center, direction='right')
    after_center_lat, after_center_lon, after_center_alt = GeodeticConverter.decimal_dms_to_degrees(after_turn_center)
    heading1 = (heading0 - angle_deg) % 360 # 右转 180°：bearing 减 180；左转则加 180（保持与 after_turn_position 一致）
    print("转弯之后的中心位置:", after_center_lat, after_center_lon, after_center_alt)

    # 计算每架无人机的转后绝对位置情况
    turn_over_pos = []
    turn_over_dms = []
    for (x_fwd, y_right, z_up) in rel_offsets:
        # 机体--EN
        e, n = body_to_world(x_fwd, y_right, heading1)
        # EN--经纬
        lat1, lon1 = en_to_ll(after_center_lat, after_center_lon, e, n)
        alt1 = after_center_alt + z_up
        turn_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(lat1, lon1, alt1))
        turn_over_pos.append([lat1, lon1, alt1])
        turn_over_dms.append(turn_dms)

    # 测试位移量
    rel1_offsets = []
    for (lat, lon, alt) in turn_over_pos:
        # EN--位移（米）
        e, n = ll_to_en(after_center_lat, after_center_lon, lat, lon)
        # 转到机体坐标（相对位移）
        x1_fwd, y1_right = world_to_body(e, n, heading0-180)
        z1_up = float(alt) - after_center_alt
        rel1_offsets.append((x1_fwd, y1_right, z1_up))
    # print("转弯之后的offset：", rel1_offsets)
    return turn_over_pos, turn_over_dms


# 2.3 绕飞路径散点
# ---------- 1) 预计算先验信息 ----------
def setup_turn(uav_direction, radius, uav_speed, uav_center,
                       enemy_center, enemy_speed, uav_position, direction):
    """
    uav_direction:     初始中心朝向（bearing, 度）
    radius:         转弯半径（米）
    enemy_speed:        阵型中心转弯速度（m/s）
    direction:        'left' 或 'right'
    ============================================
    new_center_dms: [lon_dms, lat_dms, alt]
    turn_start_pos:   转弯前我方绝对点位 [[lat,lon,alt], ...]（十进制度）
    返回 ctx: 用于按时间查询状态
    """
    # 中心经纬高（十进制度）
    turn_start_pos, turn_start_pos_dms, new_center_dms = turning_start_position(enemy_center, enemy_speed, uav_position,
                                                                               uav_speed, uav_direction, uav_center)
    lat0, lon0, alt0 = GeodeticConverter.decimal_dms_to_degrees(new_center_dms) # 转弯前中心点位置
    ψ0 = float(uav_direction) #转前航向

    # 圆心C的经纬
    offset_bearing = (ψ0 + (90 if direction=='left' else -90)) % 360
    C = distance(meters=radius).destination(Point(lat0, lon0), bearing=offset_bearing)
    latC, lonC = C.latitude, C.longitude

    # β0：从圆心C看向我方无人机中心O0的方位
    e0, n0 = ll_to_en(latC, lonC, lat0, lon0)
    β0 = bearing_from_en(e0, n0) #绕飞角度

    # 阵型相对位移（机体：前/右/上），一次性求好
    rel_offsets = []
    for (lat, lon, alt) in turn_start_pos:
        e, n = ll_to_en(lat0, lon0, lat, lon)          # 世界EN
        x_fwd, y_right = world_to_body(e, n, ψ0)       # 转到机体
        z_up = float(alt) - alt0
        rel_offsets.append((x_fwd, y_right, z_up))

    # 角速度（度/秒）
    omega_deg = (uav_speed / radius) * 180.0 / pi

    #返回圆心位置、转弯前的航向、半径、角速度、转弯方向（左or右），我方无人机与中心的相对位置
    return {
        'latC': latC, 'lonC': lonC, 'altC': alt0, 'β0': β0, 'ψ0': ψ0,
        'R': radius, 'omega_deg': omega_deg,
        'dir_sign': (+1 if direction=='left' else -1),
        'rel_offsets': rel_offsets
    }

# ---------- 2) 按任意时间t查询状态 ----------
def turn_state_at_time(ctx, t_sec, uav_center, enemy_center):
    """
    输入ctx：已求得的先验信息
    输入时间： t_sec ∈ [0, 180/omega_deg]
    返回：
      我方中心--center: (lat, lon, alt)
      当前中心航向--heading_deg
      当前阵型绝对位置--positions: [[lat,lon,alt], ...]
      角度上限--clamp_deg: 暂定180--clamp_deg=180(参数)
    """
    converter = builtconverter(uav_center, enemy_center)
    #通过角速度、时间、转向计算已绕飞角度
    dir_sign = ctx['dir_sign']
    θ = ctx['omega_deg'] * t_sec * dir_sign

    # 当前t_sec的新中心
    βt = (ctx['β0'] + θ) % 360
    Pt = distance(meters=ctx['R']).destination(Point(ctx['latC'], ctx['lonC']), bearing=βt)
    center_lat_t, center_lon_t, center_alt_t = Pt.latitude, Pt.longitude, ctx['altC']  # 若高度不变

    # 当前t_sec新朝向（航向角）
    heading_t = (ctx['ψ0'] + θ) % 360

    # 当前t_sec阵型各机绝对位置
    positions = []
    pos_dms = []
    for (x_fwd, y_right, z_up) in ctx['rel_offsets']:
        e, n = body_to_world(x_fwd, y_right, heading_t)
        lat_i, lon_i = en_to_ll(center_lat_t, center_lon_t, e, n)
        pos_dms_i = converter.local_to_geodetic_dms(converter.geodetic_to_local(lat_i, lon_i, center_alt_t + z_up))
        positions.append([lat_i, lon_i, center_alt_t + z_up])
        pos_dms.append(pos_dms_i)

    # 返回任意时间点的我方中心位置，当前航向角（朝向）、阵型绝对位置
    return (center_lat_t, center_lon_t, center_alt_t), heading_t, positions, pos_dms


def output_turn_positions(ctx, uav_center, enemy_center):
    """
    给定 ctx （包含转弯过程的所有参数），输出每 10° 转弯时刻对应的阵型位置。
    返回：
      angles: 旋转角度（10°, 20°, ..., 180°）
      centers: 每个角度对应的阵型中心位置 (lat, lon, alt)
      headings: 每个角度对应的阵型航向
      positions: 每个角度对应的阵型中各无人机的绝对位置 [[lat, lon, alt], ...]
    """

    # 最大转弯角度
    max_angle = 180
    # 角度间隔
    angle_interval = 45
    # 存储结果
    angles = [] # 旋转角度（10°, 20°, ..., 180°）
    centers = [] # 每个角度对应的阵型中心位置
    headings = [] # 每个角度对应的阵型航向
    positions_list = [] # 每个角度对应的阵型中各无人机的绝对位置local
    pos_dms_list = [] # 每个角度对应的阵型中各无人机的绝对位置dms

    # 对于每个10度的角度，从0度开始到180度
    for angle in range(0, max_angle + 1, angle_interval):
        # 计算每个角度对应的时间 t_sec
        t_sec = angle / ctx['omega_deg']  # 根据角度除以角速度来获取时间

        # 查询当前时刻的阵型状态
        center, heading, positions, pos_dms = turn_state_at_time(ctx, t_sec, uav_center, enemy_center)

        # 保存结果
        angles.append(angle)
        centers.append(center)
        headings.append(heading)
        positions_list.append(positions)
        pos_dms_list.append(pos_dms)

    return angles, centers, headings, positions_list, pos_dms_list

def to_form_list(data):
    forms = []
    for i, item in enumerate(data):
        # 取出唯一键和值
        (k, v), = item.items()
        # 只保留统一键名 "Form"
        forms.append({f"bearing_{i * 10}": v})
        # forms.append(v)
    return forms


def plot_uav_formation(turn_start_pos, turn_over_pos, positions, angles):
    """
    可视化无人机转弯前后的阵型点位

    参数:
    turn_start_pos (list): 转弯前阵型的经纬度列表，每个元素是一个坐标 [latitude, longitude]
    turn_over_pos (list): 转弯后阵型的经纬度列表，每个元素是一个坐标 [latitude, longitude]
    positions (list of lists): 转弯中阵型的点位，每个元素是一个包含多个点的列表，每个点是一个坐标 [latitude, longitude]
    angles (list): 与 positions 对应的转弯角度列表
    """
    # 提取起始和终点经纬度
    turn_start_latitudes = [point[0] for point in turn_start_pos]
    turn_start_longitudes = [point[1] for point in turn_start_pos]
    print("干什么啊！！！！",  turn_start_latitudes)
    turn_end_latitudes = [point[0] for point in turn_over_pos]
    turn_end_longitudes = [point[1] for point in turn_over_pos]

    # 绘制图形
    plt.figure(figsize=(10, 10))

    # 只绘制转弯前阵型的起始点（蓝色点）和转弯后的阵型点位（红色点）
    plt.scatter(turn_start_longitudes, turn_start_latitudes, color='blue', label='Turn Start Points', marker='o', s=100)

    # 只绘制转弯后的阵型点位
    plt.scatter(turn_end_longitudes, turn_end_latitudes, color='red', label='Turn End Points', marker='x', s=100)

    # 绘制中间的转弯点
    for i in range(len(angles)):
        plt.scatter([point[1] for point in positions[i]], [point[0] for point in positions[i]], color='green', label='Turn Mid Points', marker='+')

    # 设置标题和标签
    plt.title('UAV Formation Before and After Turning')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.show()



def turning_and_plot(
    enemy_speed: float,
    uav_speed: float,
    uav_direction: float,
    radius: float,
    uav_position: List[Any],
    enemy_center: Optional[Any] = None,
    uav_center: Optional[Any] = None
) -> Dict[str, Any]:
    # 如果参数为 None，则使用默认值
    # enemy_center = enemy_center or [
    #     "128:19:36.77E",
    #     "29:28:00.11N",
    #     "5000.0"
    # ]
    # uav_center = uav_center or [
    #     "121:09:10.25E",
    #     "29:40:36.34N",
    #     "5000.0"
    # ]
    # uav_position = uav_position or [
    #     [
    #         "120:38:16.97E",
    #         "29:46:10.11N",
    #         "5000"
    #     ],
    #     [
    #         "120:38:13.80E",
    #         "29:46:10.14N",
    #         "5000"
    #     ],
    #     [
    #         "120:38:10.64E",
    #         "29:46:10.17N",
    #         "5000"
    #     ],
    #     [
    #         "120:38:7.47E",
    #         "29:46:10.20N",
    #         "5000"
    #     ],
    #     [
    #         "120:38:19.06E",
    #         "29:46:29.55N",
    #         "5000"
    #     ],
    #     [
    #         "120:38:15.89E",
    #         "29:46:29.59N",
    #         "5000"
    #     ],
    #     [
    #         "120:38:12.73E",
    #         "29:46:29.62N",
    #         "5000"
    #     ],
    #     [
    #         "120:38:9.56E",
    #         "29:46:29.66N",
    #         "5000"
    #     ],
    #     [
    #         "120:38:6.40E",
    #         "29:46:29.69N",
    #         "5000"
    #     ]
    # ]

    # 转弯起点
    turn_start_pos, turn_start_pos_dms, new_center_dms = turning_start_position(
        enemy_center, enemy_speed, uav_position, uav_speed, uav_direction, uav_center
    )
    print("turn_start_pos：", turn_start_pos)

    # 转弯终点
    turn_over_pos, turn_over_dms = turning_over_position(
        enemy_center, enemy_speed, uav_position, uav_speed, uav_direction, uav_center, radius
    )
    # print("转弯之后的阵型点位：", turn_over_dms)

    # 先验信息
    ctx = setup_turn(
        uav_direction, radius, uav_speed, uav_center, enemy_center, enemy_speed, uav_position, direction='right'
    )
    # 绕飞路径散点 10秒间隔的角度、中心位置、航向角、阵型位置
    angles, centers, headings, positions, pos_dms = output_turn_positions(ctx, uav_center, enemy_center)

  #  plot_uav_formation(turn_start_pos, turn_over_pos, positions, angles)

    forms = []
    for i, group in enumerate(pos_dms):
        # group 已是 9 个 [lon,lat,alt]（DMS字符串），按目标结构直接塞入
        forms.append({f"Form_{i * 10}": [[p[0], p[1], p[2]] for p in group]})
    # print("阵型散点点位：", forms)

    all_forms = to_form_list(forms)
    # print("~~~~~~", all_forms)

    # 转为 JSON 字符串（中文不转义 + 缩进）
    out = {
        "Form": all_forms
    }

    return out


def api_main(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    给 Flask 等服务调用的入口。
    config 来自接口的 JSON，形如：
    {
        "enemy_center": [lat, lon, alt],
        "uav_center": [lat, lon, alt],
        "uav_position": [[lat, lon, alt], ...],
        "enemy_speed": 240,
        "uav_speed": 300,
        "uav_direction": 0,
        "radius": 200
    }
    """

    # 1) 获取敌机和无人机的坐标等参数
    enemy_center = config.get("enemy_center")
    uav_center = config.get("uav_center")
    uav_position = config.get("uav_position")
    enemy_speed = config.get("enemy_speed", 240)  # 默认值为240
    uav_speed = config.get("uav_speed", 300)  # 默认值为300
    uav_direction = config.get("uav_direction", 0)  # 默认值为0
    radius = config.get("radius", 200)  # 默认值为200

    if not enemy_center or not uav_center or not uav_position:
        raise ValueError("config 中缺少必要的字段: enemy_center, uav_center, 或 uav_position")


    # 调用原来的核心逻辑（转弯半径散点计算和绘图）
    info = turning_and_plot(
        enemy_center = enemy_center,
        uav_center = uav_center,
        uav_position = uav_position,
        enemy_speed = enemy_speed,
        uav_speed = uav_speed,
        uav_direction = uav_direction,
        radius = radius
    )

    # 直接把 info 返回给 Flask，Flask 会再用 to_jsonable 转成 JSON
    return info


# def main(config: dict | None = None):
#     return turning_and_plot(config or {})


def run_port(config: dict | None = None) -> dict:
    test_config = {
        # 敌机经纬高，这里用简单的三架，单位就是 degree + 米
        # 我方中心，可以随便给一个大致在附近的点
        "enemy_speed": 240,
        "uav_speed": 300,
        "uav_direction": 45,
        "radius": 20,
        "enemy_center": ["128:19:36.77E", "29:28:00.11N", "5000.0"],
        "uav_center": ["121:09:10.25E", "29:40:36.34N", "5000.0"],
        "uav_position": [
            [
                "120:38:16.97E",
                "29:46:10.11N",
                "5000"
            ],
            [
                "120:39:16.97E",
                "29:45:10.11N",
                "5000"
            ]

        ]
        # 输出目录，接口会在这个目录下生成 png 和 c_points_dms.json
        # "outdir": "./output_c_demo2"
    }
    # 如果外面没传 config，就用 test_execute_main 里的那一份
    if config is None:
        config = test_config

    # 获取所需参数
    uav_center = config.get("uav_center", None)
    enemy_center = config.get("enemy_center",None)
    uav_position = config.get("uav_position",None)
    enemy_speed = config.get("enemy_speed", 240)  # 默认值为240
    uav_speed = config.get("uav_speed", 300)  # 默认值为300
    uav_direction = config.get("uav_direction", 0)  # 默认值为0
    radius = config.get("radius", 200)  # 默认值为200

    info = turning_and_plot(
        enemy_center = enemy_center,
        uav_center = uav_center,
        uav_position = uav_position,
        enemy_speed = enemy_speed,
        uav_speed = uav_speed,
        uav_direction = uav_direction,
        radius = radius
    )

    # 打印一下结果，方便在终端里看
    print("得到结果：",info)
    return info


if __name__ == "__main__":
    run_port()