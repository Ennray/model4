import math
import json
from geopy.distance import geodesic
import GeodeticConverter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import List, Tuple, Any, Optional, Dict
from test_dropbomb_data import config as test_config


# 构建坐标系
def builtconverter(uav_center,  enemy_center):
    # enemy_center_lat, enemy_center_lon, enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    # uav_center_lat, uav_center_lon, uav_center_alt = GeodeticConverter.decimal_dms_to_degrees(uav_center)
    converter = GeodeticConverter.GeodeticToLocalConverter(uav_center[0], uav_center[1], uav_center[2],
                                                           enemy_center[0], enemy_center[1], enemy_center[2])
    return converter


# 计算投弹所需下落时间 (自由落体)
def calculate_fall_time(height_difference):
    g = 9.81
    time_drop = math.sqrt(2 * height_difference / g)
    return time_drop


# 计算time_to_first_drop时间我方到达后一个敌机前上方的位置，设定5秒开始投弹
def drop_first_enemy(enemy_pos, enemy_center, enemy_speed, uav_speed, uav_center,
                     time_to_first_drop, Spacing, bearing, height_difference, delta_distance, delta_time, time_buffer):
    # delta_distance = 1 # 根据动力学模型调整抛物线横向距离（可调整）
    # delta_time = 1 #设置投弹时间差为1s（可调整）
    converter = builtconverter(uav_center, enemy_center)

    # 与敌群产生相对速度
    relative_speed =  Spacing / delta_time # 间距/两个弹药投放的时间差
    uav_after_speed = uav_speed + relative_speed # 开始投弹的速度 即C点速度

    # =============这里是C点信息=======================================
    # C点→敌机位置：暂定5秒从尾斜后到达撞击点
    # enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_pos)
    enemy_action_lat, enemy_action_lon, enemy_action_alt = converter.calculate_destination_point(enemy_pos[0], enemy_pos[1],
                                                                                                 enemy_pos[2],
                                                                                                 bearing,
                                                                                                 time_to_first_drop * enemy_speed,
                                                                                                 0)
    enemy_action_pos = [enemy_action_lat, enemy_action_lon, enemy_action_alt]
    # enemy_action_pos = converter.local_to_geodetic_dms(converter.geodetic_to_local(enemy_action_lat, enemy_action_lon, enemy_action_alt))

    # C点→我方uav坐标：5s之后的投弹无人机位置，高度高出10m（>7m）
    # 由敌机位置结合时间计算：到达投弹点时间time_to_first_drop + 弹药下落时间time_drop，缓冲距离（回退距离delta_distance）由仿真软件动力学模型决定，为弹药下落预留距离
    time_drop = calculate_fall_time(height_difference) #弹药下落时间可根据仿真动力学模型调整
    drop_action_lat, drop_action_lon, drop_action_alt = converter.calculate_destination_point(enemy_pos[0], enemy_pos[1],
                                                                                              enemy_pos[2], bearing,
                                                                                              (time_to_first_drop + time_drop) * enemy_speed - delta_distance,
                                                                                              10)
    drop_action_pos = [drop_action_lat, drop_action_lon, drop_action_alt]

    # =============这里是B点信息=======================================
    # B点→敌群位置: 暂定缓冲时间time_buffer为2秒 （可调整）
    # time_buffer = 2
    enemy_B_lat, enemy_B_lon, enemy_B_alt = converter.calculate_destination_point(enemy_pos[0], enemy_pos[1], enemy_pos[2],
                                                                                  bearing, (time_to_first_drop - time_buffer) * enemy_speed, 0)
    enemy_B_pos = [enemy_B_lat, enemy_B_lon, enemy_B_alt]

    # B点→我方uav坐标：设定与C点速度一致uav_after_speed，反方向bearing + 180倒推
    # 由我方uav投弹点C→回退缓冲时间time_buffer和投弹速度uav_after_speed，得到距离
    drop_B_lat, drop_B_lon, drop_B_alt = converter.calculate_destination_point(drop_action_lat, drop_action_lon, drop_action_alt, bearing + 180,
                                                                                              time_buffer * uav_after_speed, 0)
    uav_B_pos = [drop_B_lat, drop_B_lon, drop_B_alt]

    # drop_action_pos = converter.local_to_geodetic_dms(
    #     converter.geodetic_to_local(drop_action_lat, drop_action_lon, drop_action_alt))
    return enemy_action_pos, drop_action_pos, uav_after_speed, uav_B_pos, enemy_B_pos


# 计算相邻点位的位置
def diff_between_pos(lat1, lon1, lat2, lon2):
    diff_distance = geodesic((lat1, lon1), (lat2, lon2)).meters
    return diff_distance


# 由相邻距离判断行数列数
def calculate_formation(enemy_position, threshold_distance=1000):
    """
    计算阵型行列数目
    enemy_position: 阵型的所有无人机位置，每个位置是 [latitude, longitude, altitude]
    threshold_distance: 判断是否换行的距离阈值，单位：米
    """
    if len(enemy_position) == 1:
        # 如果只有一个点位，直接返回1行1列
        return 1, 1

    num_rows = 1  # 初始化行数为1
    drones_per_row = [1]  # 第一行默认有1架无人机
    prev_position = enemy_position[0]

    for i in range(1, len(enemy_position)):
        curr_position = enemy_position[i]

        # 计算相邻两个点之间的距离
        distance = diff_between_pos(prev_position[0], prev_position[1], curr_position[0], curr_position[1])

        if distance > threshold_distance:
            # 如果距离大于阈值，则换行
            num_rows += 1
            drones_per_row.append(1)  # 新的一行开始，第一架无人机
        else:
            # 如果距离小于阈值，则当前行增加无人机
            drones_per_row[-1] += 1

        prev_position = curr_position  # 更新前一个点位为当前点位

    return num_rows, max(drones_per_row)


# 处理单层阵列的奇偶行位置
def process_layer(enemy_positions):
    print("单层的位置都在这里", enemy_positions)
    # 先计算行数和列数
    total_rows, line = calculate_formation(enemy_positions, threshold_distance=1000)
    print("total_rows, line",total_rows, line)

    # 检查最后一行是否完整
    total_enemies = len(enemy_positions)
    last_row_start = (total_rows - 1) * line
    last_row_count = total_enemies - last_row_start  # 最后一行的实际元素数量
    is_last_row_incomplete = last_row_count < line

    print(f"最后一行元素数量: {last_row_count}/{line}, 是否不完整: {is_last_row_incomplete}")

    # 判断行数是奇数还是偶数
    if total_rows % 2 == 0:
        # 偶数行：提取所有偶数索引行（实际显示的第2,4,6...行）
        selected_rows = list(range(1, total_rows, 2))  # 索引[1,3,5,...]
        print("selected_rows", selected_rows)
    else:
        if total_rows == 1:
            # 只有一行时，只提取第一行（索引0）
            selected_rows = [0]
        else:
            # 奇数行：提取偶数索引行和最后一行
            selected_rows = list(range(1, total_rows, 2)) + [total_rows - 1]

    #投弹位置
    single_layer_positions = []
    # 遍历所有投弹行的位置
    if selected_rows == [0]:
        # 只有一行，直接返回所有敌机
        return enemy_positions
    else:
        for row in selected_rows:
            # 正确提取每行的所有敌机
            pos = enemy_positions[row * line : (row + 1) * line]  # 第row行的所有列
            single_layer_positions.extend(pos)

        # 特殊处理：如果最后一行不完整，需要补充倒数第二行的后半部分
        if is_last_row_incomplete and total_rows > 1:
            second_last_row_index = total_rows - 2

            # 倒数第二行的后半部分（未被最后一行对齐的部分）
            # 从 last_row_count 位置开始到行尾
            second_last_row_start = second_last_row_index * line
            extra_start = second_last_row_start + last_row_count  # 从第last_row_count列开始
            extra_end = (second_last_row_index + 1) * line  # 到行尾

            extra_positions = enemy_positions[extra_start:extra_end]

            if extra_positions:
                single_layer_positions.extend(extra_positions)
                print(f"最后一行不完整，补充倒数第二行(索引{second_last_row_index})的后{len(extra_positions)}个敌机")

    print(f"选择的行: {selected_rows}, 总投弹数: {len(single_layer_positions)}")
    return single_layer_positions


# 判断是单层还是多层阵型，循环提取每一层应投弹的位置
def process_drop_position(enemy_positions):
    # 根据海拔高度判断是否有多层，使用±30容差
    processed_positions = []

    if not enemy_positions:
        return processed_positions

    # 提取所有海拔高度
    all_heights = [pos[2] for pos in enemy_positions]
    # print("所有海拔", all_heights)
    # 对高度进行分组，±30范围内的视为同一海拔层
    height_groups = []
    used_indices = set()

    for i, height in enumerate(all_heights):
        if i in used_indices:
            continue

        # 找到当前高度±30范围内的所有点
        current_group = []
        for j, h in enumerate(all_heights):
            if j not in used_indices and abs(h - height) <= 30:
                current_group.append(j)
                used_indices.add(j)
        # print("当前组", current_group)
        height_groups.append(current_group)
    # print("height_groups", len(height_groups))
    # 判断是单层还是多层
    if len(height_groups) == 1:
        # 只有一个海拔层，按单层处理
        processed_positions = process_layer(enemy_positions)
        # print("这是什么单层结果", processed_positions)
    else:
        # 存在多个海拔层，按每个海拔层分别处理
        for group_indices in height_groups:
            # print("group_indices", group_indices)
            # 提取该层的所有数据
            layer_positions = [enemy_positions[i] for i in group_indices]
            # print("layer_positions", layer_positions)
            processed_positions.extend(process_layer(layer_positions))

    return processed_positions


# 输入格式转换，均转为十进制的输出
def convert_to_decimal(enemy_position: List[List[Any]]) -> List[List[float]]:

    # 如果输入的是单个位置
    if isinstance(enemy_position[0], str):  # 单点输入时 enemy_position 传入的是一个位置列表
        enemy_position = [enemy_position]

    # 输出列表
    decimal_positions = []

    for position in enemy_position:
        if isinstance(position[0], str):  # 如果经度是字符串，表示是DMS格式
            # 转换DMS格式为十进制，返回的是元组，转成列表
            decimal_position = list(GeodeticConverter.decimal_dms_to_degrees(position))
        else:  # 否则，假设输入已经是十进制格式
            decimal_position = [
                position[0],  # 纬度
                position[1],  # 经度
                position[2]  # 海拔
            ]
        decimal_positions.append(decimal_position)

    return decimal_positions

# 投弹点信息====核心算法
def drop_bombs_information(
        enemy_speed: float,
        uav_speed: float,
        spacing: float,
        time_to_first_drop: float,
        bearing: float,
        delta_distance: float,
        delta_time: float,
        time_buffer: float,
        enemy_position: List[Any],
        uav_center: Optional[Any] = None,
        enemy_center: Optional[Any] = None
) -> Dict[str, Any]:
    '''参数设定
    # enemy_speed 敌机速度 m/s
    # uav_speed我方无人机速度 m/s
    # Spacing敌机之间的距离 m
    # time_to_first_drop初始到投弹点的时间10s 非固定
    # bearing航向
    # delta_distance两次投弹的距离差
    # delta_time两次投弹的时间差
    # time_buffer缓冲时间B-C
    '''
    g = 9.81  # 重力加速度 m/s^2
    height_difference = 10  # 海拔差 10米
    # 中心坐标转换
    enemy_center_local = convert_to_decimal(enemy_center)
    uav_center_local = convert_to_decimal(uav_center)

    # 获取敌方坐标数据lat/lon/alt
    enemy_positions =convert_to_decimal(enemy_position)
    # 找出不同阵型投弹对应的敌群坐标 拟合所有种类的敌群
    processed_positions = process_drop_position(enemy_positions)
    # 双方投弹BC点
    uav_C_pos_all = []
    uav_B_pos_all = []
    enemy_C_pos_all = []
    enemy_B_pos_all = []
    for enemy_pos in processed_positions:
        enemy_action_pos, drop_action_pos, uav_after_speed, uav_B_pos, enemy_B_pos = drop_first_enemy(enemy_pos,
                                                                                                      enemy_center_local[0],
                                                                                                      enemy_speed,
                                                                                                      uav_speed,
                                                                                                      uav_center_local[0],
                                                                                                      time_to_first_drop,
                                                                                                      spacing, bearing,
                                                                                                      height_difference,
                                                                                                      delta_distance, delta_time, time_buffer)
        # 这里输出都是十进制
        uav_C_pos_all.append(drop_action_pos)
        uav_B_pos_all.append(uav_B_pos)
        enemy_C_pos_all.append(enemy_action_pos)
        enemy_B_pos_all.append(enemy_B_pos)

##===============如果输入是度数dms（贾），得到结果之后需要转为度数dms==============
    converter = builtconverter(uav_center_local[0],  enemy_center_local[0]) #两种版本在这里均为十进制，可直接适配函数
    uav_C_pos_all_dms = []
    uav_B_pos_all_dms = []
    enemy_C_pos_all_dms = []
    enemy_B_pos_all_dms = []
    for pos in uav_C_pos_all:
        pos_dms = converter.local_to_geodetic_dms(
            converter.geodetic_to_local(pos[0], pos[1], pos[2]))
        uav_C_pos_all_dms.append(pos_dms)
    for pos in uav_B_pos_all:
        pos_dms = converter.local_to_geodetic_dms(
            converter.geodetic_to_local(pos[0], pos[1], pos[2]))
        uav_B_pos_all_dms.append(pos_dms)
    for pos in enemy_C_pos_all:
        pos_dms = converter.local_to_geodetic_dms(
            converter.geodetic_to_local(pos[0], pos[1], pos[2]))
        enemy_C_pos_all_dms.append(pos_dms)
    for pos in enemy_B_pos_all:
        pos_dms = converter.local_to_geodetic_dms(
            converter.geodetic_to_local(pos[0], pos[1], pos[2]))
        enemy_B_pos_all_dms.append(pos_dms)
    out = {
        "uav_B": uav_B_pos_all_dms,
        "uav_C": uav_C_pos_all_dms,
        "enemy_B": enemy_B_pos_all_dms,
        "enemy_C": enemy_C_pos_all_dms,
    }
# ============创建三维散点图===>可滑动缩放→添加到drop_bombs_information函数当中===============
    fig = go.Figure()

    # 绘制我方投弹C点
    fig.add_trace(go.Scatter3d(
        x=[pos[1] for pos in uav_C_pos_all],
        y=[pos[0] for pos in uav_C_pos_all],
        z=[pos[2] for pos in uav_C_pos_all],
        mode='markers',
        marker=dict(size=6, color='blue'),
        name='我方投弹C点'
    ))

    # 绘制敌方投弹C点
    fig.add_trace(go.Scatter3d(
        x=[pos[1] for pos in enemy_C_pos_all],
        y=[pos[0] for pos in enemy_C_pos_all],
        z=[pos[2] for pos in enemy_C_pos_all],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='敌方投弹C点'
    ))

    # 绘制我方投弹B点
    fig.add_trace(go.Scatter3d(
        x=[pos[1] for pos in uav_B_pos_all],
        y=[pos[0] for pos in uav_B_pos_all],
        z=[pos[2] for pos in uav_B_pos_all],
        mode='markers',
        marker=dict(size=6, color='green'),
        name='我方投弹B点'
    ))

    # 绘制敌方投弹B点
    fig.add_trace(go.Scatter3d(
        x=[pos[1] for pos in enemy_B_pos_all],
        y=[pos[0] for pos in enemy_B_pos_all],
        z=[pos[2] for pos in enemy_B_pos_all],
        mode='markers',
        marker=dict(size=6, color='orange'),
        name='敌方投弹B点'
    ))

    # 设置图表的标题和标签
    fig.update_layout(
        title='投弹点位三维可视化',
        scene=dict(
            xaxis_title='经度',
            yaxis_title='纬度',
            zaxis_title='海拔'
        ),
        legend=dict(
            x=0.85,
            y=0.95
        )
    )
    # 显示图形
    # fig.show()

    enemy_lons, enemy_lats, enemy_alts = [], [], []
    for dms in enemy_position:
        lon, lat, alt = GeodeticConverter.decimal_dms_to_degrees(dms)
        enemy_lons.append(lon)
        enemy_lats.append(lat)
        enemy_alts.append(alt)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(enemy_lons, enemy_lats, enemy_alts,
               s=5, c='gray', alpha=0.6, label='Enemies')


    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_zlabel("Altitude (m)")
    ax.legend()

    # plt.show()
#=======如果需要十进制的版本，就用下面的out，把上面这段全部注释======================

    # out = {
    #     "我方B缓冲点": uav_B_pos_all,
    #     "我方C投弹点": uav_C_pos_all,
    #     "敌方B点": enemy_B_pos_all,
    #     "敌方C点": enemy_C_pos_all,
    # }
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
        "uav_direction": 0
    }
    """

    # 获取敌机和无人机的坐标等参数
    enemy_speed = config.get("enemy_speed")  # 默认值为240
    uav_speed = config.get("uav_speed")  # 默认值为300
    spacing = config.get("spacing")
    time_to_first_drop = config.get("time_to_first_drop")
    bearing = config.get("bearing")
    enemy_position = config.get("enemy_position")
    uav_center = config.get("uav_center")
    enemy_center = config.get("enemy_center")
    # 固定的参数
    delta_distance = 1 #投弹间隔距离
    delta_time = 1 #投弹间隔时间
    time_buffer = 2 #缓冲时间

    if not enemy_center or not uav_center or not enemy_position:
        raise ValueError("config 中缺少必要的字段: enemy_center, uav_center, 或 uav_position")


    # 调用原来的核心逻辑（转弯半径散点计算和绘图）
    info = drop_bombs_information(
        enemy_speed = enemy_speed,
        uav_speed = uav_speed,
        spacing = spacing,
        time_to_first_drop = time_to_first_drop,
        bearing = bearing,
        delta_distance = delta_distance,
        delta_time = delta_time,
        time_buffer = time_buffer,
        enemy_position = enemy_position,
        uav_center = uav_center,
        enemy_center = enemy_center
    )

    # 直接把 info 返回给 Flask，Flask 会再用 to_jsonable 转成 JSON
    return info


# def main(config: dict | None = None):
#     return turning_and_plot(config or {})


def run_port(config: dict | None = None) -> dict:
    # test_config = {
    #     # 敌机经纬高，这里用简单的三架，单位就是 degree + 米
    #     # 我方中心，可以随便给一个大致在附近的点
    #     "enemy_speed": 240,
    #     "uav_speed": 300,
    #     "spacing": 10,
    #     "time_to_first_drop": 1,
    #     "bearing": 45,
    #     "delta_distance": 1, #抛物线横、
    #     "delta_time": 1,
    #     "time_buffer": 2,
    #     "enemy_position": [
    #         [
    #             "120:38:16.97E",
    #             "29:46:10.11N",
    #             "5000"
    #         ],
    #         [
    #             "120:39:16.97E",
    #             "29:45:10.11N",
    #             "5000"
    #         ]
    #
    #     ],
    #     "enemy_center": ["128:19:36.77E", "29:28:00.11N", "5000.0"],
    #     "uav_center": ["121:09:10.25E", "29:40:36.34N", "5000.0"]
    #     # 输出目录，接口会在这个目录下生成 png 和 c_points_dms.json
    #     # "outdir": "./output_c_demo2"
    # }
    # 如果外面没传 config，就用 test_config里的那一份
    if config is None:
        config = test_config

    enemy_speed = config.get("enemy_speed")  # 默认值为240
    uav_speed = config.get("uav_speed")  # 默认值为300
    spacing = config.get("spacing")
    time_to_first_drop = config.get("time_to_first_drop")
    bearing = config.get("bearing")
    delta_distance = config.get("delta_distance")
    delta_time = config.get("delta_time")
    time_buffer = config.get("time_buffer")
    enemy_position = config.get("enemy_position")
    uav_center = config.get("uav_center")
    enemy_center = config.get("enemy_center")

    if not enemy_center or not uav_center or not enemy_position:
        raise ValueError("config 中缺少必要的字段: enemy_center, uav_center, 或 uav_position")

    # 调用原来的核心逻辑（投弹BC双方点位计算）
    info = drop_bombs_information(
        enemy_speed=enemy_speed,
        uav_speed=uav_speed,
        spacing=spacing,
        time_to_first_drop=time_to_first_drop,
        bearing=bearing,
        delta_distance=delta_distance,
        delta_time=delta_time,
        time_buffer=time_buffer,
        enemy_position=enemy_position,
        uav_center=uav_center,
        enemy_center=enemy_center
    )

    # 打印一下结果，方便在终端里看
    print("得到结果：",len(info["uav_C"]))
    return info


if __name__ == "__main__":
    run_port()

# # ============创建三维散点图===>可滑动缩放→添加到drop_bombs_information函数当中===============
#     fig = go.Figure()
#
#     # 绘制我方投弹C点
#     fig.add_trace(go.Scatter3d(
#         x=[pos[1] for pos in uav_C_pos_all],
#         y=[pos[0] for pos in uav_C_pos_all],
#         z=[pos[2] for pos in uav_C_pos_all],
#         mode='markers',
#         marker=dict(size=6, color='blue'),
#         name='我方投弹C点'
#     ))
#
#     # 绘制敌方投弹C点
#     fig.add_trace(go.Scatter3d(
#         x=[pos[1] for pos in enemy_C_pos_all],
#         y=[pos[0] for pos in enemy_C_pos_all],
#         z=[pos[2] for pos in enemy_C_pos_all],
#         mode='markers',
#         marker=dict(size=6, color='red'),
#         name='敌方投弹C点'
#     ))
#
#     # 绘制我方投弹B点
#     fig.add_trace(go.Scatter3d(
#         x=[pos[1] for pos in uav_B_pos_all],
#         y=[pos[0] for pos in uav_B_pos_all],
#         z=[pos[2] for pos in uav_B_pos_all],
#         mode='markers',
#         marker=dict(size=6, color='green'),
#         name='我方投弹B点'
#     ))
#
#     # 绘制敌方投弹B点
#     fig.add_trace(go.Scatter3d(
#         x=[pos[1] for pos in enemy_B_pos_all],
#         y=[pos[0] for pos in enemy_B_pos_all],
#         z=[pos[2] for pos in enemy_B_pos_all],
#         mode='markers',
#         marker=dict(size=6, color='orange'),
#         name='敌方投弹B点'
#     ))
#
#     # 设置图表的标题和标签
#     fig.update_layout(
#         title='投弹点位三维可视化',
#         scene=dict(
#             xaxis_title='经度',
#             yaxis_title='纬度',
#             zaxis_title='海拔'
#         ),
#         legend=dict(
#             x=0.85,
#             y=0.95
#         )
#     )
#     # 显示图形
#     fig.show()



