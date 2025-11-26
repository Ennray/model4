from dataclasses import dataclass
import numpy as np
import GeodeticConverter as GC
from typing import Optional, List, Tuple, Dict, Any
from test_impact_dms_data import config as test_config
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# 求dms坐标直接转化为degree的经度纬度高度
def dms_to_degree(dms):
    if isinstance(dms, (list, tuple)) and len(dms) == 3 and isinstance(dms[0], str):
        lat, lon, alt = GC.decimal_dms_to_degrees(dms)
        return float(lat), float(lon), float(alt)
    return float(dms[0]), float(dms[1]), float(dms[2])


# 求dms坐标转化为局部坐标local向量的值
def dms_to_local(converter, dms):
    lat, lon, alt = dms_to_degree(dms)
    local = converter.geodetic_to_local(lat, lon, alt)
    return local


# 将航向角（北=0°，顺时针）转化为局部坐标系中的单位向量
def bearing_to_enu_unit(bearing) -> np.ndarray:
    rad = np.deg2rad(bearing)
    return np.array([np.sin(rad), np.cos(rad), 0.0], dtype = float)


# 归一化向量
def to_unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# 估计当前无人机到达B点至少要花费多少时间
def reachable_time(now_uav_point, now_uav_speed, target_point_B, a_max=None, v_max=None):
    distance = np.linalg.norm(target_point_B - now_uav_point)
    if distance < 1e-6:
        return 0.0

    v0 = max(now_uav_speed, 1e-3)
    if a_max is None or a_max <= 0:
        return distance / v0

    # 如果不给 v_max，就按无限速度上限算最小时间：
    if v_max is None or v_max <= 0:
        # 解 0.5 * a * t^2 + v0 * t = distance
        A = 0.5 * a_max
        B = v0
        C = -distance
        disc = B * B - 4 * A * C
        if disc <= 0:
            return distance / v0
        t = (-B + np.sqrt(disc)) / (2 * A)
        return t

    # 有 v_max：先看加到 v_max 的距离
    t_acc = max(0.0, (v_max - v0) / a_max)
    d_acc = v0 * t_acc + 0.5 * a_max * t_acc * t_acc

    if distance <= d_acc:
        # 在没到 v_max 之前就能到
        A = 0.5 * a_max
        B = v0
        C = -distance
        disc = B * B - 4 * A * C
        if disc <= 0:
            return distance / v0
        t = (-B + np.sqrt(disc)) / (2 * A)
        return t
    else:
        # 先加到 v_max，再匀速
        d_rest = distance - d_acc
        t_cruise = d_rest / v_max
        return t_acc + t_cruise



# 判断是否有敌机中心，没有则自己计算
def judge_enemy_center(enemy_center, enemy_dms):
    if enemy_center is None:
        degs = np.array([dms_to_degree(x) for x in enemy_dms], float)  # [lat, lon, alt]
        lat0 = float(np.mean(degs[:, 0]))
        lon0 = float(np.mean(degs[:, 1]))
        alt0 = float(np.mean(degs[:, 2])) if degs.shape[1] == 3 else 0.0
        enemy_center = [lat0, lon0, alt0]
    return enemy_center




#==========================================以敌我中心构建全局坐标系converter==============================================
def build_global_converter(uav_center, enemy_center):
    uav_lat, uav_lon, uav_alt = dms_to_degree(uav_center)
    enemy_lat, enemy_lon, enemy_alt = dms_to_degree(enemy_center)
    converter = GC.GeodeticToLocalConverter(uav_lat, uav_lon, uav_alt, enemy_lat, enemy_lon, enemy_alt)
    R_local_to_ENU = converter.rotation_matrix
    R_ENU_to_local = R_local_to_ENU.T

    return converter, R_local_to_ENU, R_ENU_to_local


def find_single_uav_B_point(converter, R_ENU_to_local, uav_point, target_point, wing_half, uav_speed, uav_course_deg, target_speed, target_course_deg, lateral:str = 'right',
                            to_C_window = (1.0, 2.0), to_CD_window = (3.0, 5.0), speed_range = (10.0, 15.0), a_max:float | None = None, v_max:float | None = None,  grid=(21, 9, 7)) ->Optional[Dict[str, Any]]:

    uav_local = dms_to_local(converter, uav_point)
    enemy_local = dms_to_local(converter, target_point)

    # 把我方及敌方航向角（方向）转化成ENU坐标系下的单位方向[E,N,0]
    uav_course_enu = bearing_to_enu_unit(uav_course_deg)
    target_course_enu = bearing_to_enu_unit(target_course_deg)

    # 将我方及敌方ENU速度转变为local坐标下的速度向量
    uav_local_vec = R_ENU_to_local @ (uav_course_enu * uav_speed)
    target_local_vec = R_ENU_to_local @ (target_course_enu * target_speed)

    s_hat = to_unit(target_local_vec) #目标航迹单位向量
    up = np.array([0.0, 0.0, 1.0]) #向上向量
    n_hat = to_unit(np.cross(up, s_hat)) #航迹左侧横向单位向量
    if np.linalg.norm(n_hat) < 1e-6:
        n_hat = np.array([1.0, 0.0, 0.0]) #极端情况：目标航迹几乎竖直（兜底给定一个水平向右向量）
    lateral_sign = +1 if lateral.lower() in ('left', 'l') else -1

    # 标量速度大小
    magnitude_target = np.linalg.norm(target_local_vec)
    magnitude_uav = np.linalg.norm(uav_local_vec)

    # 构建搜索集合（限制速度，限制到达时间）
    to_C_list = np.linspace(to_C_window[0], to_C_window[1], grid[0])  #1-2s
    to_CD_list = np.linspace(to_CD_window[0], to_CD_window[1], grid[1])  #3-5s
    speed_list = np.linspace(speed_range[0], speed_range[1], grid[2]) #10-15m/s

    # best_score评分，越早达到B越好，平均加速度越小越好
    best, best_score = None, (np.inf, np.inf, np.inf)
    best_tuple = None

    # 进行穷举，生成候选 C(B) 点与翼端 D 点，使用“软约束”快筛
    for to_C in to_C_list:
        for to_CD in to_CD_list:
            t_imp = to_C + to_CD

            # 撞击时刻敌机中心位置
            enemy_imp_local = enemy_local + target_local_vec * t_imp

            # 求机翼位置（敌机在 t_imp 的位置 + 机翼侧偏）
            point_D = enemy_imp_local + lateral_sign * 0.5 * wing_half * n_hat

            for speed in speed_list:
                # 末段匀速闭合，那么 B 到 D 路程 = 相对速度 × 时间
                distance_CD = speed * to_CD

                # 沿航迹方向从 D 回退相应距离，得到 C(B)
                point_C = point_D - distance_CD * s_hat

                # 几何退化检查（极端数值问题时跳过）
                if not np.isfinite(point_C).all():
                    continue

                # ===== 1. 时间可达性违约（软约束） =====
                min_time_needed = reachable_time(uav_local, magnitude_uav, point_C, a_max, v_max)
                vio_time = max(0.0, float(min_time_needed - to_C))  # 需要时间比规划 to_C 多了多少

                # ===== 2. 并行速度违约（软约束） =====
                required_uav_along_speed = magnitude_target + speed  # 我机沿 s_hat 分量 = 敌速 + 相对速度
                if v_max is not None:
                    vio_speed = max(0.0, float(required_uav_along_speed - v_max))
                else:
                    vio_speed = 0.0

                # ===== 3. 平均加速度违约（软约束） =====
                uav_along_speed_now = float(np.dot(uav_local_vec, s_hat))
                avg_acc_required = (required_uav_along_speed - uav_along_speed_now) / max(to_C, 1e-3)
                if a_max is not None:
                    vio_acc = max(0.0, float(avg_acc_required - a_max))
                else:
                    vio_acc = 0.0

                # ===== 4. 组合 penalty =====
                # 时间违约权重大，其次速度/加速度
                penalty = (
                        1000.0 * vio_time +
                        100.0 * vio_speed +
                        100.0 * vio_acc
                )

                # ===== 5. 评分：先看 penalty，再看到达 C 的时间，再看加速度大小 =====
                score = (penalty, to_C, abs(avg_acc_required))

                if score < best_score:
                    best_score = score
                    best_tuple = (
                        point_C,  # 0
                        point_D,  # 1
                        enemy_imp_local,  # 2 撞击时刻敌机中心
                        to_C,  # 3
                        to_CD,  # 4
                        speed,  # 5 v_imp
                        avg_acc_required,  # 6
                    )

    if best_tuple is None:
        return None

    # 当命中最优候选时，将结果打包并转回经纬度
    C_local, D_local, enemy_imp_local, t_c, t_cd, rel_v, avg_acc = best_tuple
    B_lat, B_lon, B_alt = converter.local_to_geodetic_dms(C_local)
    D_lat, D_lon, D_alt = converter.local_to_geodetic_dms(D_local)
    E_lat, E_lon, E_alt = converter.local_to_geodetic_dms(enemy_imp_local)

    return {
        "B_dms": [B_lat, B_lon, B_alt],
        "Wing_dms": [D_lat, D_lon, D_alt],
        "enemy_now_dms":[E_lat, E_lon, E_alt],
        # "tC": float(t_c),
        # "t_CD": float(t_cd),
        # "v_imp": float(rel_v),
        "avg_acc_required": float(avg_acc),
        # "notes": "尾后正向：B≡C；到C 1–2s，在C即达到相对速10–15；C→翼尖 3–5s 匀速闭合",
    }


# 尾后斜向接近路线控制
def upfind_single_uav_B_point(converter, R_ENU_to_local, uav_point, target_point, rear_point, uav_speed, uav_course_deg, target_speed, target_course_deg, wing_half: float,
                              rear_speed: float | None = None, rear_course_deg: float | None = None, lateral: str = 'right',
                              to_C_window = (1.0, 10.0), to_CD_window = (3.0, 5.0), v_imp_range = (10.0, 15.0),
                              alpha_small_deg_range = (2.0, 8.0), alpha_large_min_deg = 15.0,
                              a_max: float | None = None, v_max: float | None = None,
                              max_dist_to_front = 45.0, rear_frac_min = 1/3, grid = (15, 7, 5, 5)) -> Optional[Dict[str, Any]]:
    # print("to_C_window", to_C_window)
    # 先将坐标（当前无人机点，目标点，后方敌机点）及速度转为local
    pointA = dms_to_local(converter, uav_point)
    pointF = dms_to_local(converter, target_point)
    pointR = dms_to_local(converter, rear_point)

    # print("目标高度：", target_point[2])

    uav_dir_enu = bearing_to_enu_unit(uav_course_deg)
    front_dir_enu = bearing_to_enu_unit(target_course_deg)
    rear_dir_enu = bearing_to_enu_unit(rear_course_deg if rear_course_deg is not None else target_course_deg)

    speedA = R_ENU_to_local @ (uav_dir_enu * uav_speed)
    speedF = R_ENU_to_local @ (front_dir_enu * target_speed)
    speedR = R_ENU_to_local @ (rear_dir_enu * (rear_speed if rear_speed is not None else target_speed))

    s_hat = to_unit(speedF)
    up = np.array([0.0, 0.0, 1.0])
    n_hat_left = to_unit(np.cross(up, s_hat))
    if np.linalg.norm(n_hat_left) < 1e-6:
        n_hat_left = np.array([1.0, 0.0, 0.0])
    lateral_sign = +1 if lateral.lower() in ('left', 'l') else -1
    n_hat = lateral_sign * n_hat_left

    speed_target = np.linalg.norm(speedF)
    uav_speed_along_track_now = float(np.dot(speedA, s_hat)) #当前沿着目标航迹的速度分量
    uav_speed_now = float(np.linalg.norm(speedA))

    # 创建搜索网格
    to_C_list = np.linspace(to_C_window[0], to_C_window[1], grid[0])
    to_CD_list = np.linspace(to_CD_window[0], to_CD_window[1], grid[1])
    vimp_list = np.linspace(v_imp_range[0], v_imp_range[1], grid[2])
    deg_list = np.linspace(alpha_small_deg_range[0], alpha_small_deg_range[1], grid[3])

    best, best_score = None, (np.inf, np.inf, np.inf, np.inf)
    best_tuple = None

    # 进行穷举，生成D点，并倒推C点进行“软约束”快筛
    for to_C in to_C_list:
        for to_CD in to_CD_list:
            t_imp = to_C + to_CD

            # 敌机front及敌机rear在撞机时刻的位置
            pointF_now = pointF + speedF * t_imp
            pointR_now = pointR + speedR * t_imp

            # 敌机前后沿程间距
            linear_spacing = float(np.dot(pointF_now - pointR_now, s_hat))  # 敌机前后间距
            if linear_spacing <= 0:
                continue  # 这算几何退化，直接跳过

            # 对准点要放在前机与后机两个敌机之间，max(间距/3, 间距-45)
            distance_lower = max(linear_spacing * rear_frac_min, linear_spacing - max_dist_to_front)
            if distance_lower > linear_spacing:
                continue  # 这也是几何配置不合理，直接跳过

            # 确定对准点的位置，但不强行把它放入轨迹几何中，只用来做末端校核与记录
            distance_aim = 0.5 * (distance_lower + linear_spacing)
            pAim_axis = pointR_now + distance_aim * s_hat  # 轴线上对准点

            # 撞机点D（翼中点）
            pointD = pointF_now + lateral_sign * 0.5 * wing_half * n_hat

            for v_imp in vimp_list:
                s_CD = v_imp * to_CD  # 末段路程闭合长度

                for deg_small in deg_list:
                    deg = np.deg2rad(deg_small)

                    # 斜角过近 90° 的几何退化仍然用硬过滤
                    if np.cos(deg) < 1e-6:
                        continue

                    g_hat = to_unit(np.cos(deg) * s_hat + np.sin(deg) * (-up))

                    # 有D点后倒推C点，确保C在s_hat上的投影为s_CD
                    pointC = pointD - (s_CD / np.cos(deg)) * g_hat
                    if not np.isfinite(pointC).all():
                        continue  # 极端数值情况，跳过

                    # ===== 1. 时间可达性违约（软约束） =====
                    time_lower = reachable_time(pointA, uav_speed_now, pointC, a_max, v_max)
                    vio_time = max(0.0, float(time_lower - to_C))  # 需要时间比窗口多了多少

                    # ===== 2. 并行速度违约（软约束） =====
                    required_uav_along_speed = speed_target + v_imp  # 敌速 + 相对速度
                    if v_max is not None:
                        vio_speed = max(0.0, float(required_uav_along_speed - v_max))
                    else:
                        vio_speed = 0.0

                    # ===== 3. 平均加速度违约（软约束） =====
                    avg_acc_required = (required_uav_along_speed - uav_speed_along_track_now) / max(to_C, 1e-3)
                    if a_max is not None:
                        vio_acc = max(0.0, float(avg_acc_required - a_max))
                    else:
                        vio_acc = 0.0

                    # ===== 4. 初段斜角违约（软约束） =====
                    vec_now_to_C = pointC - pointA
                    if np.linalg.norm(vec_now_to_C) < 1e-6:
                        continue  # A 和 C 几乎重合，几何退化

                    angle_now_to_C_deg = float(
                        np.rad2deg(np.arccos(
                            np.clip(np.dot(to_unit(vec_now_to_C), s_hat), -1.0, 1.0)
                        ))
                    )
                    vio_angle = max(0.0, float(alpha_large_min_deg - angle_now_to_C_deg))

                    # ===== 5. 对准点区间违约（软约束） =====
                    dist_from_rear = distance_aim
                    dist_to_front = linear_spacing - distance_aim

                    # 原始约束：
                    #   dist_to_front <= max_dist_to_front
                    #   dist_from_rear >= linear_spacing * rear_frac_min
                    vio_front = max(0.0, float(dist_to_front - max_dist_to_front))
                    vio_rear = max(0.0, float(linear_spacing * rear_frac_min - dist_from_rear))

                    # ===== 6. 组合 penalty =====
                    # 时间违约权重最高，其次速度/加速度，再其次角度/对准点
                    penalty = (
                            1000.0 * vio_time +
                            100.0 * vio_speed +
                            100.0 * vio_acc +
                            10.0 * vio_angle +
                            5.0 * (vio_front + vio_rear)
                    )

                    # ===== 7. 评分：先看 penalty，再看到达 C 的时间、再看加速度/末段小角 =====
                    score = (penalty, to_C, abs(avg_acc_required), deg_small)

                    if score < best_score:
                        best_score = score
                        best_tuple = (
                            pointC, pointD, pAim_axis, pointF_now,
                            to_C, to_CD, v_imp,
                            deg_small, avg_acc_required,
                            dist_to_front, dist_from_rear, linear_spacing,
                            angle_now_to_C_deg,
                        )

    if best_tuple is None:
        return None

    # 打包输出，转回dms

    (pointC, pointD, pAim_axis,pointF_now,
     to_C, to_CD, v_imp,
     alpha_deg_small, avg_acc_required,
     dist_to_front, dist_from_rear, linear_spacing,
     angle_now_to_C_deg) = best_tuple


    B_lat, B_lon, B_alt = converter.local_to_geodetic_dms(pointC)
    D_lat, D_lon, D_alt = converter.local_to_geodetic_dms(pointD)
    Aim_lat, Aim_lon, Aim_alt = converter.local_to_geodetic_dms(pAim_axis)
    F_lat, F_lon, F_alt = converter.local_to_geodetic_dms(pointF_now)

    return {
        "B_dms": [B_lat, B_lon, target_point[2]],  # 元组可以直接用 list，JSON 会是数组
        "Wing_dms": [D_lat, D_lon, target_point[2]],
        "enemy_now_dms":[F_lat, F_lon, target_point[2]],
        # "Aim_dms": [Aim_lat, Aim_lon, Aim_alt],
        # "tC": float(to_C),
        # "t_CD": float(to_CD),
        # "v_imp": float(v_imp),
        "alpha_small_deg": float(alpha_deg_small),
        "avg_acc_required": float(avg_acc_required),
        # "dist_aim_to_front": float(dist_to_front),
        # "dist_aim_from_rear": float(dist_from_rear),
        # "spacing_front_rear": float(linear_spacing),
        "angle_now_to_C_deg": float(angle_now_to_C_deg),
        # "notes": "尾后斜向（仅下劈）：B≡C；末段小斜角；对准点落在前/后机之间且满足 ≤45m 与 ≥间距1/3。"
    }


def rear_dms_if_missing(
    converter, R_ENU_to_local,
    front_dms_list,
    enemy_course_deg: float,
    rear_spacing_m: float = 90.0
) -> List[tuple]:

    # 因为敌机后方无人机不好去匹配其dms且最后一排没有后方无人机，就根据间距去算一个后方敌机的位置
    s_hat = to_unit(R_ENU_to_local @ (bearing_to_enu_unit(enemy_course_deg) * 1.0))
    outs = []
    for f in front_dms_list:
        pF = dms_to_local(converter, f)
        pR = pF - float(rear_spacing_m) * s_hat
        outs.append(converter.local_to_geodetic_dms(pR))
    return outs

# 一对一地进行撞机，需要给出撞击类型
def assign_by_index_simple(
    converter, R_ENU_to_local,
    uav_dms_list,
    enemy_dms_list,
    uav_num,

    # 撞击类型：'tail' / 'oblique'
    impact_type: str = 'tail',
    lateral: str = 'right',

    # 统一速度/航向
    uav_speed: float = 90.0,
    uav_course_deg: float = 270.0,
    enemy_speed: float = 90.0,
    enemy_course_deg: float = 270.0,

    # 机翼半长
    wing_half: float = 5.0,

    # 敌群间距
    rear_spacing_m: float = 90.0,
    # 动力学包线
    a_max: float = 80,
    v_max: float = 500,

    # 直线（尾后）窗口
    to_C_window_tail=(6.0, 8.0),
    to_CD_window_tail=(3.0, 5.0),
    vimp_range_tail=(10.0, 15.0),
    grid_tail=(21, 9, 7),

    # 斜向窗口
    to_C_window_obl=(10.0, 20.0),
    to_CD_window_obl=(3.0, 5.0),
    vimp_range_obl=(10.0, 15.0),
    alpha_small_deg_range=(2.0, 8.0),
    alpha_large_min_deg=15.0,
    max_dist_to_front=45.0,
    rear_frac_min=1/3,
    grid_obl=(15, 7, 5, 5),
) -> List[Dict[str, Any]]:


    # impact_type 统一展开为列表
    if isinstance(impact_type, str):
        types = [impact_type] * uav_num
    else:
        types = impact_type
        assert len(types) >= uav_num, "impact_type 列表长度不足"

    # 斜向需要后方无人机的坐标或者说距离，直接生成
    rear_dms_list = rear_dms_if_missing(
        converter, R_ENU_to_local,
        enemy_dms_list, enemy_course_deg,
        rear_spacing_m
    )

    auto_to_C = estimate_auto_tC_window(
        converter, R_ENU_to_local,
        uav_point=uav_dms_list[0],
        target_point=enemy_dms_list[0],
        uav_speed=uav_speed,
        uav_course_deg=uav_course_deg,
        target_speed=enemy_speed,
        target_course_deg=enemy_course_deg,
        a_max=a_max,
        v_max=v_max,
        # 这些 scale / cap 可以按需要调
        scale_low=0.8,
        scale_high=1.5,
        t_min_cap=3.0,  # 近距离也不要小于 3s
        t_max_cap=300.0,  # 最长考虑到 5 min 左右，可按需求改
    )
    print("每次得到的时间窗口:", auto_to_C)
    # auto_to_C = (10,30)

    outs: List[Dict[str, Any]] = []
    for i in range(uav_num):
        method = str(types[i]).lower()
        res = None
        reason = None

        # print("当前无人机坐标",uav_dms_list[i],"对应的目标坐标：",enemy_dms_list[i])



        try:
            if method in ('tail', 'straight', 'rear', '尾后', '直线'):
                # —— 尾后正向（B≡C）——
                res = find_single_uav_B_point(
                    converter, R_ENU_to_local,
                    uav_point=uav_dms_list[i], target_point=enemy_dms_list[i],
                    wing_half=wing_half, lateral=lateral,
                    uav_speed=uav_speed, uav_course_deg=uav_course_deg,
                    target_speed=enemy_speed, target_course_deg=enemy_course_deg,
                    to_C_window=auto_to_C,
                    to_CD_window=to_CD_window_tail,
                    speed_range=vimp_range_tail,   # 直线版参数名是 speed_range
                    a_max=a_max, v_max=v_max,
                    grid=grid_tail
                )
                used_method = 'tail'

            elif method in ('oblique', 'slant', '尾后斜后方', '斜向'):
                # —— 尾后斜向（仅下劈）——
                res = upfind_single_uav_B_point(
                    converter, R_ENU_to_local,
                    uav_point=uav_dms_list[i],
                    target_point=enemy_dms_list[i],
                    rear_point=rear_dms_list[i],
                    uav_speed=uav_speed, uav_course_deg=uav_course_deg,
                    target_speed=enemy_speed, target_course_deg=enemy_course_deg,
                    wing_half=wing_half, lateral=lateral,
                    rear_speed=enemy_speed, rear_course_deg=enemy_course_deg,
                    to_C_window=auto_to_C,
                    to_CD_window=to_CD_window_obl,
                    v_imp_range=vimp_range_obl,            # 斜向版参数名是 v_imp_range
                    alpha_small_deg_range=alpha_small_deg_range,
                    alpha_large_min_deg=alpha_large_min_deg,
                    a_max=a_max, v_max=v_max,
                    max_dist_to_front=max_dist_to_front,
                    rear_frac_min=rear_frac_min,
                    grid=grid_obl
                )
                used_method = 'oblique'
            else:
                used_method = 'none'
                reason = f"未知撞机类型: {types[i]}（应为'tail'或'oblique'）"

        except Exception as ex:
            used_method = 'none'
            reason = f"调用异常: {ex}"

        if res is None and reason is None:
            reason = "无可行解（多半是到C（B）的时间可达性或对准点区间未满足）"

        outs.append({
            'idx': i,
            'method': used_method if res is not None else 'none',
            'result': res,        # 若直线/斜向
            # 'reason': reason      # 无解时的原因提示
        })

    return outs


def estimate_auto_tC_window(
    converter,
    R_ENU_to_local,
    uav_point,
    target_point,
    uav_speed,
    uav_course_deg,
    target_speed,
    target_course_deg,
    a_max,
    v_max,
    scale_low: float = 0.8,
    scale_high: float = 1.5,
    t_min_cap: float = 3.0,
    t_max_cap: float = 300.0,
):
    """
    根据当前敌我 geometry 估一个合适的 to_C_window:
    - 先估计在 a_max, v_max 限制下到达“目标附近”的最小时间作为 t_min
    - 再给一个比例范围 [0.8 t_min, 1.5 t_min]，并用 t_min_cap / t_max_cap 进行裁剪
    """

    uav_local = dms_to_local(converter, uav_point)
    enemy_local = dms_to_local(converter, target_point)

    # 我方当前速度向量
    uav_course_enu = bearing_to_enu_unit(uav_course_deg)
    uav_local_vec = R_ENU_to_local @ (uav_course_enu * uav_speed)
    now_speed = float(np.linalg.norm(uav_local_vec))

    # 这里用 “直接飞到敌机当前点/附近” 的时间做下界估计
    t_min = reachable_time(uav_local, now_speed, enemy_local, a_max, v_max)

    # 给出自适应窗口
    t_low = max(t_min_cap, scale_low * t_min)
    t_high = min(t_max_cap, scale_high * t_min)

    # 避免 low >= high 的极端情况
    if t_high <= t_low:
        t_high = t_low + 1.0

    return (t_low, t_high)



# 撞机总控制函数
def impact_control(
        enemy_dms,
        uav_dms,
        enemy_center,
        uav_center,
        enemy_course_deg,
        uav_course_deg,
        uav_speed,
        enemy_speed,
        uav_num,
        wing_half,
        rear_spacing_m,
        a_max,
        v_max,
        impact_type: str = 'tail',
        lateral: str = 'right',
):
    # 先判断敌群和我方中心是否给出，如果没有，那就直接计算
    print("uav_center",uav_center)
    enemy_center = judge_enemy_center(enemy_center, enemy_dms)
    uav_center = judge_enemy_center(uav_center, uav_dms)
    print("uav_center2",uav_center)

    # 构建全局坐标转换converter
    converter, R_local_to_ENU, R_ENU_to_local = build_global_converter(uav_center, enemy_center)

    info = assign_by_index_simple(
            converter, R_ENU_to_local,
            uav_dms,
            enemy_dms,
            uav_num,

            # 撞击类型：'tail' / 'oblique'
            impact_type,
            lateral,

            # 统一速度/航向
            uav_speed,
            uav_course_deg,
            enemy_speed,
            enemy_course_deg,

            # 机翼半长
            wing_half,

            # 敌群间距(前后间距)
            rear_spacing_m,

            # 最大速度和最大加速度
            a_max,
            v_max,

            # 直线（尾后）窗口, 窗口值均为默认值，便于后续调整
            to_C_window_tail = (10.0, 30.0),
            to_CD_window_tail = (1.0, 3.0),
            vimp_range_tail = (10.0, 15.0),
            grid_tail = (11, 5, 4),

            # 斜向窗口，窗口值均为默认值，便于后续调整
            to_C_window_obl = (10.0, 30.0),
            to_CD_window_obl = (1.0, 3.0),
            vimp_range_obl = (10.0, 60.0),
            alpha_small_deg_range = (2.0, 8.0),
            alpha_large_min_deg = 15.0,
            max_dist_to_front = 45.0,
            rear_frac_min = 1 / 3,
            grid_obl = (9, 5, 4, 4)
    )

    return info



# =================================================================================================
# Demo：两类B点求解的最小可运行测例
# =================================================================================================

def _line():
    print("-" * 72)

def _print_straight(res, v_tgt, v_uav):
    s_cd = res.v_imp * res.t_CD
    print(f"B(C):        {res.B_dms}")
    print(f"D(翼):       {res.Wing_dms}")
    print(f"tC={res.tC:.2f}s  T_CD={res.t_CD:.2f}s  v_imp={res.v_imp:.1f}  avg_acc={res.need_avg_acc:.2f}")
    print(f"C→D 沿程闭合≈ {s_cd:.1f} m   C处并行速度需={v_tgt + res.v_imp:.1f} m/s")
    print(f"备注：{res.notes}")

def _print_oblique(res, v_tgt, v_uav):
    s_cd = res.v_imp * res.t_CD
    print(f"B(C):        {res.B_dms}")
    print(f"D(翼):       {res.Wing_dms}")
    print(f"Aim(轴):     {res.Aim_dms}")
    print(f"tC={res.tC:.2f}s  T_CD={res.t_CD:.2f}s  v_imp={res.v_imp:.1f}  α_small={res.alpha_small_deg:.1f}°  avg_acc={res.avg_acc_required:.2f}")
    print(f"C→D 沿程闭合≈ {s_cd:.1f} m   C处并行速度需={v_tgt + res.v_imp:.1f} m/s")
    print(f"对准点校核：  前向距={res.dist_aim_to_front:.1f} m(≤45)  后向距={res.dist_aim_from_rear:.1f} m(≥间距1/3)  间距={res.spacing_front_rear:.1f} m")
    print(f"初段斜角：    now→C = {res.angle_now_to_C_deg:.1f}°")
    print(f"备注：{res.notes}")

def place_uav_behind_target(converter, R_ENU_to_local, target_point, target_course_deg, back_distance_m=600.0):
    target_local = dms_to_local(converter, target_point)
    s_hat = to_unit(R_ENU_to_local @ (bearing_to_enu_unit(target_course_deg) * 1.0))
    uav_local = target_local - back_distance_m * s_hat
    u_lat, u_lon, u_alt = converter.local_to_geodetic_dms(uav_local)
    return [u_lon, u_lat, u_alt]



# ======================================== 测例 1：尾后正向（B≡C） =================================================
def demo_tail_straight():
    print("\n[测例1｜尾后正向（B≡C）]")
    # —— 坐标系（以敌我中心建立） ——
    own_center   = ['125:24:00.00E','26:37:00.00N','5000.00']
    enemy_center = ['125:28:00.00E','26:38:00.00N','5650.00']
    converter, R_local_to_ENU, R_ENU_to_local = build_global_converter(own_center, enemy_center)

    # —— 目标点与我机放置（我机在目标后方50 m） ——
    target_dms   = ['125:28:32.08E','26:37:59.28N','5658.62']
    course_deg   = 270.0
    v_tgt, v_uav = 60.0, 90.0
    pT0 = dms_to_local(converter, target_dms)
    s_hat = to_unit(R_ENU_to_local @ (bearing_to_enu_unit(course_deg) * 1.0))
    pA0 = pT0 - 50.0 * s_hat
    uav_dms = converter.local_to_geodetic_dms(pA0)
    print("用向量得到的uav_dms:", uav_dms)

    # —— 调用（5–6s到C点，撞机：3–3.5s 末段：12–15 m/s相对闭合） ——
    res = find_single_uav_B_point(
        converter, R_ENU_to_local,
        uav_point=uav_dms, target_point=target_dms,
        wing_half=8.0, lateral='right',
        uav_speed=v_uav, uav_course_deg=course_deg,
        target_speed=v_tgt, target_course_deg=course_deg,
        to_C_window=(5.0, 6.0), to_CD_window=(3.5, 4.5), speed_range=(15.0, 20.0),
        a_max=80.0, v_max=500.0, grid=(21, 7, 7)
    )

    if res is None:
        print("无可行解：可尝试增加tC或减小初始落后敌机速度。")
    else:
        _print_straight(res, v_tgt, v_uav)

# =============== 测例 2：尾后斜向（上方→小角下劈） ===============
def demo_tail_oblique():
    print("\n【测例2｜尾后斜向（上方下劈）】")
    # —— 坐标系 ——
    own_center   = ['125:24:00.00E','26:37:00.00N','5000.00']
    enemy_center = ['125:28:00.00E','26:38:00.00N','5650.00']
    converter, R_local_to_ENU, R_ENU_to_local = build_global_converter(own_center, enemy_center)

    # —— 前/后机、我机初始（前后间距100m；我机后方40 m、上方60 m） ——
    front_dms  = ['125:28:32.08E','26:37:59.28N','5658.62']
    course_deg = 270.0
    v_front, v_uav = 40.0, 90.0
    pF0 = dms_to_local(converter, front_dms)
    s_hat = to_unit(R_ENU_to_local @ (bearing_to_enu_unit(course_deg) * 1.0))
    pR0 = pF0 - 90.0 * s_hat
    rear_dms = converter.local_to_geodetic_dms(pR0)
    pA0 = pF0 - 40.0 * s_hat + np.array([0.0, 0.0, 60.0])
    uav_dms = converter.local_to_geodetic_dms(pA0)
    print("平时用的时候uav_dms:", uav_dms, "还有后方的：", rear_dms)

    # —— 调用（tC≈3.0–3.3；T_CD≈3.0–3.2；v_imp=15；小角 2–4°） ——
    res = upfind_single_uav_B_point(
        converter, R_ENU_to_local,
        uav_point=uav_dms, target_point=front_dms, rear_point=rear_dms,
        uav_speed=v_uav, uav_course_deg=course_deg,
        target_speed=v_front, target_course_deg=course_deg,
        wing_half=8.0, lateral='right',
        to_C_window=(3.0, 3.3), to_CD_window=(3.0, 3.2), v_imp_range=(15.0, 15.0),
        alpha_small_deg_range=(2.0, 4.0), alpha_large_min_deg=10.0,
        a_max=80.0, v_max=500.0,   # 这里 a_max 用大些，便于通过时间快筛
        max_dist_to_front=45.0, rear_frac_min=1/3, grid=(15, 7, 5, 5)
    )

    if res is None:
        print("无可行解：可尝试增加tC或减小初始落后敌机速度")
    else:
        _print_oblique(res, v_front, v_uav)



# ==================================================== 输入输出 =========================================================
# 获取输入接口信息
def api_main(config: Dict[str, Any]) -> dict:

    # 获取输入参数
    enemy_dms = config["enemy_dms"]
    uav_dms = config["uav_dms"]
    enemy_center = config.get("enemy_center",None)
    uav_center = config.get("uav_center",None)
    enemy_course_deg = config.get("enemy_course_deg")
    uav_course_deg = config.get("uav_course_deg")
    uav_speed = config.get("uav_speed")
    enemy_speed = config.get("enemy_speed")
    uav_num = config.get("uav_num")
    wing_half = config.get("wing_half")
    rear_spacing_m = config.get("rear_spacing_m")
    a_max = config.get("a_max")
    v_max = config.get("v_max")
    impact_type = config.get("impact_type")
    lateral = config.get("lateral")


    info = impact_control(
        enemy_dms,
        uav_dms,
        enemy_center,
        uav_center,
        enemy_course_deg,
        uav_course_deg,
        uav_speed,
        enemy_speed,
        uav_num,
        wing_half,
        rear_spacing_m,
        a_max,
        v_max,
        impact_type,
        lateral,

    )

    return info


def run_port(config: dict | None = None) -> dict:

    if config is None:
        config = test_config

    # 获取输入参数
    enemy_dms = config["enemy_dms"]
    uav_dms = config["uav_dms"]
    enemy_center = config.get("enemy_center", None)
    uav_center = config.get("uav_center", None)
    enemy_course_deg = config.get("enemy_course_deg")
    uav_course_deg = config.get("uav_course_deg")
    uav_speed = config.get("uav_speed")
    enemy_speed = config.get("enemy_speed")
    uav_num = config.get("uav_num")
    wing_half = config.get("wing_half")
    rear_spacing_m = config.get("rear_spacing_m")
    a_max = config.get("a_max")
    v_max = config.get("v_max")
    impact_type = config.get("impact_type")
    lateral = config.get("lateral")

    info = impact_control(
        enemy_dms,
        uav_dms,
        enemy_center,
        uav_center,
        enemy_course_deg,
        uav_course_deg,
        uav_speed,
        enemy_speed,
        uav_num,
        wing_half,
        rear_spacing_m,
        a_max,
        v_max,
        impact_type,
        lateral,

    )
    print(info[0])

    all_B_dms = [
        item["result"]["B_dms"]
        for item in info
        if item.get("result") is not None
    ]

    # 所有 Wing_dms
    all_Wing_dms = [
        item["result"]["Wing_dms"]
        for item in info
        if item.get("result") is not None
    ]

    lons, lats, alts = [], [], []

    for dms in all_Wing_dms:
        lon, lat, alt = dms_to_degree(dms)
        lons.append(lon)
        lats.append(lat)
        alts.append(alt)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(lons, lats, alts, s=5)  # 默认颜色即可

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("Enemy Formation (3D scatter)")

    plt.show()

    return info

# —— 运行两个测例 ——
if __name__ == "__main__":
    # _line(); demo_tail_straight()
    # _line(); demo_tail_oblique()
    run_port()

























