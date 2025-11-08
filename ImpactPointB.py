from dataclasses import dataclass
import numpy as np
import GeodeticConverter as GC


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
def reachable_time(now_uav_point, now_uav_speed, target_point_B, a_max = None):
    # 到B点的直线距离
    distance = np.linalg.norm(target_point_B - now_uav_point)

    # 说明已经在目标点上
    if distance < 1e-6:
        return 0.0

    # 如果没有给加速度上限，那么用当前速度等速直达的时间作为估计
    if not a_max or a_max <= 0:
        return distance / max(now_uav_speed, 1e-3)

    # 如果给了加速度上限，那么用半程加速来进行保守估计
    half = 0.5 * distance
    time1 = half / max(now_uav_speed, 1e-3) #前半程按照当前速度跑
    time2 = np.sqrt(max(2*half/a_max, 0.0)) #后半程，按s = 0.5 * a * t ^ 2 反推时间
    return time1 + time2

@dataclass
class SingleBPointResult:
    # local 坐标结果
    pB_local: np.ndarray   # B 点(=C 点) 本地坐标
    pWing_local: np.ndarray # 敌机翼尖（撞击点）本地坐标 D
    # dmd 结果
    B_dms: tuple           # (lat, lon, alt)
    Wing_dms: tuple        # (lat, lon, alt)
    # 参数与校核
    tC: float              # 现在起到 C 点的时间（1–2 s）
    t_CD: float            # C 到 翼尖 的时间（3–5 s）
    v_imp: float           # 撞击相对速度（10–15 m/s）
    need_avg_acc: float    # 估计平均加速度需求（m/s^2）
    notes: str



#===========================以敌我中心构建全局坐标系converter===============================
def build_global_converter(uav_center, enemy_center):
    uav_lat, uav_lon, uav_alt = dms_to_degree(uav_center)
    enemy_lat, enemy_lon, enemy_alt = dms_to_degree(enemy_center)
    converter = GC.GeodeticToLocalConverter(uav_lat, uav_lon, uav_alt, enemy_lat, enemy_lon, enemy_alt)
    R_local_to_ENU = converter.rotation_matrix
    R_ENU_to_local = R_local_to_ENU.T

    return converter, R_local_to_ENU, R_ENU_to_local


def find_single_uav_B_point(converter, R_ENU_to_local, uav_point, target_point, wing_half, uav_speed, uav_course_deg, target_speed, target_course_deg, lateral:str = 'right',
                            to_C_window = (1.0, 2.0), to_CD_window = (3.0, 5.0), speed_range = (10.0, 15.0), a_max:float | None = None, v_max:float | None = None,  grid=(21, 9, 7)) ->SingleBPointResult | None:

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
    n_hat = to_unit(np.cross(up, s_hat)) #航迹左侧横向单位想了
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
    best, best_score = None, (np.inf, np.inf)
    best_tuple = None

    # 进行穷举，生成候选C（B）点与翼端D点，逐个快筛
    for to_C in to_C_list:
        for to_CD in to_CD_list:
            # 求机翼位置（敌机在to_C + to_CD的位置+机翼侧偏）
            point_D = enemy_local + target_local_vec * (to_C + to_CD) + lateral_sign * 0.5 * wing_half * n_hat

            for speed in speed_list:
                # 末段匀速闭合，那么B到D就等于速度×时间点距离
                distance_CD = speed * to_CD

                # 沿航迹方向从D回退相应距离，那么就能得到C（B）
                point_C = point_D - distance_CD * s_hat

                # 进行快筛，确保时间可达
                min_time_needed = reachable_time(uav_local, magnitude_uav, point_C, a_max)

                # 下界都超过了到达C点的时间，那么不可能达到
                if min_time_needed > to_C:
                    continue

                # 进行快筛2，求C（B）时刻的并行速度需求和速度上限
                required_uav_along_speed = magnitude_target + speed # # 我机沿 s_hat 分量 = 敌速 + 相对速度
                if v_max is not None and required_uav_along_speed > v_max + 1e-6:
                    continue

                # 进行快筛3，求平均加速度需求和加速度上限
                uav_along_speed_now = float(np.dot(uav_local_vec, s_hat))
                avg_acc_required = (required_uav_along_speed - uav_along_speed_now) / max(to_C, 1e-3)
                if a_max is not None and avg_acc_required > a_max + 1e-6:
                    continue

                # 进行评分：更早到C（B）点的优先，其次加速度小（更省力）的优先
                score = (to_C, abs(avg_acc_required))
                if score < best_score:
                    best_score = score
                    best_tuple = (point_C, point_D, to_C, to_CD, speed, avg_acc_required)


    if best_tuple is None:
        return None

    # 当命中最优候选时，将结果打包并转回经纬度
    C_local, D_local, t_c, t_cd, rel_v, avg_acc = best_tuple
    B_lat, B_lon, B_alt = converter.local_to_geodetic_dms(C_local)
    D_lat, D_lon, D_alt = converter.local_to_geodetic_dms(D_local)

    return SingleBPointResult(
        pB_local = C_local,
        pWing_local = D_local,
        B_dms = (B_lat, B_lon, B_alt),
        Wing_dms = (D_lat, D_lon, D_alt),
        tC = t_c,
        t_CD = t_cd,
        v_imp = float(rel_v),
        need_avg_acc = float(avg_acc),
        notes = "尾后正向：B≡C；到C 1–2s，在C即达到相对速10–15；C→翼尖 3–5s 匀速闭合"
    )

































