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


@dataclass
class ObliqueBPointResult:
    # 坐标
    pB_local: np.ndarray         # B(=C) 点 local
    pWing_local: np.ndarray      # D 点（翼中点）local
    pAim_local: np.ndarray       # 撞击对准点（轴线上）local
    B_dms: tuple                 # B 点 DMS
    Wing_dms: tuple              # D 点 DMS
    Aim_dms: tuple               # 对准点 DMS
    # 关键参数
    tC: float
    t_CD: float
    v_imp: float
    alpha_small_deg: float       # 末段小斜角（BC、CD 共用）
    avg_acc_required: float      # 平均加速度需求（沿航迹）
    # 约束校核
    dist_aim_to_front: float     # 对准点到前机的沿程距离（应 ≤ 45）
    dist_aim_from_rear: float    # 对准点到后机的沿程距离（应 ≥ 间距/3）
    spacing_front_rear: float    # 前后机沿程间距
    angle_now_to_C_deg: float    # 初段到C点的斜角
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


# 尾后斜向接近路线控制
def upfind_single_uav_B_point(converter, R_ENU_to_local, uav_point, target_point, rear_point, uav_speed, uav_course_deg, target_speed, target_course_deg, wing_half: float,
                              rear_speed: float | None = None, rear_course_deg: float | None = None, lateral: str = 'right',
                              to_C_window = (1.0, 2.0), to_CD_window = (3.0, 5.0), v_imp_range = (10.0, 15.0),
                              alpha_small_deg_range = (2.0, 8.0), alpha_large_min_deg = 15.0,
                              a_max: float | None = None, v_max: float | None = None,
                              max_dist_to_front = 45.0, rear_frac_min = 1/3, grid = (15, 7, 5, 5)) -> ObliqueBPointResult | None:
    # 先将坐标（当前无人机点，目标点，后方敌机点）及速度转为local
    pointA = dms_to_local(converter, uav_point)
    pointF = dms_to_local(converter, target_point)
    pointR = dms_to_local(converter, rear_point)

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

    best, best_score = None, (np.inf, np.inf, np.inf)
    best_tuple = None

    # 进行穷举，生成D点，并倒推C点进行快筛
    for to_C in to_C_list:
        for to_CD in to_CD_list:
            t_imp = to_C + to_CD

            # 敌机front及敌机rear在撞机时刻的位置
            pointF_now = pointF + speedF * t_imp
            pointR_now = pointR + speedR * t_imp

            # 敌机前后沿程间距
            linear_spacing = float(np.dot(pointF_now - pointR_now, s_hat)) # 敌机前后间距
            if linear_spacing <= 0:
                continue

            # 对准点要放在前机与后机两个敌机之间，max(间距/3, 间距-45)
            distance_lower = max(linear_spacing * rear_frac_min, linear_spacing - max_dist_to_front)
            if distance_lower > linear_spacing:
                continue

            # 确定对准点的位置，但不强行把它放入轨迹几何中，只用来做末端校核与记录
            distance_aim = 0.5 * (distance_lower + linear_spacing)
            pAim_axis = pointR_now + distance_aim * s_hat   # 轴线上对准点

            # 撞机点D
            pointD = pointF_now + lateral_sign * 0.5 * wing_half * n_hat

            for v_imp in vimp_list:
                s_CD = v_imp * to_CD #末段路程闭合长度

                for deg_small in deg_list:
                    deg = np.deg2rad(deg_small)

                    if np.cos(deg) < 1e-6:
                        continue  #将近90°那就不合理了
                    g_hat = to_unit(np.cos(deg) * s_hat + np.sin(deg) * (-up))

                    # 有D点后倒推C点，确保C在s_hat上的投影为s_CD
                    pointC = pointD - (s_CD / np.cos(deg)) * g_hat

                    # 快筛1确保时间内可到达（保守下界）
                    time_lower = reachable_time(pointA, uav_speed_now, pointC, a_max)
                    if time_lower > to_C:
                        continue

                    # 快筛2确保并行速度不会超过需求上限
                    required_uav_along_speed = speed_target + v_imp
                    if v_max is not None and required_uav_along_speed > v_max + 1e-6:
                        continue

                    # 快筛3取保平均加速度不会超过上限
                    avg_acc_required = (required_uav_along_speed - uav_speed_along_track_now) / max(to_C, 1e-3)
                    if a_max is not None and avg_acc_required > a_max + 1e-6:
                        continue

                    # 判断斜角，初段斜角应该大于拉飘段的斜角
                    vec_now_to_C = pointC - pointA
                    if np.linalg.norm(vec_now_to_C) < 1e-6:
                        continue

                    angle_now_to_C_deg = float(
                        np.rad2deg(np.arccos(
                            np.clip(np.dot(to_unit(vec_now_to_C), s_hat), -1.0, 1.0)
                        ))
                    )
                    if angle_now_to_C_deg < alpha_large_min_deg:
                        continue

                    dist_from_rear = distance_aim
                    dist_to_front = linear_spacing - distance_aim  #
                    if not (dist_to_front <= max_dist_to_front + 1e-6 and
                            dist_from_rear >= linear_spacing * rear_frac_min - 1e-6):
                        continue

                    # 评分：更早到 C 优先；其次更省力；再次末段斜角更小
                    score = (to_C, abs(avg_acc_required), deg_small)
                    if score < best_score:
                        best_score = score
                        best_tuple = (pointC, pointD, pAim_axis,
                                      to_C, to_CD, v_imp,
                                      deg_small, avg_acc_required,
                                      dist_to_front, dist_from_rear, linear_spacing,
                                      angle_now_to_C_deg)
    if best_tuple is None:
        return None

    # 打包输出，转回dms

    (pointC, pointD, pAim_axis,
     to_C, to_CD, v_imp,
     alpha_deg_small, avg_acc_required,
     dist_to_front, dist_from_rear, linear_spacing,
     angle_now_to_C_deg) = best_tuple


    B_lat, B_lon, B_alt = converter.local_to_geodetic_dms(pointC)
    D_lat, D_lon, D_alt = converter.local_to_geodetic_dms(pointD)
    Aim_lat, Aim_lon, Aim_alt = converter.local_to_geodetic_dms(pAim_axis)


    return ObliqueBPointResult(
        pB_local = pointC,
        pWing_local = pointD,
        pAim_local = pAim_axis,
        B_dms = (B_lat, B_lon, B_alt),
        Wing_dms = (D_lat, D_lon, D_alt),
        Aim_dms=(Aim_lat, Aim_lon, Aim_alt),
        tC = float(to_C),
        t_CD = float(to_CD),
        v_imp = float(v_imp),
        alpha_small_deg = float(alpha_deg_small),
        avg_acc_required = float(avg_acc_required),
        dist_aim_to_front = float(dist_to_front),
        dist_aim_from_rear = float(dist_from_rear),
        spacing_front_rear = float(linear_spacing),
        angle_now_to_C_deg = float(angle_now_to_C_deg),
        notes="尾后斜向（仅下劈）：B≡C；末段小斜角；对准点落在前/后机之间且满足 ≤45m 与 ≥间距1/3。"
    )



# =========================================
# Demo：两类 B 点求解的最小可运行测例
# 依赖：build_global_converter / dms_to_local / bearing_to_enu_unit /
#      find_single_uav_B_point / upfind_single_uav_B_point
# =========================================

# —— 通用的小工具，仅做美化输出 ——
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

# =============== 测例 1：尾后正向（B≡C） ===============
def demo_tail_straight():
    print("\n[测例1｜尾后正向（B≡C）]")
    # —— 坐标系（以敌我中心建立） ——
    own_center   = ['125:24:00.00E','26:37:00.00N','5000.00']
    enemy_center = ['125:28:00.00E','26:38:00.00N','5650.00']
    converter, R_local_to_ENU, R_ENU_to_local = build_global_converter(own_center, enemy_center)

    # —— 目标点与我机放置（我机在目标后方 40 m） ——
    target_dms   = ['125:28:32.08E','26:37:59.28N','5658.62']
    course_deg   = 270.0
    v_tgt, v_uav = 60.0, 90.0
    pT0 = dms_to_local(converter, target_dms)
    s_hat = to_unit(R_ENU_to_local @ (bearing_to_enu_unit(course_deg) * 1.0))
    pA0 = pT0 - 50.0 * s_hat
    uav_dms = converter.local_to_geodetic_dms(pA0)
    print("用向量得到的uav_dms:", uav_dms)

    # —— 调用（1–2 s 到 C；3–3.5 s 末段；12–15 m/s 相对闭合） ——
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

    # —— 前/后机、我机初始（前后间距 ~90 m；我机后方 40 m、上方 60 m） ——
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

# —— 运行两个测例 ——
if __name__ == "__main__":
    _line(); demo_tail_straight()
    _line(); demo_tail_oblique()
    _line()
























