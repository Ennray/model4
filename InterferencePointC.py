from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import GeodeticConverter as GC

# —— 你已有的工具：dms_to_local / bearing_to_enu_unit / to_unit / build_global_converter ——

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

#===========================以敌我中心构建全局坐标系converter===============================
def build_global_converter(uav_center, enemy_center):
    uav_lat, uav_lon, uav_alt = dms_to_degree(uav_center)
    enemy_lat, enemy_lon, enemy_alt = dms_to_degree(enemy_center)
    converter = GC.GeodeticToLocalConverter(uav_lat, uav_lon, uav_alt, enemy_lat, enemy_lon, enemy_alt)
    R_local_to_ENU = converter.rotation_matrix
    R_ENU_to_local = R_local_to_ENU.T

    return converter, R_local_to_ENU, R_ENU_to_local

@dataclass
class CCenter:
    # 行列索引（基于“航迹方向=行、侧向=列”的网格）
    row_idx: int
    col_idx: int
    # C点的local坐标
    C_local: np.ndarray
    # C点的dms
    C_dms: tuple


def _estimate_sn_axes_and_spacing(
    R_ENU_to_local: np.ndarray,
    enemy_course_deg: float,
    enemy_local: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """
    基于敌群统一航迹角，估计：
      - track_dir_unit: 航迹方向单位向量（s_hat）
      - lateral_dir_unit: 侧向单位向量（n_hat = up × s_hat）
      - step_along_track: 航迹向相邻点的步长（ds）
      - step_across_track: 侧向相邻点的步长（dn）
      - sn_coords: 每架敌机在 (s,n) 平面的投影坐标 (N,2)
    """
    # 航迹方向（local）
    track_dir_unit = to_unit(R_ENU_to_local @ (bearing_to_enu_unit(enemy_course_deg) * 1.0))
    up_vec = np.array([0.0, 0.0, 1.0])
    lateral_dir_unit = to_unit(np.cross(up_vec, track_dir_unit))
    if np.linalg.norm(lateral_dir_unit) < 1e-6:
        # 极端退化：给一个固定侧向
        lateral_dir_unit = np.array([1.0, 0.0, 0.0])

    # 投到统一平面，便于进行覆盖分布
    s_vals = enemy_local @ track_dir_unit
    n_vals = enemy_local @ lateral_dir_unit
    sn_coords = np.stack([s_vals, n_vals], axis=1)

    # 用最近邻差的中位数估步长
    def _median_step(vals: np.ndarray) -> float:
        vals_sorted = np.sort(vals)
        diffs = np.diff(vals_sorted)
        diffs = diffs[diffs > 1e-3]  # 去掉极小抖动
        return float(np.median(diffs)) if len(diffs) else 100.0  # 兜底 50 m

    step_along_track = _median_step(s_vals)
    step_across_track = _median_step(n_vals)
    return (track_dir_unit, lateral_dir_unit,
            step_along_track, step_across_track, sn_coords)


def _assign_grid_indices(sn_coords: np.ndarray, step_along_track: float, step_across_track: float) -> np.ndarray:
    """
    把 (s,n) 连续坐标映射为近似网格整数坐标 (row_idx, col_idx)：
      row_idx ≈ round(s / ds), col_idx ≈ round(n / dn)
    """
    row_idx = np.round(sn_coords[:, 0] / max(step_along_track, 1e-3)).astype(int)
    col_idx = np.round(sn_coords[:, 1] / max(step_across_track, 1e-3)).astype(int)
    return np.stack([row_idx, col_idx], axis=1)


def plan_C_centers_by_stripes(
    # 坐标转换输入
    converter,
    R_ENU_to_local: np.ndarray,
    # 敌群 DMS 列表（经纬高），统一航迹角（度）
    enemy_dms_list: List[list | tuple],
    enemy_course_deg: float,
    # 选择的条带（列）数量；类似“在 y=1/2/3 这 3 列上每行都放 C”
    stripes: int = 3,
    # C 点高度：用“敌群平均高度 + h_over”
    h_over: float = 120.0,
) -> List[CCenter]:
    """
    按“条带/列”排布干扰圆心 C：
      1) 用敌群坐标估计航迹/侧向轴与网格步长
      2) 将敌机映射到行(row)与列(col)
      3) 在选定的若干列上，对每一行放置一个 C
      4) C 的高度 = 敌群平均高度 + h_over

    返回：每个 C 的行列索引与 local/DMS 坐标
    """
    # 敌机dms坐标转local
    enemy_local_list = [dms_to_local(converter, d) for d in enemy_dms_list]
    enemy_local = np.vstack(enemy_local_list)  # (N,3)

    # 得到航向单位向量，侧向单位向量，沿航迹方向的网格步长以及侧向网格步长，去除高后的二维敌机坐标
    (track_dir_unit, lateral_dir_unit,
     step_along_track, step_across_track, sn_coords) = _estimate_sn_axes_and_spacing(
        R_ENU_to_local, enemy_course_deg, enemy_local
    )

    # 按步长取整，并记录所有出现过的行号以及当前出现过的列号
    grid_rc = _assign_grid_indices(sn_coords, step_along_track, step_across_track)  # (N,2)
    row_ids = np.unique(grid_rc[:, 0])
    col_ids = np.unique(grid_rc[:, 1])

    # 对r行取敌机s分量的平均值，对c列取敌机n分量的平均值
    row_to_s_value: Dict[int, float] = {}
    col_to_n_value: Dict[int, float] = {}
    for r in row_ids:
        row_to_s_value[int(r)] = float(sn_coords[grid_rc[:, 0] == r, 0].mean())
    for c in col_ids:
        col_to_n_value[int(c)] = float(sn_coords[grid_rc[:, 1] == c, 1].mean())

    # 在现有列号col_ids的范围内，均匀挑出stripes条列作为条带，这样覆盖更均匀，重叠更可控
    c_min, c_max = int(col_ids.min()), int(col_ids.max())
    if stripes >= len(col_ids):
        selected_cols = list(col_ids.astype(int))
    else:
        # 在 [c_min, c_max] 均匀采 stripes 个整数列；若去重后不足，再就近补齐
        seeds = np.linspace(c_min, c_max, stripes)
        selected_cols = sorted(set(int(round(v)) for v in seeds))
        all_cols = list(range(c_min, c_max + 1))
        i = 0
        while len(selected_cols) < stripes and i < len(all_cols):
            if all_cols[i] not in selected_cols:
                selected_cols.append(all_cols[i])
            i += 1
        selected_cols = sorted(selected_cols[:stripes])

    # —— 6) 生成所有 C（每一行 × 被选列） ——
    C_altitude = float(enemy_local[:, 2].mean() + h_over)  # 敌群平均高度 + 抬高
    out_list: List[CCenter] = []
    for r in sorted(row_ids):
        s_on_row = row_to_s_value[int(r)]
        for c in selected_cols:
            n_on_col = col_to_n_value[int(c)]
            # (s,n) → local 平面：XY = s*s_hat[:2] + n*n_hat[:2]
            C_xy = s_on_row * track_dir_unit[:2] + n_on_col * lateral_dir_unit[:2]
            C_local = np.array([C_xy[0], C_xy[1], C_altitude], dtype=float)
            C_dms = converter.local_to_geodetic_dms(C_local)
            out_list.append(
                CCenter(
                    row_idx=int(r),
                    col_idx=int(c),
                    C_local=C_local,
                    C_dms=C_dms
                )
            )

    return out_list