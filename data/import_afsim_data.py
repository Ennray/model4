# import_afsim_data.py
import json
import os
import re

def load_afsim_data():
    """从当前脚本所在目录查找 .txt(JSON) 文件并读取"""
    data_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"未找到数据目录: {data_dir}")
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    if not txt_files:
        raise FileNotFoundError("目录中未找到 .txt 数据文件")
    afsim_path = os.path.join(data_dir, txt_files[0])
    with open(afsim_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def dms_to_decimal(dms_str: str) -> float:
    """DMS -> 十进制度（E/N正, W/S负）"""
    d = dms_str[-1]
    val = dms_str[:-1]
    deg, minute, sec = val.split(':')
    x = float(deg) + float(minute)/60 + float(sec)/3600
    return -x if d in ('W', 'S') else x

def to_lon_lat_alt(uav_triplet):
    """输入 [lat, lon, alt] 或 [lon, lat, alt]，统一输出 [lon, lat, alt]（字符串）"""
    if len(uav_triplet) != 3:
        return ["", "", ""]
    a, b, c = uav_triplet
    # 通过末尾字母粗略判断
    if isinstance(a, str) and a.endswith(('N','S')) and isinstance(b, str) and b.endswith(('E','W')):
        # [lat, lon, alt] -> [lon, lat, alt]
        return [b, a, str(c)]
    return [str(a), str(b), str(c)]

def collect_second_groups(afsim_data: dict):
    """
    返回二维列表 second_groups：[[uav,uav,...], [uav,uav,...], ...]
    每个 uav 统一为 [lon, lat, alt]（字符串）
    兼容两种输入：
      1) afsim_data['second_uav_groups'] = [ [ [lat,lon,alt], ... ], [ ... ], ... ]
      2) 任意数量的键 second_uavs_<n>
    """
    groups = []

    # 优先支持更干净的结构：second_uav_groups
    if isinstance(afsim_data.get('second_uav_groups'), list):
        for grp in afsim_data['second_uav_groups']:
            if isinstance(grp, list):
                groups.append([to_lon_lat_alt(u) for u in grp])

    # 兼容旧键：second_uavs_<n>
    pattern = re.compile(r'^second_uavs_(\d+)$')
    temp = []
    for k, v in afsim_data.items():
        m = pattern.match(k)
        if m and isinstance(v, list):
            idx = int(m.group(1))
            temp.append((idx, [to_lon_lat_alt(u) for u in v]))
    if temp:
        temp.sort(key=lambda x: x[0])
        for _, grp in temp:
            groups.append(grp)

    return groups

def update_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    afsim_data = load_afsim_data()

    # 读取基础字段
    basepoint = afsim_data.get('basepoint', ["", "", ""])
    enemy_approx = afsim_data.get('enemy_approx', ["", "", ""])
    near_detect = afsim_data.get('min_detect', [900, 600])
    detect_distance = afsim_data.get('detect_distance', 30000)
    maximum_speed = afsim_data.get('maximum_speed', 300)
    minimum_speed = afsim_data.get('minimum_speed', 200)
    speed = afsim_data.get('speed', 240)
    acceleration = afsim_data.get('acceleration', 80)
    radii = afsim_data.get('radii', [])
    enemy_number = afsim_data.get('enemy_number', 300)

    enemy_lonrange = afsim_data.get('enemy_lonrange', [])
    enemy_latrange = afsim_data.get('enemy_latrange', [])

    # 统一范围从小到大
    if len(enemy_lonrange) == 2:
        if dms_to_decimal(enemy_lonrange[0]) > dms_to_decimal(enemy_lonrange[1]):
            enemy_lonrange = [enemy_lonrange[1], enemy_lonrange[0]]
    if len(enemy_latrange) == 2:
        if dms_to_decimal(enemy_latrange[0]) > dms_to_decimal(enemy_latrange[1]):
            enemy_latrange = [enemy_latrange[1], enemy_latrange[0]]

    # 第一批
    first_uavs_raw = afsim_data.get('first_uavs', [])
    first_uavs = [to_lon_lat_alt(u) for u in first_uavs_raw]
    first_num = afsim_data.get('first_num', len(first_uavs))

    # 第二批：动态纵队收集
    second_groups = collect_second_groups(afsim_data)
    K = len(second_groups)
    second_groups_counts = [len(g) for g in second_groups]
    second_num = sum(second_groups_counts)
    # 扁平化
    second_uavs = [u for grp in second_groups for u in grp]

    # 纵队相关参数
    column = afsim_data.get('column', K)  # 纵队数
    # 旧工程里 num_of_column 常当作“每纵队数量”的一个典型值；这里若未提供，就取各纵队数量的最大值
    num_of_column = afsim_data.get('num_of_columns', max(second_groups_counts) if second_groups_counts else 0)
    num_per_column = second_groups_counts[:]  # 新增，更直观

    # 生成 dataset.py（用字符串拼接，动态写出 second_uavs_1...K）
    lines = []
    p = lines.append

    p("def dataset():")
    p("    # 基准点坐标 (经度纬度高度)")
    p("    basepoint = [")
    p(f"        \"{to_lon_lat_alt(basepoint)[0]}\",")  # lon
    p(f"        \"{to_lon_lat_alt(basepoint)[1]}\",")  # lat
    p(f"        \"{to_lon_lat_alt(basepoint)[2]}\"")   # alt
    p("    ]")
    p("")
    p("    # 最近/最远检测范围")
    p(f"    near_detect_distance = {near_detect}")
    p(f"    detect_distance = {detect_distance}")
    p("")
    p("    # 速度参数")
    p(f"    maximum_speed = {maximum_speed}")
    p(f"    minimum_speed = {minimum_speed}")
    p(f"    speed = {speed}")
    p(f"    acceleration = {acceleration}")
    p("")
    p("    # 纵队参数")
    p(f"    column = {column}")
    p(f"    num_of_column = {num_of_column}")
    p(f"    num_per_column = {num_per_column}")
    p("")
    p("    # 敌方近似位置 (经度纬度高度)")
    ea = to_lon_lat_alt(enemy_approx)
    p("    enemy_approx = [")
    p(f"        \"{ea[0]}\",")
    p(f"        \"{ea[1]}\",")
    p(f"        \"{ea[2]}\"")
    p("    ]")
    p("")
    p(f"    radii = {radii}")
    p(f"    enemy_number = {enemy_number}")
    p(f"    enemy_lonrange = {enemy_lonrange}")
    p(f"    enemy_latrange = {enemy_latrange}")
    p("")
    p(f"    first_num = {first_num}")
    p("    first_uavs = [")
    for i, (lon, lat, alt) in enumerate(first_uavs):
        comma = "," if i < len(first_uavs)-1 else ""
        p("        [")
        p(f"            \"{lon}\",")
        p(f"            \"{lat}\",")
        p(f"            \"{alt}\"")
        p(f"        ]{comma}")
    p("    ]")
    p("")

    # 动态写出 second_uavs_1..K
    for gi, grp in enumerate(second_groups, start=1):
        p(f"    # 第二批第{gi}纵队")
        p(f"    second_num_{gi} = {len(grp)}")
        p(f"    second_uavs_{gi} = [")
        for j, (lon, lat, alt) in enumerate(grp):
            comma = "," if j < len(grp)-1 else ""
            p("        [")
            p(f"            \"{lon}\",")
            p(f"            \"{lat}\",")
            p(f"            \"{alt}\"")
            p(f"        ]{comma}")
        p("    ]")
        p("")

    # 汇总
    # second_groups = [ second_uavs_1, second_uavs_2, ... ]
    p("    second_groups = [")
    for gi in range(1, K+1):
        comma = "," if gi < K else ""
        p(f"        second_uavs_{gi}{comma}")
    p("    ]")
    # second_groups_counts = [ second_num_1, ... ]
    p("    second_groups_counts = [")
    for gi in range(1, K+1):
        comma = "," if gi < K else ""
        p(f"        second_num_{gi}{comma}")
    p("    ]")
    # 扁平化 second_uavs = [] + second_uavs_1 + second_uavs_2 + ...
    if K >= 1:
        cat_expr = "[] " + " ".join([f"+ second_uavs_{gi}" for gi in range(1, K+1)])
    else:
        cat_expr = "[]"
    p(f"    second_uavs = {cat_expr}")
    p(f"    second_num = {second_num}")
    p("")
    # 返回字典（包含动态键）
    p("    return {")
    p("        'basepoint': basepoint,")
    p("        'near_detect_distance': near_detect_distance,")
    p("        'detect_distance': detect_distance,")
    p("        'maximum_speed': maximum_speed,")
    p("        'minimum_speed': minimum_speed,")
    p("        'speed': speed,")
    p("        'acceleration': acceleration,")
    p("        'column': column,")
    p("        'num_of_column': num_of_column,")
    p("        'num_per_column': num_per_column,")
    p("        'enemy_approx': enemy_approx,")
    p("        'radii': radii,")
    p("        'enemy_number': enemy_number,")
    p("        'enemy_lonrange': enemy_lonrange,")
    p("        'enemy_latrange': enemy_latrange,")
    p("        'first_num': first_num,")
    p("        'first_uavs': first_uavs,")
    p("        'second_num': second_num,")
    p("        'second_uavs': second_uavs,")
    p("        'second_groups': second_groups,")
    p("        'second_groups_counts': second_groups_counts,")
    for gi in range(1, K+1):
        p(f"        'second_num_{gi}': second_num_{gi},")
        p(f"        'second_uavs_{gi}': second_uavs_{gi},")
    p("    }")

    dataset_path = os.path.join(current_dir, 'dataset.py')
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print("数据导入完成！")
    print(f"第一批无人机数量: {len(first_uavs)}")
    print(f"第二批纵队数量: {K} | 第二批总数: {second_num}")

if __name__ == '__main__':
    update_dataset()
