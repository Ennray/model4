import json
import os

def load_afsim_data():
    """从afsim.txt加载数据"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    afsim_path = os.path.join(current_dir, 'afsim.txt')
    
    with open(afsim_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def dms_to_decimal(dms_str):
    """将度分秒字符串转换为十进制度数（E/N为正，W/S为负）"""
    direction = dms_str[-1]
    value_str = dms_str[:-1]
    parts = value_str.split(':')
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    decimal = degrees + minutes / 60 + seconds / 3600
    if direction in ['W', 'S']:
        decimal = -decimal
    return decimal



def update_dataset():
    """使用afsim.txt的数据更新dataset.py"""
    
    # 获取路径信息
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载afsim数据
    afsim_data = load_afsim_data()
    
    # 获取所有数据
    first_uavs = afsim_data.get('first_uavs', [])
    second_uavs_1 = afsim_data.get('second_uavs_1', [])
    second_uavs_2 = afsim_data.get('second_uavs_2', [])
    second_uavs_3 = afsim_data.get('second_uavs_3', [])
    second_uavs = []
    for grp in (second_uavs_1, second_uavs_2, second_uavs_3):
        second_uavs.extend(grp)


    enemy_lonrange = afsim_data.get('enemy_lonrange', [])
    enemy_latrange = afsim_data.get('enemy_latrange', [])

    # 统一经度顺序 [小, 大]
    if dms_to_decimal(enemy_lonrange[0]) > dms_to_decimal(enemy_lonrange[1]):
        enemy_lonrange[0], enemy_lonrange[1] = enemy_lonrange[1], enemy_lonrange[0]

    # 统一纬度顺序 [小, 大]
    if dms_to_decimal(enemy_latrange[0]) > dms_to_decimal(enemy_latrange[1]):
        enemy_latrange[0], enemy_latrange[1] = enemy_latrange[1], enemy_latrange[0]

    
    # 生成新的dataset.py内容
    dataset_content = '''def dataset():
    # 基准点坐标 (经度纬度高度)
    basepoint = [
        "''' + afsim_data.get('basepoint', ['', '', ''])[1] + '''",
        "''' + afsim_data.get('basepoint', ['', '', ''])[0] + '''",
        "''' + afsim_data.get('basepoint', ['', '', ''])[2] + '''"
    ]
    
    # 最近检测范围
    near_detect_distance = ''' + str(afsim_data.get('min_detect', [900, 600])) + '''
    
    # 最远检测范围
    detect_distance = ''' + str(afsim_data.get('detect_distance', 30000)) + '''
    
    # 速度参数
    maximum_speed = ''' + str(afsim_data.get('maximum_speed', 300)) + '''
    minimum_speed = ''' + str(afsim_data.get('minimum_speed', 200)) + '''
    speed = ''' + str(afsim_data.get('speed', 240)) + '''
    acceleration = ''' + str(afsim_data.get('acceleration', 80)) + '''
    
    # 纵队参数
    column = ''' + str(afsim_data.get('column', 3)) + '''
    num_of_column = ''' + str(afsim_data.get('num_of_columns', 18)) + '''
    
    # 敌方近似位置 (经度纬度高度)
    enemy_approx = [
        "''' + afsim_data.get('enemy_approx', ['', '', ''])[1] + '''",
        "''' + afsim_data.get('enemy_approx', ['', '', ''])[0] + '''",
        "''' + afsim_data.get('enemy_approx', ['', '', ''])[2] + '''"
    ]
    
    # 半径数组
    radii = ''' + str(afsim_data.get('radii', [])) + '''
    
    # 敌方数量
    enemy_number = ''' + str(afsim_data.get('enemy_number', 300)) + '''
    
    
    # 敌方经度范围
    enemy_lonrange = ''' + str(enemy_lonrange) + '''
    
    # 敌方纬度范围
    enemy_latrange = ''' + str(enemy_latrange) + '''
    
    # 第一批无人机数量
    first_num = ''' + str(afsim_data.get('first_num', 45)) + '''
    
    # 第一批无人机坐标 (经度纬度高度)
    first_uavs = [
'''
    
    # 添加first_uavs数据 (改为经度纬度高度顺序)
    for i, uav in enumerate(first_uavs):
        lat, lon, alt = uav[0], uav[1], uav[2]
        if i == len(first_uavs) - 1:
            dataset_content += f'        [\n            "{lon}",\n            "{lat}",\n            "{alt}"\n        ]\n'
        else:
            dataset_content += f'        [\n            "{lon}",\n            "{lat}",\n            "{alt}"\n        ],\n'
    
    dataset_content += '''    ]
    
    
    # 第二批第1纵队无人机数量
    second_num_1 = ''' + str(afsim_data.get('second_num_1', 18)) + '''
    
    # 第二批第1纵队无人机坐标 (经度纬度高度)
    second_uavs_1 = [
'''
    
    # 添加second_uavs数据 (改为经度纬度高度顺序)
    for i, uav in enumerate(second_uavs_1):
        lat, lon, alt = uav[0], uav[1], uav[2]
        if i == len(second_uavs_1) - 1:
            dataset_content += f'        [\n            "{lon}",\n            "{lat}",\n            "{alt}"\n        ]\n'
        else:
            dataset_content += f'        [\n            "{lon}",\n            "{lat}",\n            "{alt}"\n        ],\n'
    
    dataset_content += '''    ]
    
    
    # 第二批第2纵队无人机数量
    second_num_2 = ''' + str(afsim_data.get('second_num_2', 18)) + '''
    
    # 第二批第2纵队无人机坐标 (经度纬度高度)
    second_uavs_2 = [
'''

    # 添加second_uavs数据 (改为经度纬度高度顺序)
    for i, uav in enumerate(second_uavs_2):
        lat, lon, alt = uav[0], uav[1], uav[2]
        if i == len(second_uavs_2) - 1:
            dataset_content += f'        [\n            "{lon}",\n            "{lat}",\n            "{alt}"\n        ]\n'
        else:
            dataset_content += f'        [\n            "{lon}",\n            "{lat}",\n            "{alt}"\n        ],\n'

    dataset_content += '''    ]
    
     # 第二批第3纵队无人机数量
    second_num_3 = ''' + str(afsim_data.get('second_num_3', 7)) + '''
    
    # 第二批第一纵队无人机坐标 (经度纬度高度)
    second_uavs_3 = [
'''

    # 添加second_uavs数据 (改为经度纬度高度顺序)
    for i, uav in enumerate(second_uavs_3):
        lat, lon, alt = uav[0], uav[1], uav[2]
        if i == len(second_uavs_3) - 1:
            dataset_content += f'        [\n            "{lon}",\n            "{lat}",\n            "{alt}"\n        ]\n'
        else:
            dataset_content += f'        [\n            "{lon}",\n            "{lat}",\n            "{alt}"\n        ],\n'

    dataset_content += '''    ]
    
     # 第二批无人机总数量
    second_num = ''' + str(afsim_data.get('second_num_3', 7) + afsim_data.get('second_num_2', 18) + afsim_data.get('second_num_1', 18)) + '''
    
    # 以你原来的格式写入 dataset_content（经度、纬度、高度）

    second_uavs = [
'''
    for i, uav in enumerate(second_uavs):
        lat, lon, alt = uav[0], uav[1], uav[2]
        if i == len(second_uavs) - 1:
            dataset_content += f'        [\n            "{lon}",\n            "{lat}",\n            "{alt}"\n        ]\n'
        else:
            dataset_content += f'        [\n            "{lon}",\n            "{lat}",\n            "{alt}"\n        ],\n'

    dataset_content += '''    ]
    
    
    
    
    return {
        'basepoint': basepoint,
        'near_detect_distance': near_detect_distance,
        'detect_distance': detect_distance,
        'maximum_speed': maximum_speed,
        'minimum_speed': minimum_speed,
        'speed': speed,
        'acceleration': acceleration,
        'column': column,
        'num_of_column': num_of_column,
        'enemy_approx': enemy_approx,
        'radii': radii,
        'enemy_number': enemy_number,
        'enemy_lonrange': enemy_lonrange,
        'enemy_latrange': enemy_latrange,
        'first_num': first_num,
        'first_uavs': first_uavs,
        'second_num': second_num,
        'second_uavs': second_uavs,
        'second_num_1': second_num_1,
        'second_uavs_1': second_uavs_1,
        'second_num_2': second_num_2,
        'second_uavs_2': second_uavs_2,
        'second_num_3': second_num_3,
        'second_uavs_3': second_uavs_3,
        
    }
'''
    
    # 写入新的dataset.py文件
    dataset_path = os.path.join(current_dir, 'dataset.py')
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.write(dataset_content)
    
    print(f"数据导入完成！")
    print(f"第一批无人机数量: {len(first_uavs)}")
    print(f"第二批无人机数量: {len(second_uavs)}")

if __name__ == "__main__":
    update_dataset()
