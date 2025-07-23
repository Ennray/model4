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

def update_dataset():
    """使用afsim.txt的数据更新dataset.py"""
    
    # 获取路径信息
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载afsim数据
    afsim_data = load_afsim_data()
    
    # 获取所有数据
    first_uavs = afsim_data.get('first_uavs', [])
    second_uavs = afsim_data.get('second_uavs', [])
    
    # 生成新的dataset.py内容
    dataset_content = '''def dataset():
    # 基准点坐标 (经度纬度高度)
    basepoint = [
        "''' + afsim_data.get('basepoint', ['', '', ''])[1] + '''",
        "''' + afsim_data.get('basepoint', ['', '', ''])[0] + '''",
        "''' + afsim_data.get('basepoint', ['', '', ''])[2] + '''"
    ]
    
    # 最小检测范围
    detect_distance = ''' + str(afsim_data.get('detect_distance', 30000)) + '''
    
    # 速度参数
    maximum_speed = ''' + str(afsim_data.get('maximum_speed', 300)) + '''
    minimum_speed = ''' + str(afsim_data.get('minimum_speed', 200)) + '''
    speed = ''' + str(afsim_data.get('speed', 240)) + '''
    
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
    enemy_lonrange = ''' + str(afsim_data.get('enemy_lonrange', [])) + '''
    
    # 敌方纬度范围
    enemy_latrange = ''' + str(afsim_data.get('enemy_latrange', [])) + '''
    
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
    
    # 第二批无人机数量
    second_num = ''' + str(afsim_data.get('second_num', 30)) + '''
    
    # 第二批无人机坐标 (经度纬度高度)
    second_uavs = [
'''
    
    # 添加second_uavs数据 (改为经度纬度高度顺序)
    for i, uav in enumerate(second_uavs):
        lat, lon, alt = uav[0], uav[1], uav[2]
        if i == len(second_uavs) - 1:
            dataset_content += f'        [\n            "{lon}",\n            "{lat}",\n            "{alt}"\n        ]\n'
        else:
            dataset_content += f'        [\n            "{lon}",\n            "{lat}",\n            "{alt}"\n        ],\n'
    
    dataset_content += '''    ]
    
    return {
        'basepoint': basepoint,
        'detect_distance': detect_distance,
        'maximum_speed': maximum_speed,
        'minimum_speed': minimum_speed,
        'speed': speed,
        'enemy_approx': enemy_approx,
        'radii': radii,
        'enemy_number': enemy_number,
        'enemy_lonrange': enemy_lonrange,
        'enemy_latrange': enemy_latrange,
        'first_num': first_num,
        'first_uavs': first_uavs,
        'second_num': second_num,
        'second_uavs': second_uavs
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
