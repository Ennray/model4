from flask import Flask, request, jsonify
import position_setting
# from utils import process_data

app = Flask(__name__)

@app.route('/posiset', methods=['POST'])
def json():
    data = request.get_json()
    basepoint = data.get('basepoint', ["28:11:34.95W", "10:30:35.24N", "55.0"])
    enemy_approx = data.get('enemy_approx', ["28:13:34.95W", "10:30:31.24N", "50.0"])
    radi = data.get('radii', [2500,2350,2450,2500,2400,2400,2480,2380])
    min_detect = data.get('min_detect', [900, 600])
    clear_detect = data.get('clear_detect', [240, 160])
    enemy_number = data.get('enemy_number', 350)
    interval = data.get('interval', 500)
    t = data.get('time', 5)
    speed = data.get('enemy_speed',240)
    first_num, first_uavs, second_num, second_uavs,optimal_rects, first_center= position_setting.init_position_set(basepoint, enemy_approx, radi, min_detect, clear_detect, enemy_number, interval, t, speed )

    return jsonify({'first_num': first_num,'first_uavs':first_uavs,'second_num':second_num,'second_uavs':second_uavs})

@app.route('/posiadjust', methods=['POST'])
def posiAdjustjson():
    data = request.get_json()
    basepoint = data.get('basepoint', ["28:11:34.95W", "10:30:35.24N", "55.0"])
    enemy_approx = data.get('enemy_approx', ["28:13:34.95W", "10:30:31.24N", "50.0"])
    first_uavs = data.get('first_uavs',)
    second_uavs = data.get('second_uavs',)
    type = data.get('type', 1)
    dist = data.get('dist',[500, 100, 200])
    first_uavs, second_uavs = position_setting.position_adjust(basepoint, enemy_approx, first_uavs, second_uavs, type, dist)
    return jsonify({'first_uavs': first_uavs, 'second_uavs': second_uavs})

if __name__ == '__main__':
    app.run(port=5000)
    # position_setting.init_position_set