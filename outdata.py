import json

def save_uav_multi_positions(
    uavs_current_speed, acceleration, uav_max_speed, uav_turn_speed,
    enemy_speed,
    first_init_point, second_init_point, enemy_init_point,

    last_uav_turn_point, last_uav_after_turn_point, last_uav_chase_point,
    last_uav_meet_time, last_uav_turn_time, last_uav_chase_time,

    first_uav_turn_point, first_uav_after_turn_point, first_uav_chase_point,
    first_uav_meet_time, first_uav_turn_time, first_uav_chase_time,

    second_uav_turn_point, second_uav_after_turn_point, second_uav_chase_point,
    second_uav_meet_time, second_uav_turn_time, second_uav_chase_time,

    filename="uav_full_path_record.json"
):
    data = {
        "uavs_speed": {
            "current": uavs_current_speed,
            "acceleration": acceleration,
            "max_speed": uav_max_speed,
            "turn_speed": uav_turn_speed,
        },
        "enemy_speed": enemy_speed,
        #==================初始信息===================
        "first_init_point": first_init_point,
        "second_init_point": second_init_point,
        "enemy_init_point":enemy_init_point,

        #==============每个纵队最后一架无人机移动点位============
        "last_uav_turn_point": last_uav_turn_point,
        "last_uav_after_turn_point": last_uav_after_turn_point,
        "last_uav_chase_point": last_uav_chase_point,

        #==============每个纵队最后一架无人机移动时间============
        "last_uav_meet_time": last_uav_meet_time,
        "last_uav_turn_time": last_uav_turn_time,
        "last_uav_chase_time": last_uav_chase_time,

        #==============第1波次无人机移动点位============
        "first_uav_turn_point": first_uav_turn_point,
        "first_uav_after_turn_point": first_uav_after_turn_point,
        "first_uav_chase_point": first_uav_chase_point,

        #==============第1波次无人机移动时间============
        "first_uav_meet_time": first_uav_meet_time,
        "first_uav_turn_time": first_uav_turn_time,
        "first_uav_chase_time": first_uav_chase_time,

        #==============第2波次无人机移动点位============
        "second_uav_turn_point": second_uav_turn_point,
        "second_uav_after_turn_point": second_uav_after_turn_point,
        "second_uav_chase_point": second_uav_chase_point,

        #==============第2波次无人机移动时间============
        "second_uav_meet_time": second_uav_meet_time,
        "second_uav_turn_time": second_uav_turn_time,
        "second_uav_chase_time": second_uav_chase_time,

    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# def real_time_position():


