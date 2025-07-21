import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
import velocity_recong

from numpy.ma.core import remainder

import GeodeticConverter
import outdata
import dataset

#先对UAV2按照y轴，即距离敌方的远近进行排序
def sorted_y_points(second_points, converter):
    seconds = []
    i = 0
    for point in second_points:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(point)
        point_local = converter.geodetic_to_local(lat, lon, alt)
        seconds.append([point_local, i, 0])
        i += 1
    seconds_sorted = sorted(seconds, key=lambda item: item[0][1], reverse=True)#按y轴排序
    return seconds_sorted








if __name__ == "__main__":
    enemy_geo, second_points = dataset.dataset()


