# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 19:37:27 2024

@author: hongyu
"""

import time

time_points_list = []

def time_record(flag = 'not end'):
    global time_points_list
    
    time_points_list.append(time.perf_counter())
    print(time_points_list)
    if flag == 'end':
        print(time_points_list[-1] - time_points_list[0])
    