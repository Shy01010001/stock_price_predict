# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 19:48:03 2024

@author: hongyu
"""
from IndicatorUtils import *
import time
import baostock as bs
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from tqdm import tqdm
import yfinance as yf
# from DataUtils import *
from datetime import datetime
import random
import json
from utils import *
from functools import lru_cache

time_record()

datas = data_get()
ziguang = datas.GetAllHistData(['sz.002049'], "date,time,code,open,high,low,close,volume,amount,adjustflag", frequency='5')
datas.data_save(ziguang, 'ziguang')

# print(datas.codes_dict['sz'])

time_record('end')

