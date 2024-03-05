# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 07:40:38 2024

@author: hongyu
"""

import pandas as pd
import datetime
import requests
from requests.exceptions import ConnectionError
from bs4 import BeautifulSoup
import pdb

def web_content_div(web_content, class_path):
    web_content_div = web_content.find_all('div', {'class': class_path})
    try:
        spans = web_content_div[0].find_all('fin_streamer')
        texts = [span.get_text() for span in spans]
    except IndexError:
        texts = []
        return texts
    
def real_time_price(stock_code):

    url = f'https://finance.yahoo.com/quote/{stock_code}?p={stock_code}'
    try:
        pdb.set_trace()
        r = requests.get(url)
        web_content = BeautifulSoup(r.text, 'lxml')
        
        texts = web_content_div(web_content, 'My(6px) Pos(r) smartphone_Mt(6px) W(100%) ')
        if texts != []:
            price, change = texts[0], texts[1]
        else:
            price, change = [], []
    except ConnectionError:
        price, change = [], []
        
        
    return price, change

stock_code = ['NVDA']
print(real_time_price('NVDA'))