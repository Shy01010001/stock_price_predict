# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 20:01:08 2023

@author: hongyu
"""
import pandas as pd
from IndicatorUtils import GetAllHistData, GetAllStockCodes
from tqdm import tqdm


# def main():
codes_list = GetAllStockCodes()
print(codes_list)
historical_data = GetAllHistData(codes_list)

# 创建一个新的DataFrame来存储涨跌幅数据
result_df = pd.DataFrame(columns=["Code", "Change_Percentage"])

def calculate_change_percentage(dataframe):
    # 计算涨跌百分比
    dataframe["close"] = dataframe["close"].astype(float)

    # 计算涨跌百分比
    dataframe["change_percentage"] = dataframe["close"].pct_change()
    return dataframe
# 计算涨跌幅并将结果添加到result_df中
writer = pd.ExcelWriter("涨跌幅.xlsx", mode = 'a', engine='openpyxl')
for i, (code, data) in tqdm(enumerate(historical_data.items()), position = 0):
    # print(data["close"])
    # if "close" in data.columns:
    data = calculate_change_percentage(data)
    data.to_excel(writer, sheet_name=code, index=False)
    if (i + 1 % 100) == 0:
        writer.save()
        writer.close()
        writer = pd.ExcelWriter("涨跌幅.xlsx", mode = 'a', engine='openpyxl')
writer.save()
writer.close()
# 保存结果到Excel文件

