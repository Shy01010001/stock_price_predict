# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 21:45:09 2023

@author: hongyu
"""


import baostock as bs
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from tqdm import tqdm
import yfinance as yf
# from DataUtils import *
from datetime import datetime, timedelta
import random
import json
import pickle
from functools import lru_cache
import os

def drawPlot(long, short, code):
    global short_window, long_window
    x = [i for i in range(1, len(long) + 1)]
    print('draw')
    plt.plot(x, short, label='short')
    plt.plot(x, long, label='long')

    plt.legend()
    plt.xlabel('days')
    plt.ylabel('meanVal')
    plt.title(f"{short_window},{long_window} plot")
    plt.savefig(f'./meanFig/{code}.jpg')
    plt.show()
    


    
class data_get:
    def __init__(self):
        
        self.codes_dict = {}
        self.today = self.get_latest_val_date()
        self.GetAllStockCodes()
    
    def get_latest_val_date(self):
        today = datetime.now()
        weekday = today.weekday()
        if weekday == 5:
            today = today - timedelta(1)
        elif weekday == 6:
            today = today - timedelta(2)
        today = today.strftime('%Y-%m-%d')
        
        return today
    def data_save(self, data, name):
        with open(name + '.pkl', 'wb') as file:
            pickle.dump(data, file)
    
    def data_load(self, name):
        with open(name + '.pkl', 'rb') as file:
            return pickle.load(file)
    
    # def stock_select(self, marcket = ['sz', 'sh'], ratio = [1, 1]):
        
    
    def GetAllStockCodes(self, kinds='all', login=True):
        code_name = 'codes_dict'
        if os.path.exists(f'./{code_name}.pkl'):
            self.codes_dict = self.data_load(code_name)
        else:
            lg = bs.login()
            rs = bs.query_all_stock(day=self.today)
            df = rs.get_data()
            index_mask = df['code_name'].str.contains('指数')
            index_df = df[index_mask]
            single_stock_df = df[~index_mask]
            ### split sz, bj, sh ######
            for kind in ['bj', 'sh', 'sz']:
                single_stock_mask = single_stock_df["code"].str.startswith(kind)
                split_code_df = single_stock_df[single_stock_mask]
                self.codes_dict[kind] = split_code_df.set_index('code')['code_name'].to_dict()
            self.codes_dict['指数'] = index_df.set_index('code')['code_name'].to_dict()
            if login:
                bs.logout()
            
            self.data_save(self.codes_dict, code_name)
            
    def name_to_code(self,name):

        return self.code_df[self.code_df['code_name'] == name]['code'].iloc[0]
        
    def code_to_name(self,code):
        
        return self.code_df[self.code_df['code'] == code]['code_name'].iloc[0]
        
    def GetAllHistData(self, stockList, data_sel, start_data = '2008-01-04', end_data='2024-01-08', frequency = 'd'):
        lg = bs.login()
        per = []
        stockHistData = {}
        flag = 0
        for stockCode in stockList:
            # if flag == 100:
            #     break
            if 'bj' in stockCode:
                continue
            rs = bs.query_history_k_data_plus(stockCode,
                                              data_sel,
                                              start_date=start_data, end_date=end_data,
                                              frequency=frequency, adjustflag="2")
            if rs is None:
                print(f"查询股票代码{stockCode}的历史数据失败")
                continue
            if rs.error_code != '0':
                print(f"查询股票代码{stockCode}的历史数据失败，错误信息：{rs.error_msg}")
                continue


            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            result = pd.DataFrame(data_list, columns=rs.fields)
            result = pd.DataFrame(data_list, columns=rs.fields)
    
            # 转换日期列为日期时间对象
            result['date'] = pd.to_datetime(result['date'])
    
            # 将数值列转换为浮点数类型
            result[['open', 'high', 'low', 'close', 'volume', 'amount']] = result[['open', 'high', 'low', 'close', 'volume', 'amount']].astype(float)
    
            stockHistData[stockCode] = result
            self.data_save(stockHistData, f'{stockCode}')
        bs.logout()
        
        # result.to_excel('close_data.xlsx', index=False)  # index=False表示不保存索引列
        return stockHistData 
    
def CalcRsiValue(data, n=14, buy_threshold=30, sell_threshold=70):
    """
    计算每只股票的RSI值并输出买卖信号
    :param stockHistData: 一个字典，键为股票代码，值为对应的历史数据
    :param n: RSI计算所需的天数，默认为14天
    :param buy_threshold: RSI低于此阈值则视为买入信号，默认为30
    :param sell_threshold: RSI高于此阈值则视为卖出信号，默认为70
    :return: 一个字典，键为股票代码，值为包含RSI值列表和对应的交易信号字典的元组
    """
    close_prices = data['close'].tolist()
    close_prices = [float(price) for price in close_prices]  # 将列表中的元素都转换为浮点数类型
    rsi_values = [np.nan] * n
    up_sum = 0
    down_sum = 0
    trade_signal_dict = {}
    for i in range(n, len(close_prices)):
        diff = close_prices[i] - close_prices[i-n]
        if diff > 0:
            up_sum += diff
        else:
            down_sum += abs(diff)
        if i == n:
            up_avg = up_sum / n
            down_avg = down_sum / n
        else:
            up_avg = (up_avg * (n-1) + max(diff, 0)) / n
            down_avg = (down_avg * (n-1) + abs(min(diff, 0))) / n
        if down_avg == 0:
            rsi_value = 100
        else:
            rs = up_avg / down_avg
            rsi_value = 100 - 100 / (1 + rs)
        rsi_values.append(rsi_value)
        if len(rsi_values) > n:
            if rsi_values[-1] < buy_threshold:
                trade_signal_dict[i] = 'buy'
            elif rsi_values[-1] > sell_threshold:
                trade_signal_dict[i] = 'sell'
            else:
                trade_signal_dict[i] = 'hold'
    return rsi_values, trade_signal_dict
    
    


def GetAllHistData(stockList, data_sel, start_data = '2008-08-08', end_data='2024-01-08', frequency = 'd'):

    per = []
    stockHistData = {}
    flag = 0
    for stockCode in stockList:
        if flag == 100:
            break
        if 'bj' in stockCode:
            continue
        rs = bs.query_history_k_data_plus(stockCode,
                                          data_sel,
                                          start_date=start_data, end_date=end_data,
                                          frequency=frequency, adjustflag="3")
        if rs is None:
            print(f"查询股票代码{stockCode}的历史数据失败")
            continue
        if rs.error_code != '0':
            print(f"查询股票代码{stockCode}的历史数据失败，错误信息：{rs.error_msg}")
            continue


        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        # result.set_index("date", inplace=True)
        result = result.astype('float')
        per = result.pct_change()
        
        per = per.fillna(float(0.0))
        stockHistData[stockCode] = per
        flag += 1
        # break
    bs.logout()
    # result.to_excel('close_data.xlsx', index=False)  # index=False表示不保存索引列
    return stockHistData

def calculate_rsrs(stockData):
    stockRSRS = {}
    # 遍历每个股票代码
    for stockCode in stockData.keys():
        # 获取股票历史数据
        stock = stockData[stockCode]
        # 计算RSRS指标
        close = stock['close'].astype(float)
        sma_short = close.rolling(window=10).mean()
        sma_long = close.rolling(window=30).mean()
        resid = close - (sma_short + sma_long) / 2
        std_resid = resid.rolling(window=30).std()
        rsrs = resid / std_resid
        # 添加RSRS值到字典中
        stockRSRS[stockCode] = rsrs
    return stockRSRS



def calculate_obv(stock_data):
    stock_data['OBV'] = 0
    obv_values = [0]
    signal_dict = {}
    
    for i in range(1, len(stock_data)):
        if stock_data['close'][i] > stock_data['close'][i-1]:
            obv_values.append(obv_values[-1] + stock_data['volume'][i])
        elif stock_data['close'][i] < stock_data['close'][i-1]:
            obv_values.append(obv_values[-1] - stock_data['volume'][i])
        else:
            obv_values.append(obv_values[-1])

        if obv_values[i] > obv_values[i-1]:
            signal_dict[i] = 'buy'
        elif obv_values[i] < obv_values[i-1]:
            signal_dict[i] = 'sell'
        else:
            signal_dict[i] = 'hold'

    stock_data['OBV'] = obv_values
    
    return stock_data['OBV'], signal_dict


def backtest(strategy_func, stock_code):
    """
    获取输入股票代码的历史3年数据，回测量化策略盈亏额，初始10000元
    :param strategy_func: 策略函数，有一个输入值，DataFrame结构的股票历史信息，有两个return值。
                          第一个是一个列表指标值，第二个是字典结构，key是第n天，value是第n天的操作，
                          包括sell，buy和hold信号。
    :param stock_code: 股票代码，字符串类型，如 'sh.600000'
    :return: 每一天资产+现金的总和，列表类型
    """
    # 登陆系统
    bs.login()
    # 获取股票历史数据
    rs = bs.query_history_k_data(stock_code, "date,open,high,low,close,volume", start_date='2022-03-28',
                                 end_date='2008-03-28', frequency="d", adjustflag="3")
    # 转换数据格式
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    df_stock = pd.DataFrame(data_list, columns=rs.fields)
    df_stock = df_stock.set_index('date')
    df_stock = df_stock.astype(float)
    # 执行策略函数
    indicators, signals = strategy_func(df_stock)
    df_stock = df_stock.reset_index()
    # 绘制股价和买卖标记的折线图
    plt.plot(df_stock.index, df_stock['close'])
    for day, signal in signals.items():
        if signal == 'buy':
            plt.plot(day, df_stock.loc[day, 'close'], '^', markersize=10, color='green')
        elif signal == 'sell':
            plt.plot(day, df_stock.loc[day, 'close'], 'v', markersize=10, color='red')
    plt.title('Stock Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    # 进行回测
    cash = 10000  # 初始资金
    shares = 0  # 初始股票持仓
    profits = []  # 每天的盈利情况
    for day, signal in signals.items():
        close_price = df_stock.loc[day, 'close']
        if signal == 'buy':
            num_shares = min(int(cash / close_price), 10000 // int(close_price))  # 不买超过本金的股票
            shares += num_shares
            cash -= num_shares * close_price
        elif signal == 'sell':
            cash += shares * close_price
            shares = 0
        profits.append(cash + shares * close_price)
    # 退出系统
    bs.logout()
    return profits

def visualize_rsrs(stockRSRS):
    for stockCode, rsrs in stockRSRS.items():
        plt.plot(rsrs, label=stockCode)
        plt.axhline(y=1, color='r', linestyle='--')
        plt.axhline(y=-1, color='g', linestyle='--')
        plt.title(stockCode + ' RSRS')
        plt.legend()
        plt.show()
        
def generate_signals(stockRSRS):
    signals = {}
    for stockCode, rsrs in stockRSRS.items():
        if rsrs[-1] > 1:
            signals[stockCode] = 'BUY'
            plt.plot(rsrs, label=stockCode)
            plt.axhline(y=1, color='r', linestyle='--')
            plt.axhline(y=-1, color='g', linestyle='--')
            plt.title(stockCode + ' RSRS, BUY')
            plt.legend()
            plt.show()            
        elif rsrs[-1] < -1:
            signals[stockCode] = 'BUY'
            # plt.plot(rsrs, label=stockCode)
            # plt.axhline(y=1, color='r', linestyle='--')
            # plt.axhline(y=-1, color='g', linestyle='--')
            # plt.title(stockCode + ' RSRS, SELL')
            # plt.legend()
            # plt.show()              
        else:
            signals[stockCode] = 'HOLD'
    return signals

def CalcNDayMa(hist_data, n):
    """
    计算每一支股票的N日均线
    :param hist_data: 一个字典，键为股票代码，值为对应的历史数据
    :param n: N日均线的天数
    :return: 一个字典，键为股票代码，值为对应的N日均线数据
    """
    ma_dict = {}
    # hist_data[1] = hist_data[1].set_index('date')
    for stock_code, data in hist_data.items():
        
        close_prices = data['close'].tolist()
        close_prices = [float(price) for price in close_prices]  # 将列表中的元素都转换为浮点数类型
        ma_values = []
        for i in range(len(close_prices)):
            if i < n-1:
                continue
            else:
                ma_value = sum(close_prices[i-n+1:i+1]) / n
                ma_values.append(ma_value)
        ma_dict[stock_code] = ma_values
    return ma_dict


def calculate_capm(stock_code: str):
    # 获取股票历史数据
    stock_df = get_stock_data(stock_code, "2020-01-01", "2021-12-31")

    # 计算股票收益率
    stock_returns = stock_df.pct_change().dropna()

    # 获取市场指数历史数据
    market_df = get_stock_data("sh.000300", "2020-01-01", "2021-12-31") # 使用沪深300指数作为市场指数代表

    # 计算市场指数收益率
    market_returns = market_df.pct_change().dropna()

    # 计算股票与市场指数收益率的协方差和市场指数收益率的方差
    covariance = np.cov(stock_returns["close"], market_returns["close"])[0][1]
    market_variance = np.var(market_returns["close"])

    # 计算贝塔系数
    beta = covariance / market_variance

    # 获取国债收益率作为无风险利率
    risk_free_rate = get_bond_yield_rate("2021-12-31")
    if risk_free_rate:
        risk_free_rate = float(risk_free_rate[0]) / 100
    else:
        risk_free_rate = 0.0

    # 计算市场指数历史平均收益率
    expected_market_return = market_returns.mean()[0]

    # 使用 CAPM 计算预期收益率
    capm_return = risk_free_rate + beta * (expected_market_return - risk_free_rate)

    return capm_return

def SelectCross(short, long):
    l = []
    for code in short:
        
        try:
            if short[code][-1] > long[code][-1] and short[code][-2] < long[code][-2]:
                print('cross stock:', code)
                l.append(code)
        except:
            print('error:', code)
    return l

def get_list(stock_data, window_size, interval):
    all_data = pd.concat(stock_data.values(), axis=1)
    all_data = all_data.fillna(0.0)
    
    # 转置数据，使得日期成为行而不是列
    all_data = all_data.T
    num_rows, num_cols = all_data.shape
    values = all_data.values.tolist()
    # 将数据按照步长为1的窗口分割为新的DataFrame
    windows = [all_data.iloc[i:i+window_size] for i in range(0, num_rows - window_size + 1, interval)]
    # 将每个窗口数据转换为列表，并将所有窗口数据合并成一个大列表
    final_data = [window.values.tolist() for window in windows]
    return values, final_data

def set_0_instead_nan(stock_data):
    for i in range(len(stock_data)):
        for j in range(len(stock_data[i])):
            for k in range(len(stock_data[i][j])):
                if np.isnan(stock_data[i][j][k]):
                    stock_data[i][j][k] = 0.0
    return stock_data

def get_gtr(stock_data_list, interval, window_size):
    gtr = []
    for i in range(window_size, len(stock_data_list), interval):
        gtr.append(stock_data_list[i])
        
    return gtr

interval = 5
window_size = 30

# stocks = GetAllStockCodes()s
# stock_data = GetAllHistData(stocks, "close")
# # print(stock_data)

# stock_data_list, new_stock_data = get_list(stock_data, window_size, interval) # 3 time_window
# gtr = get_gtr(stock_data_list, interval, window_size)



# train_ratio = 0.9
# total_num = len(new_stock_data)

# train_num = int(total_num * train_ratio)
# random.seed(1)
# random.shuffle(new_stock_data)
# random.seed(1)
# random.shuffle(gtr)

# train_input = new_stock_data[: train_num]
# train_gtr = gtr[:train_num]

# test_input = new_stock_data[train_num:]
# test_gtr = gtr[train_num:]

# data_dict = {}
# data_dict['train'] = {'input' : train_input, 'label' : train_gtr}

# data_dict['test'] = {'input' : test_input, 'label' : test_gtr} 

# with open('./data.json', "w") as json_file:
#     json.dump(data_dict, json_file)
# new_stock_data = set_0_instead_nan(new_stock_data)

# 将所有DataFrame合并成一个大的DataFrame

# with pd.ExcelWriter('stock_data.xlsx', engine='xlsxwriter') as writer:
    # for stock_code, df in stock_data.items():
    #     # 将每个DataFrame保存为一个工作表
    #     df.to_excel(writer, sheet_name=stock_code, index=False)
# short_window = 10;
# long_window = 50;
# # 获取股票列表
# # stockCode = 'sh.600000'
# stockData = GetAllHistData(stocks)
# short = CalcNDayMa(stockData, short_window)
# long = CalcNDayMa(stockData, long_window)
# sd = SelectCross(short, long)
# for cd in sd:
#     data = stockData[cd]
#     print(cd,' Rsi:',CalcRsiValue(data)[0][-1])
# # calculate_capm(stockCode)
