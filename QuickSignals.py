import streamlit as st
from datetime import date
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import pymongo
import dns.resolver
import pandas_ta
from plotly import graph_objs as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pandas_ta
import pandas_ta as ta
import math

import requests


import logging

logging.basicConfig(level='INFO')

mlogger = logging.getLogger('matplotlib')
mlogger.setLevel(logging.WARNING)


pd.set_option("display.precision", 2)

from datetime import date
import numpy as np
# from pytictoc import TicToc
# t = TicToc()


st.set_page_config(layout="wide")

def sortWRTDates(indf):
    indf['DataFrame Column'] = pd.to_datetime(indf['Date'], format="%Y-%m-%d")
    indf = indf.sort_values(by='DataFrame Column',ignore_index=True)
    
    indf = indf.drop('DataFrame Column', axis=1)
    return indf

def getUpdatedData(mo,ds,symbol,symbolType):
    df = mo.getDailyData(symbol)
    td = ds.getData(symbol,symbolType,isDebug=False)
    td[0]=df.iloc[-1].Close

    today = date.today()
    todayDate = today.strftime("%Y-%m-%d")

    y = [todayDate,td[0],td[1],td[2],td[3]]
    a_series = pd.Series(y, index = df.columns)
    df=df.append(a_series,ignore_index=True)

    return df

def getTodayData(copydf):
    df = copydf.iloc[:]
    df.reset_index(inplace=True)
    # print(df.tail())
    today = date.today()   
    todayDate = today.strftime("%d/%m/%Y")
    todayData = df[df['Date'].map(lambda s: s.startswith(todayDate))]
    if(todayData['Close'].shape[0]==1):
        return [date.today().strftime("%Y-%m-%d"),np.nan,np.nan,np.nan,np.nan]
    else:
        return [date.today().strftime("%Y-%m-%d"),todayData['Close'].iloc[0],todayData['Close'].min(),todayData['Close'].max(),todayData['Close'].iloc[len(todayData['Close'])-1]]

class MongoObject:
    def __init__(self,db):
        self.db = db

    def getUsers(self):
        return self.db.userData.distinct("userID")

    def getDailyData(self,symbol):
        # print(symbol)
        results = self.db.dailyData.find(    { "symbol": symbol },
                                {"_id": 0,'symbol':1,'Date':1,'Open':1,'High':1,'Low':1,'Close':1}
                            )
        output = []
        for result in results:
            # print(result)
            thisOutput = [result['Date'],
                            float(result['Open']),
                            float(result['Low']),
                            float(result['High']),
                            float(result['Close'])
                            ]
            output.append(thisOutput)

        df = pd.DataFrame(output,columns=['Date','Open','Low','High','Close'])
        
        return df

    def getUpdatedDailyData(self,symbol='Karachi 100'):
        dateData = self.getDailyData(symbol)
        if(datetime.today().weekday()<5):
            quickData = self.getQuickData(symbol)
            y=getTodayData(quickData)
            a_series = pd.Series(y, index = dateData.columns)
            dateData=dateData.append(a_series,ignore_index=True)
            return dateData
        else:
            return dateData

    def getQuickData(self,symbol):
        results = self.db.quickData.find(   { "symbol": symbol },
                                            {"_id": 0,'Date':1,'symbol':1,'Close':1}
                                        )
        output = []
        for result in results:
            # print(result)
            thisOutput = [result['Date'],
                            float(result['Close'])
                            ]
            output.append(thisOutput)   
        df = pd.DataFrame(output,columns=['Date','Close'])
        df.set_index('Date',inplace=True)
        
        return df

    def getGotStocks(self,userID):
        results = pd.DataFrame(list(self.db.gotData.find( {'userID':userID },
                        {"_id": 0,
                        'userID':1,
                        'shares':1,
                        'actualSym':1,
                        'bought_price':1,
                        'bought_on':1,
                        'trail_fixed':1,
                        'slVal':1})
                      ))
        return results
    
    def getWatchStocks(self,userID):
        results = pd.DataFrame(list(self.db.watchData.find( {'userID':userID },
                        {"_id": 0,
                        'actualSym':1
                        })
        ))
        # print(results)
        return results

    def getUserInfo(self,userID):
        results = pd.DataFrame(list(self.db.userData.find( {'userID':userID},
                                {"_id": 0,
                                "userID":1,
                                "email":1,
                                "allName":1
                                })
        ))
        return results.iloc[0]

def plot_raw_data(data,caption):
    fig = go.Figure()

    data = data.iloc[-200:]
    data['EMA9']=data['Close'].ewm(span=9, adjust=True).mean()
    data['EMA21']=data['Close'].ewm(span=21, adjust=True).mean()
    data['EMA100']=data['Close'].ewm(span=100, adjust=True).mean()
    macd,signal = getmacd(data,12,26,9)
    data['MACD']=macd
    data['Signal']=signal
    data['Hist']=macd-signal
    data['RSI'] = pandas_ta.rsi(data['Close'], length = 14)
    df = data.iloc[-70:]

    fig = make_subplots(    rows=3,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.0, 
                            row_heights=[0.7,0.15,0.15])
        
    # fig.add_trace(go.Scatter(y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(   x=df['Date'],y=df['Close'], 
                                name="Close",
                                showlegend=False
                                ))
    fig.add_trace(go.Scatter(x=df['Date'],y=df['EMA9'], 
                                opacity=0.7, 
                                line=dict(color='blue', width=1), 
                                name='EMA 9',
                                showlegend=False))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA21'], 
                            opacity=0.7, 
                            line=dict(color='green', width=1), 
                            name='EMA 21',
                            showlegend=False))
    fig.add_trace(go.Scatter(x=df['Date'],y=df['EMA100'], 
                            opacity=0.7, 
                            line=dict(color='orange', width=2), 
                            name='EMA 100',
                            showlegend=False))

    #---------------------------MACD-------------------------------------------
    fig.add_trace(go.Scatter(x=df['Date'],y=df['Signal'],
                            line=dict(color='blue', width=2),name='Signal',showlegend=False
                            ), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'],y=df['MACD'],
                            line=dict(color='red', width=2),name='MACD',showlegend=False
                            ), row=2, col=1)
    colors = ['green' if val >= 0 
            else 'red' for val in df['Hist']]
    fig.add_trace(go.Bar(x=df['Date'], 
                        y=df['Hist'],
                        marker_color=colors,name='Histogram',showlegend=False
                        ), row=2, col=1)
    #---------------------------RSI-------------------------------------------
    fig.add_trace(go.Scatter(x=df['Date'],y=df['RSI'],
                            line=dict(color='blue', width=2),name='RSI',showlegend=False
                            ), row=3, col=1)

    fig.update_xaxes(tickangle=-45)
    fig.layout.update(template='none',title_text=caption)
    fig.update_layout(margin=go.layout.Margin(l=40,r=25,t=25))
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(tickformat=',d')
    
    
    st.plotly_chart(fig, use_container_width=True)

def getmacd(df,a,b,c):
    close = df['Close']
    exp1 = close.ewm(span=a, adjust=False).mean()
    exp2 = close.ewm(span=b, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=c, adjust=False).mean()
    return macd,signal

def plot_ohlc_data(data,caption):

    data = data.iloc[-150:]
    data['EMA9']=data['Close'].ewm(span=9, adjust=True).mean()
    data['EMA21']=data['Close'].ewm(span=21, adjust=True).mean()
    data['EMA100']=data['Close'].ewm(span=100, adjust=True).mean()
    macd,signal = getmacd(data,12,26,9)
    data['MACD']=macd
    data['Signal']=signal
    data['Hist']=macd-signal
    data['RSI'] = pandas_ta.rsi(data['Close'], length = 14)
    data['RSI-EMA'] = data['RSI'].ewm(span=9, adjust=True).mean()
    df = data.iloc[30:]

    
    # fig = go.Figure()
    fig = make_subplots(    rows=3,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.0, 
                        row_heights=[0.7,0.15,0.15])
    
    fig.add_trace(go.Candlestick(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], 
                    showlegend=False))
    # fig.add_trace(go.Scatter(x=df['Date'],
    #                              y=df['Close'], 
    #                              showlegend=True,
    #                              line=dict(color='blue', width=2),
    #                              name='Close'))
    # add moving average traces
    fig.add_trace(go.Scatter(x=df['Date'], 
                            y=df['EMA9'], 
                            opacity=0.7, 
                            line=dict(color='blue', width=1), 
                            # name='EMA 9',
                            showlegend=False))
    fig.add_trace(go.Scatter(x=df['Date'], 
                            y=df['EMA21'], 
                            opacity=0.7, 
                            line=dict(color='green', width=1), 
                            # name='EMA 21',
                            showlegend=False))
    fig.add_trace(go.Scatter(x=df['Date'], 
                    y=df['EMA100'], 
                    opacity=0.7, 
                    line=dict(color='orange', width=2), 
                    # name='EMA 100',
                    showlegend=False))

    #---------------------------MACD-------------------------------------------
    fig.add_trace(go.Scatter(x=df['Date'],y=df['Signal'],
                            line=dict(color='blue', width=2),name='Signal',showlegend=False
                            ), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'],y=df['MACD'],
                            line=dict(color='red', width=2),name='MACD',showlegend=False
                            ), row=2, col=1)
    colors = ['green' if val >= 0 
            else 'red' for val in df['Hist']]
    fig.add_trace(go.Bar(x=df['Date'], 
                        y=df['Hist'],
                        marker_color=colors,name='Histogram',showlegend=False
                        ), row=2, col=1)
    #---------------------------RSI-------------------------------------------
    fig.add_trace(go.Scatter(x=df['Date'],y=df['RSI'],
                            line=dict(color='blue', width=2),name='RSI',showlegend=False
                            ), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'],y=df['RSI-EMA'],
                                line=dict(color='green', width=1),name='RSI-EMA',showlegend=False
                                ), row=3, col=1)

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]), #hide weekends
        ]
    )
    
    fig.layout.update(template='none',title_text=caption,xaxis_rangeslider_visible=False)
    fig.update_layout(margin=go.layout.Margin(l=40,r=25,t=25))   
    
    
    st.plotly_chart(fig, use_container_width=True)   

def showPlot_KMI_EntryExit(df,
                        params=[5,-3.5,4,12,12,26,9,25]
                        ):
    


    # df = df.reset_index()
    # df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = df.set_index('Date')

    dipDays=params[0]
    dipThresh=params[1]
    
    df['emaShort']=df['Close'].ewm(span=params[6], adjust=True).mean()
    df['emaMid']=df['Close'].ewm(span=params[7], adjust=True).mean()
    df['RSI'] = pandas_ta.rsi(df['Close'], length = 14)
    df['RSI-SMA'] = df['RSI'].ewm(span=14, adjust=True).mean()

    macd,signal = getmacd(df,params[2],params[3],9)
    df['Sh_MACD']=macd
    df['Sh_Signal']=signal
    
    macd,signal = getmacd(df,params[4],params[5],9)
    df['MACD']=macd
    df['Signal']=signal

    df=df.iloc[30:] #TODO: Fixed number of days clipping to avoid RSI and ATR
    df = df.iloc[-100:]

    x=df.iloc[-1]
    outStr =("Sh_MACD_Signal:"+str(x.Sh_Signal<x.Sh_MACD)+"\n"+
            "Quick_Histogram:"+str(math.floor(x.Sh_MACD-x.Sh_Signal))+"\n"+
            "Long_Histogram:"+str(math.floor(x.MACD-x.Signal))+"\n"+
            "Closing:"+str(df.iloc[-1].Close)
            )
    
    plt.rcParams['figure.figsize'] = (15, 10)

    fig, axs = plt.subplots(4, sharex=True, gridspec_kw={'height_ratios':[1,3,1,1]})
    plt.subplots_adjust(hspace=.0)
    plt.xticks(rotation=40)

    #RSI
    axs[0].plot(df['RSI'], color='green',linewidth='1', label="RSI")
    axs[0].plot(df['RSI-SMA'], color='blue',linewidth='1', label="SMA")
    axs[0].axhline(y=70, color='r', linestyle='--',linewidth='0.75')
    axs[0].axhline(y=30, color='b', linestyle='--',linewidth='0.75')
    axs[0].axhline(y=50, color='g', linestyle='--',linewidth='1')
    axs[0].axis(ymin=20,ymax=80)

    ##------------------------CandleStick Plot------------------------
    prices = df

    #define width of candlestick elements
    width = 0.9
    width2 = .1

    #define up and down prices
    up = prices[prices.Close>=prices.Open]
    down = prices[prices.Close<prices.Open]

    #define colors to use
    col1 = 'green'
    col2 = 'red'

    #plot up prices
    axs[1].bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1, alpha=0.35)
    axs[1].bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1, alpha=0.35)
    axs[1].bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1, alpha=0.35)

    #plot down prices
    axs[1].bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2, alpha=0.35)
    axs[1].bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2, alpha=0.35)
    axs[1].bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2, alpha=0.35)

    axs[1].plot(df['Close'],color='blue', label="Close")
    
    #--------------------------------------Dip Detection
    df['change'] = df['Close'].pct_change()*100
    x = df['change'].rolling(dipDays).sum()
    allDips = x[x<-2.0]
    for i in range(len(allDips)):
        axs[1].axvline(allDips.index[i], color ='orange',linestyle='--')
    allDips = x[x<dipThresh]
    for i in range(len(allDips)):
        axs[1].axvline(allDips.index[i], color ='r',linestyle='--')
    ##------------------------CandleStick Plot------------------------


    axs[1].plot(df['emaShort'],linestyle='--', label="EMA-Short")
    axs[1].plot(df['emaMid'],linestyle='--', label="EMA-Mid")

    axs[1].fill_between(df.index, max(df.Close), min(df.Close),  
                where=(df.Sh_MACD>df.Sh_Signal) , color = 'green', alpha = 0.1)

    ax = plt.gca()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.1, 0.1, outStr, transform=ax.transAxes, fontsize=14,verticalalignment='bottom', bbox=props)

    #Sh_MACD
    
    axs[2].plot(df['Sh_Signal'], color='orange',linewidth='1', label="Signal")
    axs[2].plot(df['Sh_MACD'], color='green',linewidth='1', label="MACD")
    axs[2].axhline(y=0, color='r', linestyle='--',linewidth='1')
        
    axs[3].plot(df['Signal'], color='orange',linewidth='1', label="Signal")
    axs[3].plot(df['MACD'], color='green',linewidth='1', label="MACD")
    axs[3].axhline(y=0, color='r', linestyle='--',linewidth='1')

    x=df.iloc[-1]
    outStr =("Sh_MACD_Signal:"+str(x.Sh_MACD<x.Sh_Signal)+"\n"+
            "Long_Histogram:"+str(x.MACD-x.Signal)
            )

    for ax in axs:
        ax.label_outer()
        ax.grid()
        ax.legend(loc="upper left")
        
        # ax.grid(which='minor', linestyle='--')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

    
    #plt.savefig(direc+fileName+'.png', bbox_inches='tight')
    #plt.close()
    
    st.pyplot(fig)

def showPlot_KMI_ST_EntryExit(df,
                        params=[5,-3.5,4,12,12,26,9,25]
                        ):
    df = df.set_index('Date')

    dipDays=params[0]
    dipThresh=params[1]

    sti = ta.supertrend(df['High'], df['Low'], df['Close'], 7, 1)
    df['ST_Sh_s'] = sti['SUPERTs_7_1.0']
    df['ST_Sh_l'] = sti['SUPERTl_7_1.0']

    sti = ta.supertrend(df['High'], df['Low'], df['Close'], 7, 2)
    df['ST_Mi_s'] = sti['SUPERTs_7_2.0']
    df['ST_Mi_l'] = sti['SUPERTl_7_2.0']

    sti = ta.supertrend(df['High'], df['Low'], df['Close'], 7, 3)
    df['ST_Lo_s'] = sti['SUPERTs_7_3.0']
    df['ST_Lo_l'] = sti['SUPERTl_7_3.0']

    outStr = ""

    if(math.isnan(df.iloc[-1].ST_Sh_s)):
        outStr += "Short_ST:Long\n"
    else:
        outStr += "Short_ST:Short\n"

    if(math.isnan(df.iloc[-1].ST_Mi_s)):
        outStr += "Mid_ST:Long\n"
    else:
        outStr += "Mid_ST:Short\n"

    if(math.isnan(df.iloc[-1].ST_Lo_s)):
        outStr += "Long_ST:Long"
    else:
        outStr += "Long_ST:Short"

    df['RSI'] = pandas_ta.rsi(df['Close'], length = 14)
    df['RSI-SMA'] = df['RSI'].ewm(span=14, adjust=True).mean()

    macd,signal = getmacd(df,params[4],params[5],9)
    df['MACD']=macd
    df['Signal']=signal

    df=df.iloc[30:] #TODO: Fixed number of days clipping to avoid RSI and ATR
    df = df.iloc[-100:]

    plt.rcParams['figure.figsize'] = (15, 10)

    fig, axs = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios':[1,3,1]})
    plt.subplots_adjust(hspace=.0)
    plt.xticks(rotation=40)

    #RSI
    axs[0].plot(df['RSI'], color='green',linewidth='1', label="RSI")
    axs[0].plot(df['RSI-SMA'], color='blue',linewidth='1', label="SMA")
    axs[0].axhline(y=70, color='r', linestyle='--',linewidth='0.75')
    axs[0].axhline(y=30, color='b', linestyle='--',linewidth='0.75')
    axs[0].axhline(y=50, color='g', linestyle='--',linewidth='1')
    axs[0].axis(ymin=20,ymax=80)

    ##------------------------CandleStick Plot------------------------
    prices = df

    #define width of candlestick elements
    width = 0.9
    width2 = .1

    #define up and down prices
    up = prices[prices.Close>=prices.Open]
    down = prices[prices.Close<prices.Open]

    #define colors to use
    col1 = 'green'
    col2 = 'red'

    #plot up prices
    axs[1].bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1, alpha=0.35)
    axs[1].bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1, alpha=0.35)
    axs[1].bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1, alpha=0.35)

    #plot down prices
    axs[1].bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2, alpha=0.35)
    axs[1].bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2, alpha=0.35)
    axs[1].bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2, alpha=0.35)

    axs[1].plot(df['Close'],color='blue', label="Close")

    #--------------------------------------Dip Detection
    df['change'] = df['Close'].pct_change()*100
    x = df['change'].rolling(dipDays).sum()
    allDips = x[x<dipThresh]
    for i in range(len(allDips)):
        axs[1].axvline(allDips.index[i], color ='r',linestyle='--')
    ##------------------------CandleStick Plot------------------------


    # axs[1].plot(df['emaShort'],linestyle='--')
    # axs[1].plot(df['emaMid'],linestyle='--')
    axs[1].plot(df['ST_Sh_l'],color='green',linewidth='1',linestyle='-')#,label="ST_Sh_l")
    axs[1].plot(df['ST_Mi_l'],color='green',linewidth='2',linestyle='-')#,label="ST_Mi_l")
    axs[1].plot(df['ST_Lo_l'],color='green',linewidth='1',linestyle='-')#,label="ST_Lo_l")

    axs[1].plot(df['ST_Sh_s'],color='red',linewidth='1',linestyle='-')#,label="ST_Sh_s")
    axs[1].plot(df['ST_Mi_s'],color='red',linewidth='2',linestyle='-')#,label="ST_Mi_s")
    axs[1].plot(df['ST_Lo_s'],color='red',linewidth='1',linestyle='-')#,label="ST_Lo_s")

    axs[1].fill_between(df.index, max(df.Close), min(df.Close),  
                where=(df.MACD>df.Signal) , color = 'green', alpha = 0.1)

    ax = plt.gca()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.1, 0.1, outStr, transform=ax.transAxes, fontsize=14,verticalalignment='bottom', bbox=props)

        
    axs[2].plot(df['Signal'], color='orange',linewidth='1', label="Signal")
    axs[2].plot(df['MACD'], color='green',linewidth='1', label="MACD")
    axs[2].axhline(y=0, color='r', linestyle='--',linewidth='1')

    for ax in axs:
        ax.label_outer()
        ax.grid()
        ax.legend(loc="upper left")
        
        # ax.grid(which='minor', linestyle='--')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

    st.pyplot(fig)

def parseDPS2DataFrame(dataURL):
    output = []
    for i in  range(len(dataURL['data'])):
        tsdt = datetime.fromtimestamp(int(dataURL['data'][i][0])) + timedelta(hours=5)
        ts = tsdt.strftime('%d/%m/%Y %H:%M:%S')
        close = dataURL['data'][i][1]
        vol = dataURL['data'][i][2]
        output.append([ts,close,vol])
    data = pd.DataFrame(output, columns = ['Date', 'Close', 'Volume'])
    return data.iloc[::-1]

def show_KMI_Plotly(df,
                        params=[5,-3.5,4,12,12,26,9,25]):
    data = df.iloc[-100:]
    data['EMA9']=data['Close'].ewm(span=9, adjust=True).mean()
    data['EMA21']=data['Close'].ewm(span=21, adjust=True).mean()
    data['EMA100']=data['Close'].ewm(span=50, adjust=True).mean()
    macd,signal = getmacd(data,12,26,9)
    data['MACD']=macd
    data['Signal']=signal
    data['Hist']=macd-signal
    data['RSI'] = pandas_ta.rsi(data['Close'], length = 14)
    data['RSI-EMA'] = data['RSI'].ewm(span=9, adjust=True).mean()

    macd,signal = getmacd(df,params[2],params[3],9)
    data['Sh_MACD']=macd
    data['Sh_Signal']=signal
    data['Sh_Hist']=macd-signal

    df = data.iloc[:]


    # fig = go.Figure()
    fig = make_subplots(    rows=4,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.0, 
                            row_heights=[0.55,0.15,0.15,0.15])

    fig.add_trace(go.Candlestick(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], 
                    showlegend=False))

    fig.add_trace(go.Scatter(x=df['Date'], 
                                y=df['EMA9'], 
                                opacity=0.7, 
                                line=dict(color='blue', width=1), 
                                name='EMA 9',
                                showlegend=False))
    fig.add_trace(go.Scatter(x=df['Date'], 
                            y=df['EMA21'], 
                            opacity=0.7, 
                            line=dict(color='green', width=1), 
                            name='EMA 21',
                            showlegend=False))
    fig.add_trace(go.Scatter(x=df['Date'], 
                    y=df['Close'], 
                    opacity=0.7, 
                    line=dict(color='brown', width=3), 
                    name='Close',
                    showlegend=False))

    #====================Long MACD========================================
    fig.add_trace(go.Scatter(x=df['Date'],y=df['Signal'],
                                line=dict(color='blue', width=2),name='Signal',showlegend=False
                                ), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'],y=df['MACD'],
                            line=dict(color='red', width=2),name='MACD',showlegend=False
                            ), row=3, col=1)
    colors = ['green' if val >= 0 
            else 'red' for val in df['Hist']]
    fig.add_trace(go.Bar(x=df['Date'], 
                        y=df['Hist'],
                        marker_color=colors,name='Histogram',showlegend=False
                        ), row=3, col=1)
    #====================Long MACD========================================

    #====================Short MACD========================================
    fig.add_trace(go.Scatter(x=df['Date'],y=df['Sh_Signal'],
                                line=dict(color='blue', width=2),name='Sh_Signal',showlegend=False
                                ), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'],y=df['Sh_MACD'],
                            line=dict(color='red', width=2),name='Sh_MACD',showlegend=False
                            ), row=2, col=1)
    colors = ['green' if val >= 0 
            else 'red' for val in df['Sh_Hist']]
    fig.add_trace(go.Bar(x=df['Date'], 
                        y=df['Sh_Hist'],
                        marker_color=colors,name='Histogram',showlegend=False
                        ), row=2, col=1)
    #====================Short MACD========================================

    fig.add_trace(go.Scatter(x=df['Date'],y=df['RSI'],
                                line=dict(color='blue', width=1),name='RSI',showlegend=False
                                ), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['Date'],y=df['RSI-EMA'],
                                line=dict(color='green', width=1),name='RSI-EMA',showlegend=False
                                ), row=4, col=1)

    fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]), #hide weekends
            ]
        )
        
    x=df.iloc[-1]
    caption =(  "Closing:"+str(math.floor(x.Close))+" <br> "+
                "Sh_MACD_Signal:"+str(x.Sh_MACD>x.Sh_Signal)+" ::: "+
                "MACD_Signal:"+str(x.MACD>x.Signal)+" <br> "+
                "Long_Histogram:"+str(math.floor(x.MACD-x.Signal))
            )

    fig.update_yaxes(tickformat=',d')
    fig.layout.update(template='none',title_text=caption,xaxis_rangeslider_visible=False)
    fig.update_layout(  margin=go.layout.Margin(r=25,t=25),
                        height = 800
                        )

    st.plotly_chart(fig, use_container_width=True)  

def plotBasicTA(ds,params):
    [shortLen,midLen,longLen,macdShortLen,macdLongLen] = params

    df = ds.copy()
    df=df.set_index('Date')
    df = df.iloc[-150:]
    df['emaShort']=df['Close'].ewm(span=shortLen, adjust=True).mean()
    df['emaMid']=df['Close'].ewm(span=midLen, adjust=True).mean()
    df['emaLong']=df['Close'].ewm(span=longLen, adjust=True).mean() 

    df['RSI'] = pandas_ta.rsi(df['Close'], length = 14)
    df['RSI-SMA'] = df['RSI'].ewm(span=14, adjust=True).mean()
            
    macd,signal = getmacd(df,macdShortLen,macdLongLen,9)
    df['MACD']=macd
    df['Signal']=signal        

    df=df.iloc[50:] #TODO: Fixed number of days clipping to avoid RSI and ATR

    plt.rcParams['figure.figsize'] = (15, 10)

    fig, axs = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios':[1,3,1]})
    plt.subplots_adjust(hspace=.0)
    plt.xticks(rotation=40)
    plt.locator_params(axis='x', nbins=10)

    axs[0].plot(df['RSI'], color='green',linewidth='1', label="RSI")
    axs[0].plot(df['RSI-SMA'], color='blue',linewidth='1', label="RSI-EMA")
    axs[0].axhline(y=70, color='r', linestyle='--',linewidth='0.75')
    axs[0].axhline(y=30, color='b', linestyle='--',linewidth='0.75')
    axs[0].axhline(y=50, color='g', linestyle='--',linewidth='1')
    axs[0].axis(ymin=20,ymax=80)

    ##------------------------CandleStick Plot------------------------
    if(df.isnull().sum().Open<3):
        prices = df

        #define width of candlestick elements
        width = 0.9
        width2 = .1

        #define up and down prices
        up = prices[prices.Close>=prices.Open]
        down = prices[prices.Close<prices.Open]

        #define colors to use
        col1 = 'green'
        col2 = 'red'

        #plot up prices

        axs[1].bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1, alpha=0.35)
        axs[1].bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1, alpha=0.35)
        axs[1].bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1, alpha=0.35)

        #plot down prices
        axs[1].bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2, alpha=0.35)
        axs[1].bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2, alpha=0.35)
        axs[1].bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2, alpha=0.35)

    axs[1].fill_between(df.index, max(df.Close), min(df.Close),  
                where=(df.MACD>df.Signal) , color = 'green', alpha = 0.1)

    axs[1].plot(df['Close'],color='blue', label="Close")
    ##------------------------CandleStick Plot------------------------


    axs[1].plot(df['emaShort'],linestyle='--')
    axs[1].plot(df['emaMid'],linestyle='--')
    axs[1].plot(df['emaLong'],linewidth='2',linestyle='--')

    x=df.iloc[-1]
    xm1 = df.iloc[-2]
    outStr =("Closing:"+str(x.Close)+"\n"+
            "Histogram:"+str(x.MACD-x.Signal)+"\n"+
            "Histogram[n-1]:"+str(xm1.MACD-xm1.Signal)+"\n"+
            str(midLen)+"-EMA:"+str(x.emaMid)+"\n"+
            str(shortLen)+"-EMA:"+str(x.emaShort)+"\n"+
            "RSI:"+str(math.floor(x.RSI))            
            )
    ax = plt.gca()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.1, 0.1, outStr, transform=ax.transAxes, fontsize=14,verticalalignment='bottom', bbox=props)

    ax = plt.gca()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    axs[2].plot(df['Signal'], color='orange',linewidth='1', label="Signal")
    axs[2].plot(df['MACD'], color='green',linewidth='1', label="MACD")
    axs[2].axhline(y=0, color='r', linestyle='-',linewidth='1')

    for ax in axs:
        ax.label_outer()
        ax.grid()
        ax.legend(loc="upper left")
        ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    
    st.pyplot(fig)

st.header('QuickData Analyzer!!!')

plotDefault = False

urlParams = st.experimental_get_query_params()
if ('stock' in urlParams):
    valDefault = urlParams['stock'][0]
    plotDefault = True

# t.tic()
dns.resolver.default_resolver=dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers=['8.8.8.8']
myclient = pymongo.MongoClient("mongodb+srv://readonly:readonly@cluster0.ss8kmkn.mongodb.net/?retryWrites=true&w=majority")
db = myclient["StockData"]
mo = MongoObject(db)
userCol = db['userData']
gotCol = db['gotData']
watchCol = db['watchData']
st.success('MongoDB connected!')
# st.success('MongoDB connected! Took '+str(round(t.tocvalue(),2))+' sec for MongoConnection')

plotDefault = False

urlParams = st.experimental_get_query_params()
if ('stock' in urlParams):
    valDefault = urlParams['stock'][0]
    plotDefault = True

if plotDefault:
    data = mo.getQuickData(valDefault)
    plot_raw_data(data.reset_index(),'Quick Data for '+valDefault)

else:
    stocks = db.dailyData.distinct("symbol")

    # st.text(stocks)
    def remove_values_from_list(the_list, val):
        return [value for value in the_list if value != val]

    stocks = remove_values_from_list(stocks,'Karachi 100')
    stocks = remove_values_from_list(stocks,'Karachi Meezan 30')
    stocks = remove_values_from_list(stocks,'OGTi')
    stocks = remove_values_from_list(stocks,'BKTi')

    stocks.insert(0, 'BKTi')
    stocks.insert(0, 'OGTi')
    stocks.insert(0, 'Karachi Meezan 30')
    stocks.insert(0, 'Karachi 100')
    
    with st.form('Add Stocks',clear_on_submit=False):
        stocksToAdd = st.selectbox('Ticker*:',stocks)
        addStockButton = st.form_submit_button('Analyze!!!')

        if addStockButton: 
            data = mo.getQuickData(stocksToAdd)
            plot_raw_data(data.reset_index(),'Quick Data for '+stocksToAdd)
    
    
    # stocksToAdd = mo.getGotStocks('asifrajput')['actualSym'].to_list()
    # stocksToAdd = ['Karachi 100','SYS','TRG','NETSOL','AVN']
    # stocksToAdd = mo.getWatchStocks('asifrajput')['actualSym'].to_list()

    # for i in range(len(stocksToAdd)):
    #     stock = stocksToAdd[i]
    #     t.tic()
    #     data = mo.getQuickData(stock)
    #     # st.text(t.tocvalue())

    #     t.tic()
    #     plot_raw_data(data.reset_index(),'Quick Data for '+stock)
    #     # st.text(t.tocvalue())
