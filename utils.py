import os
import re
import sys
import pandas as pd
import numpy as np
import datetime as dt
import time
import operator
import random

from pandas import DataFrame
from pandas.tseries.offsets import BDay # business days
from pandas.io.common import urlopen

#conda install -c https://conda.anaconda.org/anaconda pandas-datareader
import pandas_datareader.data as web
import pandas_datareader as pdr
from pandas_datareader._utils import RemoteDataError

from bs4 import BeautifulSoup

import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

from yahoo_finance import Share

# matplotlib
import matplotlib.pyplot as plt

# Exchange symbols:
#   NMS = NasdaqGS; NGM = NasdagGM; NCM = NasdaqCM; ASE = AMEX; NYQ = NYSE;
#   GER = XETRA; FRA = Frankfurt; PNK = Other OTC; LSE = LSE
EXCH_SYM_TO_STR = {'NMS':'NASDAQ', 'NGM':'NASDAQ', 'NCM':'NASDAQ', 'NYQ':'NYSE', 'ASE':'AMEX',
                   'GER':'XETRA', 'FRA': 'FRA', 'LSE':'LSE'}
STR_TO_EXCH_SYM = {'NASDAQ':'NMS','NYSE':'NYQ', 'AMEX':'ASE'}
DEFAULT_START_DATE = '2000-01-01'

def get_exchange_by_sym(sym):
    if sym in EXCH_SYM_TO_STR.keys():
        return EXCH_SYM_TO_STR[sym]
    else:
        return 'NYSE'

def to_date(d):
    """
    Convert input to datetime.date
    """
    if type(d) == dt.date:
        return d
    elif type(d) == str:
        return pd.to_datetime(d).date()
    elif type(d) == pd.Timestamp:
        return d.date()
    else:
        print('Error: unsupported type.')
        return dt.datetime.today().date()

def parse_start_end_date(start, end):
    """
    Convert input date time to datetime.date.
    """
    if start == None:
        start = DEFAULT_START_DATE
    if type(start) == str:
        start_date = pd.to_datetime(start).date()
    elif type(start) == pd.Timestamp:
        start_date = start.date()
    else:
        start_date = start
    if end != None:
        if type(end) == str:
            end_date = pd.to_datetime(end).date()
        elif type(end) == pd.Timestamp:
            end_date = end.date()
        else:
            end_date = end
    else:
        end_date = dt.datetime.today().date()
    return [start_date, end_date]

def get_stats_intervals(end=None):
    if end == None:
        end_date = dt.date.today()
    else:
        end_date = end
    one_week_ago = end_date - dt.timedelta(days=7)
    one_month_ago = end_date - dt.timedelta(days=30)
    three_month_ago = end_date - dt.timedelta(days=90)
    half_year_ago = end_date - dt.timedelta(days=180)
    one_year_ago = end_date - dt.timedelta(days=365)
    two_year_ago = end_date - dt.timedelta(days=365*2)
    three_year_ago = end_date - dt.timedelta(days=365*3)
    five_year_ago = end_date - dt.timedelta(days=365*5)
    return [end_date, one_week_ago, one_month_ago, three_month_ago, half_year_ago, one_year_ago, two_year_ago, three_year_ago, five_year_ago]

def str2list(symbols, split='+'):
    """
    Convert different types of symbols into list.
    For the input symbols string, multiple symbols can be concatenated by '+', e.g. 'AAPL+NKE+^GSPC'
    """
    if type(symbols) == list:
        return symbols
    elif type(symbols) == pd.Series:
        return symbols.tolist()
    elif type(symbols) == str:
        # Support multiple symbols separated by '+', e.g. 'AAPL+NKE+^GSPC'
        return symbols.split(split)
    else:
        print("ERROR: unsupported input type %s" %type(symbols))
        return None

def str2num(s, m2b=False):
    """
    Convert financial data from string to integer/float number.

    s: input string
    m2b: convert Million('M') to Billion('B')

    Example:
        '-1.2031%' => -0.012031
        '21,065,937' => 21065937
        '158.86B' => 158.86
        '158.86M' => 0.15886
    """
    if s == None:
        return np.nan
    if type(s) == int or type(s) == float or type(s) == np.float64:
        return s
    if len(s) == 0:
        return np.nan
    if type(s) != str:
        print('Error: str2num: inavlid input.')
        return np.nan
    s = s.upper()
    if s == '-' or s == 'N/A' or s == 'NA':
        return np.nan
    factor = 1.0
    if s[0] == '-':
        factor *= -1
    if s[-1] == '%':
        factor /= 100
    elif s[-1] == 'M' and m2b:
        factor /= 1000 # million to billion
    elif s[-1] == 'B' and not m2b:
        factor *= 1000 # billion to million
    num = s.replace(',','').replace('-','').replace('+','').replace('%','').replace('M','').replace('B','')
    return float(num)*factor

#financial_fmt = lambda y: pd.Series([str(x).replace(',','').replace('-','0') for x in y], index=y.index)[::-1].astype(np.float)
def financial_fmt(y):
    """
    Convert Google Financial inputs(Pandas Series) into a Series of numbers.
    """
    out = []
    for s in y:
        out.append(str2num(s))
    return pd.Series(out, index=y.index)[::-1]

def min_max_norm(x):
    """
    Min-Max normalization.

    x: Numpy array or Pandas Series
    """
    return (x-x.min()) / (x.max()-x.min())

def get_symbol_names(symbols):
    """
    Get a list of symbols' names from Yahoo Finance.
    In case comma is in the symbol's name, download it separately.
    """
    sym_list = str2list(symbols)
    if sym_list == None:
        return None

    url_str = 'http://download.finance.yahoo.com/d/quotes.csv?'
    # Form a BUNCH of STOCK SYMBOLS separated by "+",
    # e.g. XOM+BBDb.TO+JNJ+MSFT
    sym_str = '+'.join(sym_list)
    url_str += 's=' + sym_str

    tags = {'s':'Symbol', 'n':'Name'}
    url_str += '&f=' + ''.join(pd.compat.iterkeys(tags))
    with urlopen(url_str) as resp:
        raw = resp.read()
    lines = raw.decode('utf-8').strip().split('\n')
    """
    Examples:
        '"YUM","Yum! Brands, Inc."',
        '"ZBH","Zimmer Biomet Holdings, Inc. Co"',
        '"ZION","Zions Bancorporation"',
        '"ZTS","Zoetis Inc. Class A Common Stoc"']
    """
    lines = [line.strip().strip('"').replace('",', '').split('"') for line in lines]
    symnames = DataFrame(lines, columns=list(tags.values()))
    symnames = symnames.drop_duplicates()
    symnames = symnames.set_index('Symbol')
    return symnames

def get_symbol_yahoo_stats_url(symbols):
    """
    Get the symbols' basic statistics from Yahoo Finance.
    Input:
       symbols - a list of symbol strings, e.g. ['AAPL']
    Output: stats in Pandas DataFrame.
    This function is ported from pandas_datareader/yahoo/components.py
    """
    sym_list = str2list(symbols)
    if sym_list == None:
        return DataFrame()

    url_str = 'http://download.finance.yahoo.com/d/quotes.csv?'
    # Form a BUNCH of STOCK SYMBOLS separated by "+",
    # e.g. XOM+BBDb.TO+JNJ+MSFT
    sym_str = '+'.join(sym_list)
    url_str += 's=' + sym_str
    url_str = url_str.strip().replace(' ','') # remove all spaces

    # Yahoo Finance tags, refer to http://www.financialwisdomforum.org/gummy-stuff/Yahoo-data.htm
    # More tags: https://github.com/herval/yahoo-finance/blob/master/lib/yahoo-finance.rb
    tags = {'s':'Symbol', 'x':'Exchange', 'j1':'Market Cap', 'b4':'Book Value', 'r':'P/E', 'p5':'Price/Sales',
            'p6':'Price/Book', 'j4':'EBITDA', 'j':'52-week Low', 'k':'52-week High', 'l1':'Last Trade',
            'd':'Dividend/Share', 'y':'Dividend Yield', 'e':'EPS', 's7':'Short Ratio', 's1':'Shares Owned',
            'f6':'Float Shares'}
    url_str += '&f=' + ''.join(pd.compat.iterkeys(tags))
    with urlopen(url_str) as resp:
        raw = resp.read()
    lines = raw.decode('utf-8').strip().replace('"', '').split('\n')
    lines = [line.strip().split(',') for line in lines]
    if len(lines) < 1 or len(lines[0]) < len(tags) :
        print('Error: failed to download Yahoo stats from %s' %url_str)
        return DataFrame()
    stats = DataFrame(lines, columns=list(tags.values()))
    stats = stats.drop_duplicates()
    stats = stats.set_index('Symbol')
    return stats

def get_symbol_yahoo_stats_yql(symbols, exclude_name=False):
    """
    Get the symbols' basic statistics from Yahoo Finance.
    Input:
       symbols - a list of symbol strings, e.g. ['AAPL']
    Output: stats in Pandas DataFrame.
    This function is ported from pandas_datareader/yahoo/components.py
    """
    sym_list = str2list(symbols)
    if sym_list == None:
        return DataFrame()

    # Yahoo Finance tags, refer to http://www.financialwisdomforum.org/gummy-stuff/Yahoo-data.htm
    tags = ['Symbol']
    if not exclude_name:
        tags += ['Name'] 
    real_tags = ['MarketCap', 'Volume', 'AverageDailyVolume', 'BookValuePerShare', 'P/E', 'PEG', 'Price/Sales',
            'Price/Book', 'EBITDA', 'EPS', 'EPSEstimateNextQuarter', 'EPSEstimateCurrentYear', 'EPSEstimateNextYear',
            'OneyrTargetPrice', 'PriceEPSEstimateCurrentYear', 'PriceEPSEstimateNextYear', 'ShortRatio',
            'Dividend/Share', 'DividendYield', 'DividendPayDate', 'ExDividendDate']
    tags += real_tags

    lines = []
    for sym in sym_list:
        line = [sym]
        # download stats
        success = False
        for n_tries in range(0,4): # try at most 4 times
            try:
                stock = Share(sym)
                success = True
                break
            except:
                print('Warning: YQL query faied for %s, try again...' %sym)
                time.sleep(0.1 * random.randint(0,10))
                continue
        
        if success:
            if not exclude_name:
                line += [stock.get_name()]
            line += [str2num(stock.get_market_cap(), m2b=True), str2num(stock.get_volume()),
                    str2num(stock.get_avg_daily_volume()), str2num(stock.get_book_value()),
                    str2num(stock.get_price_earnings_ratio()), str2num(stock.get_price_earnings_growth_ratio()),
                    str2num(stock.get_price_sales()), str2num(stock.get_price_book()), str2num(stock.get_ebitda()),
                    str2num(stock.get_earnings_share()), str2num(stock.get_EPS_estimate_next_quarter()),
                    str2num(stock.get_EPS_estimate_current_year()), str2num(stock.get_EPS_estimate_next_year()),
                    str2num(stock.get_one_yr_target_price()), str2num(stock.get_price_EPS_estimate_current_year()),
                    str2num(stock.get_price_EPS_estimate_next_year()), str2num(stock.get_short_ratio()),
                    str2num(stock.get_dividend_share()), str2num(stock.get_dividend_yield()), stock.get_dividend_pay_date(),
                    stock.get_ex_dividend_date()]
            lines.append(line)
        else:
            print('!!!Error: failed to get stats from YQL for sym %s!!!' %sym)
            if not exclude_name:
                line += ["N/A"]
            line += [0]*len(real_tags)
            lines.append(line)

    stats = DataFrame(lines, columns=tags)
    stats = stats.drop_duplicates()
    stats = stats.set_index('Symbol')
    return stats

def get_symbol_yahoo_stats(symbols, exclude_name=False):
    return get_symbol_yahoo_stats_yql(symbols, exclude_name)

def get_symbol_exchange(sym):
    """
    Download the stock exchange sym from Yahoo Finance.
    YQL is required.
    """
    stock = Share(sym)
    return stock.get_stock_exchange()

def moving_average(x, n=10, type='simple'):
    """
    Calculate simple/exponential moving average.

    Inputs:
        x - list, Numpy array, Pandas Series
        n - window of the moving average
        type - 'simple' or 'exponential'
    Return: pandas Series with the same length as input x.

    Exponential Moving Average(EMA), calculated by
        SMA: 10 period sum / 10 
        Multiplier: (2 / (Time periods + 1) ) = (2 / (10 + 1) ) = 0.1818 (18.18%)
        EMA: {Close - EMA(previous day)} x multiplier + EMA(previous day).
    """
    x = np.asarray(x)
    if type == 'simple':
        # SMA
        w = np.ones(n)
        w /= w.sum() # weights
        avg = np.convolve(x, w, mode='full')[:len(x)]
        avg[:n] = avg[n]
    else:
        # EMA
        avg = np.zeros_like(x)
        avg[:n] = x[:n].mean() # initialization
        m = 2/(n+1) # multiplier
        for i in np.arange(n,len(x)):
            avg[i] = (x[i] - avg[i-1]) * m + avg[i-1]
    return avg

def find_trend(y, fit_poly=True):
    """
    Find the trend of input data.

    Inputs:
        y: array-like numbers.
        fit_poly: True for fitting polynomial, False for fitting line.
    Return: slope for line or 0 for turnaround.
    """
    if len(y) < 2:
        return 0
    if (True in np.isinf(list(y))) or (True in np.isnan(list(y))):
        return np.nan

    counts = len(y)
    if type(y) == pd.Series:
        # calculate intervals based on the index
        if type(y.index[0]) == str:
            counts = pd.to_datetime(y.last_valid_index()).date() - pd.to_datetime(y.first_valid_index()).date()
            counts = counts.days
        elif type(y.index[0]) == pd.DatetimeIndex:
            counts = y.last_valid_index() - y.first_valid_index()
            counts = counts.days

    x = np.linspace(0, counts, len(y))
    yy = np.array(y, copy=True) # make a copy to avoid changing input data
    if yy[0] == 0:
        yy[0] += 0.0000001
    yy /= yy[0] # normalization
    if len(yy) < 4:
        fit_poly = False

    if fit_poly:
        # line-fitting the first and second half data - if the slopes are both positive or negative,
        # then these data can be fitted by a line. Otherwise, this is a turn over.
        mid = int(len(yy)/2)
        y1 = yy[:mid]
        y2 = yy[mid:]
        x1 = np.arange(len(y1))
        x2 = np.arange(len(y2))
        p1 = np.polyfit(x1, y1, 1)
        p2 = np.polyfit(x2, y2, 1)
        if (p1[0] > 0 and p2[0] < 0) or (p1[0] < 0 and p2[0] > 0):
            return 0 # turnaround

    # these data can only be fitted by a line
    p = np.polyfit(x,yy,1)
    return p[0]


# 
# Plot Candlestick chart, from https://ntguardian.wordpress.com/2016/09/19/introduction-stock-market-data-python-1/.
# Example:
#   plot_candlestick(apple.loc['2016-01-04':'2016-08-07',:], otherseries = "20d")
#
def plot_candlestick(dat, stick = "day", otherseries = None):
    from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
    from matplotlib.finance import candlestick_ohlc

    """
    :param dat: pandas DataFrame object with datetime64 index, and float columns "Open", "High", "Low", and "Close", likely created via DataReader from "yahoo"
    :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines
 
    This will show a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
    """
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    dayFormatter = DateFormatter('%d')      # e.g., 12
 
    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    transdat = dat.loc[:,["Open", "High", "Low", "Close"]]
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1 # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years
            grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                            "High": max(group.High),
                                            "Low": min(group.Low),
                                            "Close": group.iloc[-1,3]},
                                           index = [group.index[0]]))
            if stick == "week": stick = 5
            elif stick == "month": stick = 30
            elif stick == "year": stick = 365
 
    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                        "High": max(group.High),
                                        "Low": min(group.Low),
                                        "Close": group.iloc[-1,3]},
                                       index = [group.index[0]]))
    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')
 
    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)
 
    ax.grid(True)
 
    # Create the candelstick chart
    candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                      plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                      colorup = "black", colordown = "red", width = stick * .4)
 
    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)
 
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()


