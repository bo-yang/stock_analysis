import os
import re
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

#conda install -c https://conda.anaconda.org/anaconda pandas-datareader
import pandas_datareader.data as web
import pandas_datareader as pdr
from pandas import DataFrame
from pandas_datareader._utils import RemoteDataError
from pandas.io.common import urlopen
from bs4 import BeautifulSoup

# conda install -c conda-forge selenium=3.0.1
from selenium import webdriver

# Exchange symbols:
#   NMS = NasdaqGS; NGM = NasdagGM; NCM = NasdaqCM; ASE = AMEX; NYQ = NYSE;
#   GER = XETRA; FRA = Frankfurt; PNK = Other OTC; LSE = LSE
EXCH_SYM_TO_STR = {'NMS':'NASDAQ', 'NGM':'NASDAQ', 'NCM':'NASDAQ', 'NYQ':'NYSE', 'ASE':'AMEX',
                   'GER':'XETRA', 'FRA': 'FRA', 'LSE':'LSE'}
DEFAULT_START_DATE = '2000-01-01'

def get_exchange_by_sym(sym):
    if sym in EXCH_SYM_TO_STR.keys():
        return EXCH_SYM_TO_STR[sym]
    else:
        return 'NYSE'

def parse_start_end_date(start, end):
    """
    Convert date time in string to datetime.
    For datetime inputs, do nothing.
    """
    if start == None:
        start = DEFAULT_START_DATE
    if type(start) == str:
        start_date=pd.to_datetime(start).to_pydatetime()
    else:
        start_date = start
    if end != None:
        if type(end) == str:
            end_date = pd.to_datetime(end).to_pydatetime()
        else:
            end_date = end
    else:
        end_date = dt.date.today()
    return [start_date, end_date]

def get_stats_intervals(end=None):
    if end == None:
        end_date = dt.date.today()
    else:
        end_date = end
    three_month_ago = end_date - dt.timedelta(days=90)
    half_year_ago = end_date - dt.timedelta(days=180)
    one_year_ago = end_date - dt.timedelta(days=365)
    two_year_ago = end_date - dt.timedelta(days=365*2)
    three_year_ago = end_date - dt.timedelta(days=365*3)
    five_year_ago = end_date - dt.timedelta(days=365*5)
    return [end_date, three_month_ago, half_year_ago, one_year_ago, two_year_ago, three_year_ago, five_year_ago]

def _parse_google_financial_table(tables, keyword=None):
    """
    Parse Google Financial table into DataFrame.
    tables - selenium.webdriver.remote.webelement.WebElement
    """
    tbl = None
    for t in tables:
        if len(t.text) <= 1:
            continue
        if keyword != None and keyword not in t.text:
            continue
        else:
            tbl = t
            break
    if tbl == None:
        return None

    lines = tbl.text.strip().splitlines()
    # Get quaters from the first row, e.g.
    # 'In Millions of USD (except for per share items) 3 months ending 2016-10-31 3 months ending 2016-07-31 3 months ending 2016-04-30 3 months ending 2016-01-31 3 months ending 2015-10-31'
    quarters=re.findall(r'([0-9]+-[0-9]+-[0-9]+)', lines[0])
    # Store the following lines into DataFrame
    rows = list()
    for line in lines[1:]:
        l = line.strip().split(' ')
        values = l[-len(quarters):] # the right part
        key = ' '.join(l[:-len(quarters)]) # the left part
        rows.append([key] + values)
    colstr = ['Entries'] + quarters
    fin_df = DataFrame(rows, columns=colstr)
    fin_df = fin_df.drop_duplicates()
    fin_df = fin_df.set_index('Entries')
    return fin_df

def convert_string_to_list(symbols):
    """
    Convert different types of symbols into list.
    For the input symbols string, multiple symbols can be concatenated by '+', e.g. 'AAPL+NKE+^GSPC'
    """
    if type(symbols) == list:
        return symbols
    elif type(symbols) == pd.core.series.Series:
        return symbols.tolist()
    elif type(symbols) == str:
        # Support multiple symbols separated by '+', e.g. 'AAPL+NKE+^GSPC'
        return symbols.split('+')
    else:
        print("ERROR: unsupported input type %s" %type(symbols))
        return None

def get_symbol_names(symbols):
    """
    Get a list of symbols' names from Yahoo Finance.
    In case comma is in the symbol's name, download it separately.
    """
    sym_list = convert_string_to_list(symbols)
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

def get_symbol_yahoo_stats(symbols):
    """
    Get the symbols' basic statistics from Yahoo Finance.
    Input:
       symbols - a list of symbol strings, e.g. ['AAPL']
    Output: stats in Pandas DataFrame.
    This function is ported from pandas_datareader/yahoo/components.py
    """
    sym_list = convert_string_to_list(symbols)
    if sym_list == None:
        return None

    url_str = 'http://download.finance.yahoo.com/d/quotes.csv?'
    # Form a BUNCH of STOCK SYMBOLS separated by "+",
    # e.g. XOM+BBDb.TO+JNJ+MSFT
    sym_str = '+'.join(sym_list)
    url_str += 's=' + sym_str

    # Yahoo Finance tags, refer to http://www.financialwisdomforum.org/gummy-stuff/Yahoo-data.htm
    tags = {'s':'Symbol', 'x':'Exchange', 'j1':'Market Cap', 'b4':'Book Value', 'r':'P/E', 'p5':'Price/Sales',
            'p6':'Price/Book', 'j4':'EBITDA', 'j':'52-week Low', 'k':'52-week High', 'l1':'Last Trade',
            'd':'Dividend/Share', 'y':'Dividend Yield', 'e':'EPS', 's7':'Short Ratio', 's1':'Shares Owned',
            'f6':'Float Shares'}
    url_str += '&f=' + ''.join(pd.compat.iterkeys(tags))
    with urlopen(url_str) as resp:
        raw = resp.read()
    lines = raw.decode('utf-8').strip().replace('"', '').split('\n')
    lines = [line.strip().split(',') for line in lines]
    stats = DataFrame(lines, columns=list(tags.values()))
    stats = stats.drop_duplicates()
    stats = stats.set_index('Symbol')
    return stats

class Symbol:
    """
    Class of a stock symbol.
    """
    def __init__(self, sym, name=None, start=DEFAULT_START_DATE, end=None, datapath='./data', loaddata=True):
        self.sym = sym # e.g. 'AAPL'
        self.exch = None # stock exchange symbol, e.g. NMS, NYQ
        self.quotes = DataFrame()
        self.stats = DataFrame()
        self.income = DataFrame()  # Income Statement
        self.balance = DataFrame() # Balance Sheet
        self.cashflow = DataFrame() # Cash Flow
        self.name = name
        if name != None:
            self.datapath = os.path.normpath(datapath+'/'+name)
        else:
            self.datapath = os.path.normpath(datapath+'/'+sym)
        self.files = {'quotes':self.datapath + '/quotes.csv',
                      'stats':self.datapath + '/stats.csv',
                      'income':self.datapath + '/income.csv',
                      'balance':self.datapath + '/balance.csv',
                      'cashflow':self.datapath + '/cashflow.csv'}
        [self.start_date, self.end_date] = parse_start_end_date(start, end)
        if loaddata:
            self.load_data(from_file=True)

    def _handle_start_end_dates(self, start, end):
        if start == None and end == None:
            return [self.start_date, self.end_date]
        else:
            return parse_start_end_date(start, end)

    def get_quotes(self, start=None, end=None, sym=None):
        """
        Download history quotes from Yahoo Finance.
        Return Pandas DataFrame in the format of
                             Open       High    Low      Close    Volume  Adj Close
            Date                                                                   
            2004-06-23  15.000000  17.299999  14.75  17.200001  43574400       4.30
            2004-06-24  17.549999  17.690001  16.50  16.760000   8887200       4.19
            2004-06-25  16.510000  16.750000  15.79  15.800000   6710000       3.95
            2004-06-28  16.000000  16.209999  15.44  16.000000   2270800       4.00
            2004-06-29  16.000000  16.700001  15.83  16.400000   2112000       4.10
        """
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        if sym == None:
            sym = self.sym
        try:
            self.quotes = web.DataReader(sym, "yahoo", start_date, end_date)
        except RemoteDataError:
            print('Error: failed to get quotes for '+sym+' from Yahoo Finance.')
        return self.quotes

    def get_financials(self, browser=None):
        """
        Download financial data from Google Finance.
        The financial data are stored in *reversed* time order from left to right.
        """
        # e.g. https://www.google.com/finance?q=NYSE%3ACRM&fstype=ii
        site='https://www.google.com/finance?q=' + get_exchange_by_sym(self.exch) + '%3A' + self.sym + '&fstype=ii'

        close_browser = False
        if browser == None:
            browser=webdriver.Chrome()
            close_browser = True

        # Income Statement
        browser.get(site)
        tables=browser.find_elements_by_id('fs-table')
        if len(tables) < 1:
            print('Error: %s: failed to find income statement.' %self.sym)
        else:
            self.income = _parse_google_financial_table(tables, 'Revenue')
        # TODO: combine new income with hist income

        # Balance Sheet
        link=browser.find_element_by_link_text('Balance Sheet')
        link.click()
        tables=browser.find_elements_by_id('fs-table')
        tables=browser.find_elements_by_id('fs-table')
        if len(tables) < 1:
            print('Error: %s: failed to find balance sheet.' %self.sym)
        else:
            self.balance = _parse_google_financial_table(tables, 'Total Assets')
        # TODO: combine new balance with hist income

        # Cash Flow
        link=browser.find_element_by_link_text('Cash Flow')
        link.click()
        tables=browser.find_elements_by_id('fs-table')
        if len(tables) < 1:
            print('Error: %s: failed to find cash flow.' %self.sym)
        else:
            self.cashflow = _parse_google_financial_table(tables, 'Amortization')
        # TODO: combine new cash flow with hist income

        if close_browser:
            browser.close()

    def get_nasdaq_report(self):
        """
        NASDAQ stock report: http://www.nasdaq.com/symbol/nvda
        """
        link = 'http://stockreports.nasdaq.edgar-online.com/<sym>.html'
        # TODO: download the data


    def load_data(self, from_file=True):
        """
        Get stock data from file or web.
        """
        if from_file:
            if os.path.isfile(self.files['quotes']):
                self.quotes = pd.read_csv(self.files['quotes'])
                self.quotes = self.quotes.set_index('Date')

            if os.path.isfile(self.files['stats']):
                self.stats = pd.read_csv(self.files['stats'])
                self.stats = self.stats.set_index('Symbol')

            if os.path.isfile(self.files['income']):
                self.income = pd.read_csv(self.files['income'])
                self.income = self.income.set_index('Entries')

            if os.path.isfile(self.files['balance']):
                self.balance = pd.read_csv(self.files['balance'])
                self.balance = self.balance.set_index('Entries')

            if os.path.isfile(self.files['cashflow']):
                self.cashflow = pd.read_csv(self.files['cashflow'])
                self.cashflow = self.cashflow.set_index('Entries')
        else:
            self.get_quotes()
            self.get_stats()
            #self.get_financials()

    def save_data(self):
        """
        Save stock data into files.
        """
        if not os.path.isdir(self.datapath):
            os.makedirs(self.datapath)
        if len(self.quotes) > 0:
            self.quotes.to_csv(self.files['quotes'])
        if len(self.stats) > 0:
            self.stats.to_csv(self.files['stats']) 
        if len(self.income) > 0:
            self.income.to_csv(self.files['income'])
        if len(self.balance) > 0:
            self.balance.to_csv(self.files['balance'])
        if len(self.cashflow) > 0:
            self.cashflow.to_csv(self.files['cashflow'])

    def return_on_investment(self, start=None, end=None, exclude_dividend=False):
        """
        Calculate stock Return On Investiment(ROI, or Rate Of Return) for a given period.
            Total Stock Return = ((P1 - P0) + D) / P0
        where
            P0 = Initial Stock Price
            P1 = Ending Stock Price
            D  = Dividends
        """
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        adj_close = self.quotes.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d'),'Adj Close']
        no_dividend = (str(self.stats['Dividend Yield'][self.sym]).upper() == 'N/A')
        if exclude_dividend or no_dividend:
            dividend = 0
        else:
            # For simplicity, suppose the dividend yield is calculated as
            #   Dividend Yield = (Annual Dividends Per Share) / (Avg Price Per Share)
            # This is not accurate and need to be enhanced.
            dividend = float(self.stats['Dividend Yield'][self.sym]) * adj_close.mean() / 100
        roi = (adj_close[len(adj_close)-1] - adj_close[0] + dividend) / adj_close[0]
        return roi

    def get_additional_stats(self, exclude_dividend=False):
        """
        Additional stats that calculated based on history price.
        """
        labels = ['Symbol', 'Last-Quarter Return', 'Half-Year Return', '1-Year Return', '2-Year Return', '3-Year Return', '5-Year Return', 'Price In 52-week Range']
        if self.quotes.empty:
            self.get_quotes()
        if self.quotes.empty:
            # Failed to get history quotes, insert position holders.
            st = np.zeros(len(labels)) - 99999999
            stats = DataFrame([st.tolist()], columns=labels)
            return stats

        [end_date, three_month_ago, half_year_ago, one_year_ago, two_year_ago, three_year_ago, five_year_ago] = get_stats_intervals(self.end_date)

        quarter_return = self.return_on_investment(three_month_ago, end_date, exclude_dividend)
        half_year_return = self.return_on_investment(half_year_ago, end_date, exclude_dividend)
        one_year_return = self.return_on_investment(one_year_ago, end_date, exclude_dividend)
        two_year_return = self.return_on_investment(two_year_ago, end_date, exclude_dividend)
        three_year_return = self.return_on_investment(three_year_ago, end_date, exclude_dividend)
        five_year_return = self.return_on_investment(five_year_ago, end_date, exclude_dividend)

        adj_close = self.quotes.loc[one_year_ago.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d'),'Adj Close']
        current = adj_close[len(adj_close) - 1]
        # Current price in 52-week range should between [0, 1] - larger number means more expensive.
        pos_in_range = (current - adj_close.min()) / (adj_close.max() - adj_close.min())

        st = [[self.sym, quarter_return, half_year_return, one_year_return, two_year_return, three_year_return, five_year_return, pos_in_range]]
        stats = DataFrame(st, columns=labels)
        stats = stats.drop_duplicates()
        stats = stats.set_index('Symbol')
        return stats

    def sma(self, n=20, start=None, end=None):
        """
        Calculate the Simple Moving Average.
        """
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        stock = self.quotes.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d'),:]
        move_avg = np.round(stock["Adj Close"].rolling(window = n, center = False).mean(), 2)
        return move_avg

    def sma_diff_to_index(self, index, n=10, start=None, end=None):
        """
        Calculate the acculated differences(simple moving average) of this symbol and the given index.
        index: Symbol of index(e.g. sp500)
        """
        if self.quotes.empty or index.quotes.empty:
            return -99999999
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        # use the latest available starting date
        latest = max(self.quotes.first_valid_index(), index.quotes.first_valid_index(), pd.Timestamp(start_date.ctime()))
        start_date = latest.to_pydatetime()
        move_avg_index = index.sma(n, start_date, end_date).dropna()
        move_avg_index /= move_avg_index[0]
        move_avg_symbol = self.sma(n, start_date, end_date).dropna()
        move_avg_symbol /= move_avg_symbol[0]
        diff = move_avg_symbol - move_avg_index
        return diff.sum()/len(diff) # normalization

    def sma_stats(self, index=None):
        """
        Calculate Simple Moving Average related stats.

        index: a Symbol class of index, e.g. S&P 500.
        """
        if index == None:
            index = Symbol('^GSPC', name='SP500') # S&P500
            index.get_quotes() # only quotes needed
        labels = ['Symbol', 'Half-Year SMA Diff '+index.name, '1-Year SMA Diff '+index.name, '2-Year SMA Diff '+index.name, '3-Year SMA Diff '+index.name, '5-Year SMA Diff '+index.name]
        [end_date, three_month_ago, half_year_ago, one_year_ago, two_year_ago, three_year_ago, five_year_ago] = get_stats_intervals(self.end_date)
        half_year_sma_diff = self.sma_diff_to_index(index, start=half_year_ago, end=end_date)
        one_year_sma_diff = self.sma_diff_to_index(index, start=one_year_ago, end=end_date)
        two_year_sma_diff = self.sma_diff_to_index(index, start=two_year_ago, end=end_date)
        three_year_sma_diff = self.sma_diff_to_index(index, start=three_year_ago, end=end_date)
        five_year_sma_diff = self.sma_diff_to_index(index, start=five_year_ago, end=end_date)
        stats = [[self.sym, half_year_sma_diff, one_year_sma_diff, two_year_sma_diff, three_year_sma_diff, five_year_sma_diff]]
        stats_df = DataFrame(stats, columns=labels)
        stats_df = stats_df.drop_duplicates()
        stats_df = stats_df.set_index('Symbol')
        return stats_df

    def mma(self, values, n=14):
        """
        Modified Moving Average(MMA)
        """
        #TODO
        # quotes.loc[pd.date_range(start=quotes.index[0], periods=5)].dropna().mean()
        return ((n-1)*values[0]+values[1])/n

    def rsi(self):
        """
        Relative Strenth Index(RSI)
        """
        #TODO
        return

    def get_stats(self, index=None):
        """
        Calculate all stats.
        """
        # Add symbol name
        self.stats = get_symbol_names([self.sym]) # reset stats

        # Yahoo Finance statistics - it must be downloaded before other stats
        basic_stats = get_symbol_yahoo_stats([self.sym])
        self.stats = self.stats.join(basic_stats)
        self.exch = self.stats['Exchange'][self.sym]

        # Add additional stats
        add_stats = self.get_additional_stats()
        self.stats = self.stats.join(add_stats)

        # SMA stats
        sma_stats = self.sma_stats(index)
        self.stats = self.stats.join(sma_stats)

        return self.stats

    def get_insider_trade(self):
        """
        NOT IMPLEMENTED YET
        """
        # links: http://insidertrading.org/
        # and http://openinsider.com/search?q=AMD
        # TODO: download insider trade history
        return
