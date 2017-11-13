from stock_analysis.utils import *

# conda install -c conda-forge selenium=3.0.1
from selenium import webdriver

def parse_google_financial_table(tables, keyword=None):
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
        return DataFrame()

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

class Symbol:
    """
    Class of a stock symbol.
    """
    def __init__(self, sym, name=None, start=DEFAULT_START_DATE, end=None, datapath='./data', loaddata=True):
        self.sym = sym # ticker symbol, e.g. 'AAPL'
        self.cik = str() # CIK
        self.exch = None # stock exchange symbol, e.g. NMS, NYQ
        self.quotes = DataFrame()
        self.stats = DataFrame()
        self.earnings = DataFrame()
        self.income = DataFrame()  # Income Statement
        self.balance = DataFrame() # Balance Sheet
        self.cashflow = DataFrame() # Cash Flow
        self.name = name
        if name != None:
            self.datapath = os.path.normpath(datapath+'/'+name)
        else:
            self.datapath = os.path.normpath(datapath+'/'+sym)
        self.edgarpath = os.path.normpath(self.datapath+'/edgar')
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

    def _adj_close(self):
        if 'Adj Close' in self.quotes.columns:
            return 'Adj Close'
        elif 'Close' in self.quotes.columns:
            return 'Close'
        else:
            return 'NA'

    def get_quotes(self, start=None, end=None):
        """
        Download history quotes from Yahoo or Google Finance.
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
        sym = self.sym
        for n_tries in range(0,5): # try at most 5 times
            try:
                self.quotes = web.DataReader(sym, 'yahoo', start_date, end_date)
                break
            except:
                print('Error: %s: failed to download historical quotes from Yahoo Finance, try Google Finance...' %sym)
                try:
                    self.quotes = web.DataReader(sym, 'google', start_date, end_date)
                    break
                except:
                    print('Error: %s: failed to download historical quotes from Google Finance.' %sym)

        if self.quotes.empty:
            if os.path.isfile(self.files['quotes']):
                print('%s: loading quotes from %s' %(sym, self.files['quotes']))
                self.quotes = self.quotes.from_csv(self.files['quotes']) # quotes manually downloaded
            else:
                print('!!!Error: %s: failed to download historical quotes!!!' %sym)
                return None

        # remove possible strings and convert to numbers
        if self.quotes[self._adj_close()].dtypes != np.dtype('float64'):
            m = self.quotes != 'null'
            self.quotes = self.quotes.where(m, np.nan).dropna(how='any').astype(float)
        self.start_date = to_date(self.quotes.first_valid_index()) # update start date
        return self.quotes

    def get_financials(self, exchange=None, browser=None):
        """
        Download financial data from Google Finance.
        The financial data are stored in *reversed* time order from left to right.
        """
        if exchange == None:
            if self.exch == None:
                if 'Exchange' in self.stats.columns:
                    self.exch = self.stats['Exchange'][self.sym]
                else:
                    # get exchange from Yahoo Finance
                    self.exch = get_symbol_exchange(self.sym)
            exchange = get_exchange_by_sym(self.exch)
        if exchange == None:
            exchange = "NASDAQ" # Final resort, just a guess

        # e.g. https://www.google.com/finance?q=NYSE%3ACRM&fstype=ii
        site='https://www.google.com/finance?q=' + exchange + '%3A' + self.sym + '&fstype=ii'

        close_browser = False
        if browser == None:
            browser=webdriver.Chrome()
            close_browser = True
        # set timeout
        browser.set_page_load_timeout(75)
        browser.set_script_timeout(60)

        # Income Statement
        try:
            browser.get(site)
        except:
            print("%s: Download financial failed, try again..." %self.sym)
            #time.sleep(1)
            try:
                browser.get(site)
            except:
                print("Error: %s: failed to get link: %s." %(self.sym, site))
                if close_browser:
                    browser.quit()
                return

        try:
            tables=browser.find_elements_by_id('fs-table')
        except:
            print('Error: timeout when finding \'fs-table\' for %s, exchange %s' %(self.sym, exchange))
            if close_browser:
                browser.quit()
            return

        if len(tables) < 1:
            # Make sure the current page is the Financial page
            try:
                link=browser.find_element_by_link_text('Financials')
                link.click()
                tables=browser.find_elements_by_id('fs-table')
            except:
                print("Error: Financials link not found for %s, exchange %s" %(self.sym, exchange))
                if close_browser:
                    browser.quit()
                return

        if len(tables) < 1:
            print('Error: %s: failed to find income statement, exchange %s.' %(self.sym, exchange))
            if close_browser:
                browser.quit()
            return
        else:
            self.income = parse_google_financial_table(tables, 'Revenue')

        # Balance Sheet
        link=browser.find_element_by_link_text('Balance Sheet')
        link.click()
        tables=browser.find_elements_by_id('fs-table')
        tables=browser.find_elements_by_id('fs-table')
        if len(tables) < 1:
            print('Error: %s: failed to find balance sheet.' %self.sym)
            if close_browser:
                browser.quit()
            return
        else:
            self.balance = parse_google_financial_table(tables, 'Total Assets')

        # Cash Flow
        link=browser.find_element_by_link_text('Cash Flow')
        link.click()
        tables=browser.find_elements_by_id('fs-table')
        if len(tables) < 1:
            print('Error: %s: failed to find cash flow.' %self.sym)
            if close_browser:
                browser.quit()
            return
        else:
            self.cashflow = parse_google_financial_table(tables, 'Amortization')

        if close_browser:
            browser.quit()
        return

    def download_earning(self, baseurl, fname, form='10-Q', force_update=False):
        """
        Download the specified earning report from given URL and save it to edgarpath/form/.
        """
        form_path = self.edgarpath+'/'+form
        if not os.path.isdir(form_path):
            os.makedirs(form_path)

        report = form_path+'/'+fname
        if force_update or not os.path.isfile(report):
            # download the data
            url = baseurl+'/'+fname
            try:
                r = requests.get(url)
            except requests.exceptions.RequestException as e:
                print('%s: request failure:' %self.sym)
                print(e)
                return
            if r.status_code != requests.codes.ok:
                print('%s: earning download failure, status %d, link %s' %(self.sym, r.status_code, url))
            # save the result anyway
            with open(report, 'w') as f:
                f.write(r.text)
            f.close()

    def load_earnings(self, form='10-Q'):
        """
        Load all earning reports from dir edgarpath/form/.
        """
        pass


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
        else:
            self.get_quotes()
            self.get_stats()
            #self.get_financials()
        self.load_financial_data(from_file)

    def load_financial_data(self, from_file=True):
        """
        Load financial data from file or web.
        """
        if from_file:
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
            self.get_financials()

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
        self.save_financial_data()

    def save_financial_data(self):
        """
        Save financial data.
        """
        if not os.path.isdir(self.datapath):
            os.makedirs(self.datapath)
        if len(self.income) > 0:
            self.income.to_csv(self.files['income'])
        if len(self.balance) > 0:
            self.balance.to_csv(self.files['balance'])
        if len(self.cashflow) > 0:
            self.cashflow.to_csv(self.files['cashflow'])

    def return_on_investment(self, start=None, end=None, exclude_dividend=True):
        """
        Calculate stock Return On Investiment(ROI, or Rate Of Return) for a given period.
            Total Stock Return = ((P1 - P0) + D) / P0
        where
            P0 = Initial Stock Price
            P1 = Ending Stock Price
            D  = Dividends
        """
        if self.quotes.empty:
            self.get_quotes()
        if self.quotes.empty:
            return 0
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        adj_close = self.quotes.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d'), self._adj_close()]
        if self.quotes.empty or len(adj_close) < 1:
            return -99999999
        no_dividend = ('DividendYield' not in self.stats.columns) or np.isnan(self.stats['DividendYield'][self.sym])
        if exclude_dividend or no_dividend:
            dividend = 0
        else:
            # For simplicity, suppose the dividend yield is calculated as
            #   Dividend Yield = (Annual Dividends Per Share) / (Avg Price Per Share)
            # This is not accurate and need to be enhanced.
            dividend = self.stats['DividendYield'][self.sym] * adj_close.mean() / 100 # yearly dividend
            dividend = dividend / 365 * (end_date-start_date).days # dividend in the range
        start_price = adj_close[0]
        end_price = adj_close[-1]
        if type(start_price) == str:
            start_price = str2num(start_price)
        if type(end_price) == str:
            end_price = str2num(end_price)
        if start_price == 0 or np.isnan(start_price):
            start_price = 0.00001
        if end_price == 0 or np.isnan(end_price):
            end_price == 0.00001
        roi = (end_price - start_price + dividend) / start_price
        return roi

    def return_periodic(self, periods=6, freq='365D'):
        """
        Calculate periodic average/median returns.

        periods and freq are parameters passed to Pandas date_range().
        """
        if self.quotes.empty:
            self.get_quotes()
        if self.quotes.empty:
            return [np.nan, np.nan]
        returns = []
        start_date = self.quotes.first_valid_index()
        end_date = self.quotes.last_valid_index()
        [start_date, end_date] = self._handle_start_end_dates(start_date, end_date)
        days = pd.date_range(end=end_date, periods=periods, freq=freq)[::-1] # The past (periods-1) periods in reverse order
        for i in range(1, len(days)):
            if days[i].date() < start_date:
                break # out of boundary
            #print('yearly: %s - %s' %(days[i].ctime(), days[i-1].ctime()))  # FIXME: TEST
            returns.append(self.return_on_investment(days[i], days[i-1], exclude_dividend=True))
        if len(returns) > 0:
            ret_avg = np.mean(returns)
            ret_median = np.median(returns)
        else:
            print('Error: %s: failed to calculate periodic(%s) returns.' %(self.sym, freq))
            ret_avg = np.nan
            ret_median = np.nan
        return [ret_avg, ret_median]

    def growth_stats(self, exclude_dividend=False):
        """
        Additional stats that calculated based on history price.
        """
        labels = ['Symbol', 'LastWeekReturn', 'LastMonthReturn', 'LastQuarterReturn', 'HalfYearReturn', '1YearReturn', '2YearReturn', '3YearReturn', 'AvgMonthlyReturn', 'MedianMonthlyReturn', 'AvgQuarterlyReturn', 'MedianQuarterlyReturn', 'AvgYearlyReturn', 'MedianYearlyReturn', 'PriceIn52weekRange', 'LastQuarterGrowth', 'LastYearGrowth']
        if self.quotes.empty:
            self.get_quotes()
        if self.quotes.empty:
            # Failed to get history quotes, insert position holders.
            st = np.zeros(len(labels)) - 99999999
            stats = DataFrame([st.tolist()], columns=labels)
            return stats

        [end_date, one_week_ago, one_month_ago, three_month_ago, half_year_ago, one_year_ago, two_year_ago, three_year_ago, five_year_ago] = get_stats_intervals(self.end_date)

        last_week_return = self.return_on_investment(one_week_ago, end_date, exclude_dividend)
        last_month_return = self.return_on_investment(one_month_ago, end_date, exclude_dividend)
        quarter_return = self.return_on_investment(three_month_ago, end_date, exclude_dividend)
        half_year_return = self.return_on_investment(half_year_ago, end_date, exclude_dividend)
        one_year_return = self.return_on_investment(one_year_ago, end_date, exclude_dividend)
        two_year_return = self.return_on_investment(two_year_ago, end_date, exclude_dividend)
        three_year_return = self.return_on_investment(three_year_ago, end_date, exclude_dividend)

        [yearly_ret_avg, yearly_ret_median] = self.return_periodic(periods=6, freq='365D') # yearly returns in the past 5 years
        [quart_ret_avg, quart_ret_median] = self.return_periodic(periods=13, freq='90D') # quarterly returns in the past 3 years
        [monthly_ret_avg, monthly_ret_median] = self.return_periodic(periods=25, freq='30D') #  monthly returns in the past 2 years

        adj_close = self.quotes.loc[one_year_ago.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d'), self._adj_close()].dropna(how='all')
        if not adj_close.empty and len(adj_close) > 0:
            current = adj_close[-1]
            last_year_growth = current - adj_close[0]
            # Current price in 52-week range should between [0, 1] - larger number means more expensive.
            pos_in_range = (current - adj_close.min()) / (adj_close.max() - adj_close.min())

            adj_close = self.quotes.loc[three_month_ago.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d'), self._adj_close()].dropna(how='all')
            if not adj_close.empty and len(adj_close) > 0:
                last_quarter_growth = adj_close[-1] - adj_close[0]
            else:
                last_quarter_growth = np.nan
        else:
            pos_in_range = np.nan
            last_year_growth = np.nan
            last_quarter_growth = np.nan

        st = [[self.sym, last_week_return, last_month_return, quarter_return, half_year_return, one_year_return, two_year_return,
               three_year_return, monthly_ret_avg, monthly_ret_median, quart_ret_avg, quart_ret_median, yearly_ret_avg, yearly_ret_median,
               pos_in_range, last_quarter_growth, last_year_growth]]
        stats = DataFrame(st, columns=labels)
        stats = stats.drop_duplicates()
        stats = stats.set_index('Symbol')
        return stats

    def sma(self, n=20, start=None, end=None):
        """
        Calculate the Simple Moving Average.
        Return - pandas Series.
        """
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        stock = self.quotes[self._adj_close()]
        move_avg = pd.Series(moving_average(stock, n, type='simple'), index=stock.index)
        return move_avg[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')].dropna(how='all')

    def ema(self, n=10, start=None, end=None):
        """
        Exponential Moving Average(EMA)
        Return - pandas Series.
        """
        if self.quotes.empty:
            self.get_quotes()
        if self.quotes.empty:
            return pd.Series()
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        # EMA is start date sensitive
        tmp_start = start_date - BDay(n) # The first n days are used for init, so go back for n business days
        stock = self.quotes[self._adj_close()][tmp_start.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]
        avg = pd.Series(moving_average(stock, n, type='exponential'), index=stock.index)
        return avg[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')].dropna(how='all')

    def diverge_to_index(self, index, n=10, start=None, end=None):
        """
        Calculate the diverge between this symbol and the given index.
        Exponential moving average is used for smoothing the prices.

        Inputs:
            index - Symbol of index(e.g. sp500)
            n - window passed to EMA
        Return - mean diverages
        """
        if self.quotes.empty:
            self.get_quotes()
        if index.quotes.empty:
            index.get_quotes()
        if self.quotes.empty or index.quotes.empty:
            return np.nan
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        # use the latest available starting date
        start_date = max(to_date(self.quotes.first_valid_index()), to_date(index.quotes.first_valid_index()), start_date)
        move_avg_index = index.ema(n, start_date, end_date).dropna(how='all')
        move_avg_symbol = self.ema(n, start_date, end_date).dropna(how='all')
        if move_avg_symbol.empty or move_avg_index.empty:
            return np.nan
        move_avg_index /= move_avg_index[0] # normalization
        move_avg_symbol /= move_avg_symbol[0] # normalization
        diff = move_avg_symbol - move_avg_index
        if diff.empty:
            return np.nan
        else:
            return diff.mean()

    def relative_growth(self, index, start=None, end=None):
        """
        Percentage of price change relative to the given index.
        """
        if self.quotes.empty:
            self.get_quotes()
        if index.quotes.empty:
            index.get_quotes()
        if self.quotes.empty or index.quotes.empty:
            return np.nan
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        # use the latest available starting date
        start_date = max(to_date(self.quotes.first_valid_index()), to_date(index.quotes.first_valid_index()), start_date)
        stock_quote = self.quotes[self._adj_close()][start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')].dropna(how='all')
        index_quote = index.quotes[self._adj_close()][start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')].dropna(how='all')
        if stock_quote.empty or index_quote.empty:
            return np.nan
        stock_growth = str2num(stock_quote[-1]) / str2num(stock_quote[0])
        index_growth = str2num(index_quote[-1]) / str2num(index_quote[0])
        return stock_growth / index_growth

    def _relative_average_periodic(self, index, start_date, end_date, periods, freq, cb, median=True):
        """
        Average of relative growth for given periods
        """
        growths = []
        days = pd.date_range(end=end_date, periods=periods, freq=freq)[::-1] # in reverse order
        for i in range(1, len(days)):
            if days[i].date() < start_date:
                break # out of boundary
            diff = cb(index, start=days[i], end=days[i-1])
            if np.isnan(diff):
                break
            else:
                growths.append(diff)
        if median:
            return np.median(growths)
        else:
            return np.mean(growths)

    def relative_growth_stats(self, index=None):
        """
        Calculate stats of relative growth to S&P 500.

        index: a Symbol class of index, e.g. S&P 500.
        """
        if index == None:
            index = Symbol('^GSPC', name='SP500', loaddata=False) # S&P500
        if self.quotes.empty:
            self.get_quotes()
        if index.quotes.empty:
            index.get_quotes()
        if self.quotes.empty or index.quotes.empty:
            return DataFrame()

        labels = ['Symbol', 'RelativeGrowthLastWeek', 'RelativeGrowthLastMonth', 'RelativeGrowthLastQuarter', 'RelativeGrowthHalfYear', 'RelativeGrowthLastYear', 'RelativeGrowthLast2Years', 'RelativeGrowthLast3Years', 'WeeklyRelativeGrowth', 'MonthlyRelativeGrowth', 'QuarterlyRelativeGrowth', 'YearlyRelativeGrowth']
        [end_date, one_week_ago, one_month_ago, three_month_ago, half_year_ago, one_year_ago, two_year_ago, three_year_ago, five_year_ago] = get_stats_intervals(self.end_date)
        relative_growth_one_week = self.relative_growth(index, start=one_week_ago, end=end_date)
        relative_growth_one_month = self.relative_growth(index, start=one_month_ago, end=end_date)
        relative_growth_last_quarter = self.relative_growth(index, start=three_month_ago, end=end_date)
        relative_growth_half_year = self.relative_growth(index, start=half_year_ago, end=end_date)
        relative_growth_last_year = self.relative_growth(index, start=one_year_ago, end=end_date)
        relative_growth_two_year = self.relative_growth(index, start=two_year_ago, end=end_date)
        relative_growth_three_year = self.relative_growth(index, start=three_year_ago, end=end_date)

        # periodic stats
        start_date = max(to_date(self.quotes.first_valid_index()), to_date(index.quotes.first_valid_index()) )
        weekly_rel_growth = self._relative_average_periodic(index, start_date, end_date, 13, '7D', self.relative_growth) # The past 3 months
        monthly_rel_growth = self._relative_average_periodic(index, start_date, end_date, 13, '30D', self.relative_growth) # The past 12 months
        quarterly_rel_growth = self._relative_average_periodic(index, start_date, end_date, 13, '90D', self.relative_growth) # The past 3 years
        yearly_rel_growth = self._relative_average_periodic(index, start_date, end_date, 7, '365D', self.relative_growth) # The past 6 years

        # 'QuarterlyDivergeIndex' and 'YearlyDivergeIndex'
        #quarterly_diverge = self._relative_average_periodic(index, start_date, end_date, 13, '90D', self.diverge_to_index) # The past 3 years
        #yearly_diverge = self._relative_average_periodic(index, start_date, end_date, 6, '365D', self.diverge_to_index) # The past 5 years

        stats = [[self.sym, relative_growth_one_week, relative_growth_one_month, relative_growth_last_quarter, relative_growth_half_year,
                  relative_growth_last_year, relative_growth_two_year, relative_growth_three_year, weekly_rel_growth, monthly_rel_growth,
                  quarterly_rel_growth, yearly_rel_growth]]
        stats_df = DataFrame(stats, columns=labels)
        stats_df = stats_df.drop_duplicates()
        stats_df = stats_df.set_index('Symbol')
        return stats_df

    def trend_stats(self):
        """
        Get all the technical details of trend.
        """
        if self.quotes.empty:
            self.get_quotes()
        if self.quotes.empty:
            print('Error: %s: history quotes are not available.' %self.sym)
            return DataFrame()
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=90)
        one_month_ago = end_date - dt.timedelta(days=30)
        labels = ['Symbol', 'ROC', 'ROC Trend 7D', 'ROC Trend 14D', 'RSI', 'MACD Diff', 'FSTO', 'SSTO', 'AvgFSTOLastMonth', 'AvgFSTOLastQuarter', 'VolumeChange']

        roc = self.roc(start=start_date, end=end_date)
        if roc.empty or len(roc) < 1:
            roc_stat = np.nan
        else:
            roc_stat = roc[-1]

        rsi = self.rsi(start=start_date, end=end_date)
        if rsi.empty or len(rsi) < 1:
            rsi_stat = np.nan
        else:
            rsi_stat = rsi[-1]

        [macd, signal, diff] = self.macd(start=start_date, end=end_date)
        if diff.empty or len(diff) < 1:
            macd_stat = np.nan
        else:
            macd_stat = diff[-1]

        [K,D] = self.stochastic(start=start_date, end=end_date)
        if K.empty or len(K) < 1:
            fsto_stat = np.nan
            avg_fsto_past_month = np.nan
            avg_fsto_past_quarter = np.nan
        else:
            fsto_stat = K[-1]
            avg_fsto_past_month = K[one_month_ago.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')].mean()
            avg_fsto_past_quarter = K.mean()
        if D.empty or len(D) < 1:
            ssto_stat = np.nan
        else:
            ssto_stat = D[-1]

        # ROC Trend
        seven_days_ago = end_date - dt.timedelta(days=7)
        forteen_days_ago = end_date - dt.timedelta(days=14)
        roc_trend_7d = find_trend(roc[seven_days_ago.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')])
        roc_trend_14d = find_trend(roc[forteen_days_ago.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')])

        # Volume changes in the past two weeks
        this_week_start = end_date - dt.timedelta(days=7)
        last_week_end = end_date - dt.timedelta(days=8)
        last_week_start = end_date - dt.timedelta(days=15)
        volume_this_week = self.quotes['Volume'].loc[this_week_start.strftime('%Y-%m-%d'):].mean()
        volume_last_week = self.quotes['Volume'].loc[last_week_start.strftime('%Y-%m-%d') : last_week_end.strftime('%Y-%m-%d')].mean()
        if volume_last_week != 0:
            volume_change = volume_this_week / volume_last_week
        else:
            volume_change = 999999

        stats = [[self.sym, roc_stat, roc_trend_7d, roc_trend_14d, rsi_stat, macd_stat, fsto_stat, ssto_stat, avg_fsto_past_month, avg_fsto_past_quarter, volume_change]]
        stats_df = DataFrame(stats, columns=labels)
        stats_df = stats_df.drop_duplicates()
        stats_df = stats_df.set_index('Symbol')
        return stats_df

    def financial_stats(self, exchange=None, browser=None, update=False):
        """
        Calculate financial stats.

        exchange: string of stock exchange, e.g. NASDAQ or NYSE
        browser:  selenium webdriver
        update: force to update financial data from web
        """
        if update: #FIXME: or self.income.empty or self.balance.empty or self.cashflow.empty:
            self.get_financials(exchange=exchange, browser=browser)
            self.save_financial_data()
        else:
            print('Loading financials under %s .' %self.datapath) # FIXME
            self.load_financial_data()

        labels = ['Symbol', 'RevenueMomentum', 'ProfitMargin', 'AvgProfitMargin', 'ProfitMarginMomentum', 'OperatingMargin', 'AvgOperatingMargin', 'OperatingMarginMomentum', 'AssetMomentum', 'Debt/Assets', 'Avg Debt/Assets', 'Debt/Assets Momentum', 'OperatingCashMomentum', 'InvestingCashMomentum', 'FinancingCashMomentum']

        net_income = pd.Series()
        operate_income = pd.Series()
        revenue = pd.Series()
        total_assets = pd.Series()
        total_debt = pd.Series()
        total_liabilities = pd.Series()
        total_liab_equity = pd.Series()
        total_equity = pd.Series()
        cash_change = pd.Series()
        cash_operating = pd.Series()
        cash_investing = pd.Series()
        cash_financing = pd.Series()

        if not self.income.empty:
            if '-' not in self.income.loc['Revenue'].tolist():
                revenue = financial_fmt(self.income.loc['Revenue'])
            else:
                revenue = financial_fmt(self.income.loc['Total Revenue'])
            net_income = financial_fmt(self.income.loc['Net Income'])
            operate_income = financial_fmt(self.income.loc['Operating Income'])
        if not self.balance.empty:
            total_assets = financial_fmt(self.balance.loc['Total Assets'])
            total_debt = financial_fmt(self.balance.loc['Total Debt'])
            total_liabilities = financial_fmt(self.balance.loc['Total Liabilities'])
            total_liab_equity = financial_fmt(self.balance.loc['Total Liabilities & Shareholders\' Equity'])
        if not self.cashflow.empty:
            cash_change = financial_fmt(self.cashflow.loc['Net Change in Cash'])
            cash_operating = financial_fmt(self.cashflow.loc['Cash from Operating Activities'])
            cash_investing = financial_fmt(self.cashflow.loc['Cash from Investing Activities'])
            cash_financing = financial_fmt(self.cashflow.loc['Cash from Financing Activities'])

        if len(revenue) > 0:
            revenue_momentum = find_trend(revenue, fit_poly=False)
            profit_margins = net_income / revenue
            profit_margin_moment = find_trend(profit_margins.dropna(how='all'), fit_poly=False)
            operating_margins = operate_income / revenue
            operate_margin_moment = find_trend(operating_margins.dropna(how='all'), fit_poly=False)
        else:
            revenue_momentum = 0
            profit_margins = np.zeros(4)
            profit_margin_moment = 0
            operating_margins = np.zeros(4)
            operate_margin_moment = 0

        if len(total_assets) > 0:
            asset_momentum = find_trend(total_assets.dropna(how='all'), fit_poly=False)
            debt_to_assets = total_debt / total_assets
            debt_assets_moment = find_trend(debt_to_assets.dropna(how='all'), fit_poly=False)
        else:
            asset_momentum = 0
            debt_to_assets = np.zeros(4)
            debt_assets_moment = 0

        cash_operate_moment = find_trend(cash_operating.dropna(how='all'), fit_poly=False)
        cash_invest_moment = find_trend(cash_investing.dropna(how='all'), fit_poly=False)
        cash_finance_moment = find_trend(cash_financing.dropna(how='all'), fit_poly=False)

        stats = [[self.sym, revenue_momentum, profit_margins[-1], profit_margins.mean(), profit_margin_moment, operating_margins[-1], operating_margins.mean(), operate_margin_moment, asset_momentum, debt_to_assets[-1], debt_to_assets.mean(), debt_assets_moment, cash_operate_moment, cash_invest_moment, cash_finance_moment]]
        stats_df = DataFrame(stats, columns=labels)
        stats_df = stats_df.drop_duplicates()
        stats_df = stats_df.set_index('Symbol')
        return stats_df

    def additional_stats(self):
        """
        Additional stats
        """
        if self.quotes.empty:
            self.get_quotes()
        if self.quotes.empty:
            return DataFrame()

        labels = ['Symbol', 'EPSGrowth', 'Forward P/E', 'EarningsYield', 'ReturnOnCapital', 'ReceivablesTurnover', 'InventoryTurnover', 'AssetUtilization', 'OperatingProfitMargin']
        eps_growth = (self.stats['EPSEstimateCurrentYear'][self.sym] - self.stats['EPS'][self.sym]) / self.stats['EPS'][self.sym] * 100 # percent
        # Forward P/E = (current price / EPS estimate next year)
        forward_pe = self.quotes[self._adj_close()][-1] / self.stats['EPSEstimateCurrentYear'][self.sym]

        # Earnings Yield = (EPS last year) / (Share Price)
        # Greenblatt's updated version:
        #   Earnings Yield = (EBIT + Depreciation – CapEx) / Enterprise Value,
        # where Enterprise Value = (Market Value + Debt – Cash)
        earnings_yield = 0
        if (not self.balance.empty) and (self.stats['MarketCap'][0] > 0) and (self.stats['EBITDA'][0] > 0):
            total_debt = financial_fmt(self.balance.loc['Total Debt'])[0] # in million
            ent_value = self.stats['MarketCap'][0]*1000 + total_debt # TODO: minus cash
            earnings_yield = self.stats['EBITDA'][0] / ent_value
        if earnings_yield == 0: # try a different definition
            earnings_yield = self.stats['EPS'][self.sym] / self.quotes[self._adj_close()][-1]

        # Return on capital:
        #   ROIC = (NetOperatingProfit - AdjustedTaxes) / InvestedCapital
        # or
        #   ROIC = (Net Income - Dividends) / TotalCapital
        # where,
        #   InvestedCapital = FixedAssets + IntangibleAssets + CurrentAssets - CurrentLiabilities - Cash

        # Receivables Turnover:
        #   Receivables Turnover = (12-month sales) / (4-quarter average receivables)
        # a high ratio indicates that the company efficiently collects its accounts receivables or has quality customers.

        # Inventory Turnover:
        #   Inventory Turnover = (12-month cost of goods sold (COGS)) / (a 4-quarter average inventory)
        # a high value of the ratio indicates a low level of inventory relative to COGS, while a low ratio
        # signals that the company has excess inventory.

        # Asset Utilization = (12-month sales) / (12-month average of total assets)
        # the higher the ratio, the greater is the chance that the company is utilizing its assets efficiently.

        # Operating profit margin = (12-month operating income) / (12-month sales)
        # indicates how well a company is controlling its operating expenses.

        roic = 0
        receivables_turnover = 0
        inventory_turnover = 0
        asset_utilization = 0
        operating_profit_margin = 0
        if not (self.income.empty or self.balance.empty or self.cashflow.empty):
            #net_income = financial_fmt(self.income.loc['Net Income'])
            operating_income = financial_fmt(self.income.loc['Operating Income']) # TODO: need a reliable way to cover negative operting income
            adjusted_tax = financial_fmt(self.income.loc['Income Before Tax']) - financial_fmt(self.income.loc['Income After Tax'])
            total_assets = financial_fmt(self.balance.loc['Total Assets'])
            total_liabilities = financial_fmt(self.balance.loc['Total Liabilities'])
            #cash_from_operating = financial_fmt(self.cashflow.loc['Cash from Operating Activities'])
            total_receivables = financial_fmt(self.balance.loc['Total Receivables, Net'])
            total_inventory = financial_fmt(self.balance.loc['Total Inventory'])
            if '-' not in self.income.loc['Total Revenue'].tolist():
                total_sales = financial_fmt(self.income.loc['Total Revenue'])
            else:
                total_sales = financial_fmt(self.income.loc['Revenue'])
            total_cost = financial_fmt(self.income.loc['Cost of Revenue, Total'])
            cash = 0 # TODO: calculate total cash
            l = min(len(operating_income), len(total_assets), len(total_liabilities))
            if l > 0:
                invested_capital = sum(total_assets[:l]) - sum(total_liabilities[:l]) - cash
                #roic = np.divide(net_income[:l], invested_capital) # TODO: minus dividends
                roic = np.divide((sum(operating_income[:l]) - sum(adjusted_tax[:l])), invested_capital)

            l = min(len(total_sales), len(total_receivables), 4)
            if l > 0:
                receivables_turnover = np.divide(sum(total_sales[:l]), sum(total_receivables[:l]))

            l = min(len(total_cost), len(total_inventory), 4)
            if l > 0:
                inventory_turnover = np.divide(sum(total_cost[:l]), sum(total_inventory[:l]))

            l = min(len(total_sales), len(total_assets), 4)
            if l > 0:
                asset_utilization = np.divide(sum(total_sales[:l]), sum(total_assets[:l]))

            l = min(len(operating_income), len(total_sales), 4)
            if l > 0:
                operating_profit_margin = np.divide(sum(operating_income[:l]), sum(total_sales[:l]))

        stat = [[self.sym, eps_growth, forward_pe, earnings_yield, roic, receivables_turnover, inventory_turnover, asset_utilization, operating_profit_margin]]
        stat = DataFrame(stat, columns=labels)
        stat.drop_duplicates(inplace=True)
        stat.set_index('Symbol', inplace=True)

        return stat

    def get_stats(self, index=None, exclude_name=False, exclude_dividend=False):
        """
        Calculate all stats.
        index: Symbol of index
        """
        if self.quotes.empty:
            self.get_quotes()
        if self.quotes.empty:
            print("%s: ERROR - cannot download quotes, no statistics available." %self.sym)
            return DataFrame()
        elif self.quotes is None:
            print("%s: ERROR - invalid quotes." %self.sym)
            return DataFrame()

        # make sure quotes are numbers
        if self.quotes[self._adj_close()].dtypes != np.dtype('float64'):
            m = (self.quotes != 'null')
            self.quotes = self.quotes.where(m, np.nan).dropna(how='any').astype(float)

        # Yahoo Finance statistics - it must be downloaded before other stats
        self.stats = get_symbol_yahoo_stats([self.sym], exclude_name=exclude_name)
        if 'Exchange' in self.stats.columns:
            self.exch = self.stats['Exchange'][self.sym]

        # stats of return based on history quotes
        growth_stats = self.growth_stats(exclude_dividend=exclude_dividend)
        self.stats = self.stats.join(growth_stats)

        # diverge to index stats
        relative_growth_stats = self.relative_growth_stats(index)
        self.stats = self.stats.join(relative_growth_stats)

        # trend & momentum
        trend_stats = self.trend_stats()
        self.stats = self.stats.join(trend_stats)

        # financial stats
        financial_stats = self.financial_stats(exchange=self.exch)
        self.stats = self.stats.join(financial_stats)

        # additional stats
        add_stats = self.additional_stats()
        self.stats = self.stats.join(add_stats)

        return self.stats.transpose() # transpose for the sake of display


    ### Momentum ###
    def momentum(self, n=2, start=None, end=None):
        """
        Momentum, defined as
            Momentum = Today's closing price - Closing price X days ago
        Return - pandas Series of price differences
        """
        if self.quotes.empty:
            self.get_quotes()
        if self.quotes.empty:
            return pd.Series()
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        stock = self.quotes[self._adj_close()] # calc momentum for all hist data
        calc = lambda x: x[-1] - x[0]
        m = stock.rolling(window = n, center = False).apply(calc).dropna()
        return m[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]

    def roc(self, n=10, start=None, end=None):
        """
        Rate of Change(ROC), defined as
            ROC = ((current value / previous value) - 1) x 100
        Return - pandas Series with dates as index.
        """
        if self.quotes.empty:
            self.get_quotes()
        if self.quotes.empty:
            return pd.Series()
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        stock = self.quotes[self._adj_close()]
        calc = lambda x: (x[-1]/x[0] - 1) * 100
        rates = stock.rolling(window = n, center = False).apply(calc).dropna()
        return rates[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]

    def macd(self, start=None, end=None):
        """
        Moving Average Convergence/Divergence(MACD)

        The MACD indicator (or "oscillator") is a collection of three time series calculated
        from historical price data: the MACD series proper, the "signal" or "average" series,
        and the "divergence" series which is the difference between the two. 
        
        The most commonly used values are 12, 26, and 9 days, that is, MACD(12,26,9):
            MACD Line = (12-period EMA – 26-period EMA)
            Signal Line = 9-period EMA
            Histogram = MACD Line – Signal Line

        Return: list of [MACD Line, Signal Line, Histogram], all in pandas Series format.
        """
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        rng = pd.date_range(start=start_date, end=end_date, freq='D')
        fastema = self.ema(n=12)
        slowema = self.ema(n=26)
        macdline = fastema-slowema
        macdline = macdline.dropna()
        signal = pd.Series(moving_average(macdline, n=9, type='exponential'), index=macdline.index)
        hist = macdline-signal
        return [macdline[rng].dropna(), signal[rng].dropna(), hist[rng].dropna()]

    def rsi(self, n=14, start=None, end=None):
        """
        Relative Strenth Index(RSI)

        Return a Pandas Series of RSI.

        The standard algorithm of calculating RSI is:
                          100
            RSI = 100 - --------
                         1 + RS
            RS = Average Gain / Average Loss

            The very first calculations for average gain and average loss are simple 14 period averages.
            
            First Average Gain = Sum of Gains over the past 14 periods / 14.
            First Average Loss = Sum of Losses over the past 14 periods / 14.

            The second, and subsequent, calculations are based on the prior averages and the current gain loss:
            
            Average Gain = [(previous Average Gain) x 13 + current Gain] / 14.
            Average Loss = [(previous Average Loss) x 13 + current Loss] / 14.
        """
        if self.quotes.empty:
            self.get_quotes()
        if self.quotes.empty:
            return pd.Series()

        # RSI is start date sensitive
        [start_date, end_date] = self._handle_start_end_dates(start, end)
        tmp_start = start_date - BDay(n) # The first n days are used for init, so go back for n business days
        prices = self.quotes[self._adj_close()][tmp_start.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]
        m = np.diff(prices)

        # initialization
        seed = m[:n+1] # cause the diff is 1 shorter
        up = seed[seed>=0].sum()/n
        down = -seed[seed<0].sum()/n # losses should be positive
        rsi = np.zeros_like(prices)
        rsi[:n] = 100. - 100./(1. + up/down)

        # subsequent calculations
        for i in np.arange(n, len(prices)):
            d = m[i-1]
            if d > 0:
                gain = d
                loss = 0
            else:
                gain = 0
                loss = -d  # losses should be positive
            up = (up*(n - 1) + gain)/n
            down = (down*(n - 1) + loss)/n
            rsi[i] = 100. - 100/(1. + up/down)

        rsi = pd.Series(rsi, index=prices.index) # price diff drops the fist date
        return rsi[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')].dropna()

    def stochastic(self, nK=14, nD=3, start=None, end=None):
        """
        Stochastic Oscillator

        Inputs:
            nK - window of fast stochastic oscillator %K
            nD - window of slow stochastic oscillator %D
        Return both fast and slow stochastic oscillators, as Pandas Series.

        They are calculated by:
            Stochastic Oscillator(%K) = (Close Price - Lowest Low) / (Highest High - Lowest Low) * 100
            Fast %D = 3-day SMA of %K
            Slow %D = 3-day SMA of fast %D
        where typical values for N are 5, 9, or 14 periods.
        """
        if self.quotes.empty:
            self.get_quotes()
        if self.quotes.empty:
            return [pd.Series(), pd.Series()]

        close = self.quotes[self._adj_close()]
        if len(close) <= nK:
            return [pd.Series(), pd.Series()]

        try:
            ratio = self.quotes[self._adj_close()] / self.quotes['Close']
        except ZeroDivisionError:
            ratio = 1
        high = self.quotes['High'] * ratio # adjusted high
        low = self.quotes['Low'] * ratio   # adjusted low

        sto = np.zeros_like(close)
        for i in np.arange(nK, len(close)+1):
            s = close[i-nK : i]
            h = high[i-nK : i]
            l = low[i-nK : i]
            if max(h) != min(l):
                denominator = max(h) - min(l)
            else:
                # avoid devide-by-zero error
                denominator = random.random() / 1000
            sto[i-1] = (s[-1]-min(l))/denominator * 100
        sto[:nK-1] = sto[nK-1]
        K = pd.Series(sto, index=close.index)
        D = pd.Series(moving_average(K, n=nD, type='simple'), index=K.index)

        [start_date, end_date] = self._handle_start_end_dates(start, end)
        rng = pd.date_range(start=start_date, end=end_date, freq='D')
        return [K[rng].dropna(), D[rng].dropna()]

    def plot(self, start=None, end=None):
        """
        Plot price changes and related indicators.
        """
        if start == None and end == None:
            # set default range to 180 days
            end_date = dt.datetime.today().date()
            start_date = end_date - dt.timedelta(days=180)
        else:
            [start_date, end_date] = self._handle_start_end_dates(start, end)
        if self.name != None:
            ticker = self.name
        else:
            ticker = self.sym

        fillcolor_gold = 'darkgoldenrod'
        fillcolor_red = 'lightsalmon'
        fillcolor_green = 'lightgreen'
        nrows = 6
        fig,ax=plt.subplots(nrows,1,sharex=True)

        # plot price, volume and EMA
        ema10 = self.ema(n=10, start=start_date, end=end_date)
        ema30 = self.ema(n=30, start=start_date, end=end_date)
        price = self.quotes[self._adj_close()][start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]

        ax_ema = plt.subplot(nrows, 1, (1,2))
        ax_ema.fill_between(np.asarray(price.index), price.min(), np.asarray(price), facecolor='lightskyblue', linewidth=0.0)
        ema10.plot(grid=True, label='EMA(10)', color='red')
        ema30.plot(grid=True, label='EMA(30)', color='darkgreen')
        plt.legend(fontsize='xx-small', loc='upper left')
        ax_ema.set_ylim(bottom=price.min()) # change bottom scale
        ax_ema.set_ylabel('Price')
        ax_ema.set_xticklabels([]) # hide x-axis labels

        # plot ROC
        window = 10
        roc = self.roc(n=window, start=start_date, end=end_date)

        ax_roc = plt.subplot(nrows, 1, 3)
        roc.plot(grid=True, label='ROC(%d)'%window)
        bottom, top = ax_roc.get_ylim()
        ax_roc.set_yticks(np.round(np.linspace(bottom, top, num=4), decimals=0)) # reduce y-axis ticks
        if top >= 0 and bottom <= 0:
            ax_roc.axhline(0, color=fillcolor_gold)
        plt.legend(fontsize='xx-small', loc='upper left')
        ax_roc.set_ylabel('ROC')
        ax_roc.set_xticklabels([]) # hide x-axis labels

        # plot RSI
        window = 14
        rsi = self.rsi(n=window, start=start_date, end=end_date)

        ax_rsi = plt.subplot(nrows, 1, 4)
        rsi.plot(grid=True, label='RSI(%d)'%window)
        ax_rsi.set_ylim(0, 100)
        bottom, top = ax_rsi.get_ylim()
        ax_rsi.set_yticks(np.round(np.linspace(bottom, top, num=4), decimals=0)) # reduce y-axis ticks
        ax_rsi.fill_between(np.asarray(rsi.index), 70, 100, facecolor=fillcolor_red, alpha=0.5, linewidth=0.0)
        ax_rsi.fill_between(np.asarray(rsi.index), 0, 30, facecolor=fillcolor_green, alpha=0.5, linewidth=0.0)
        plt.legend(fontsize='xx-small', loc='upper left')
        ax_rsi.set_ylabel('RSI')
        ax_rsi.set_xticklabels([]) # hide x-axis labels

        # plot MACD
        [macd, signal, hist] = self.macd(start=start_date, end=end_date)
        ax_macd = plt.subplot(nrows, 1, 5)
        ax_macd.bar(np.asarray(hist.index), np.asarray(hist), width=0.1, color=fillcolor_gold)
        ax_macd.fill_between(np.asarray(hist.index), np.asarray(hist), 0, facecolor=fillcolor_gold, edgecolor=fillcolor_gold)
        macd.plot(grid=True, label='MACD(12,26)', color='red')
        signal.plot(grid=True, label='EMA(9)', color='darkgreen')
        plt.legend(fontsize='xx-small', loc='upper left')
        bottom, top = ax_macd.get_ylim()
        if top >= 0 and bottom <= 0:
            ax_roc.axhline(0, color=fillcolor_gold)
        ax_macd.set_yticks(np.round(np.linspace(bottom, top, num=4), decimals=1)) # reduce y-axis ticks
        ax_macd.set_ylabel('MACD')
        ax_macd.set_xticklabels([]) # hide x-axis labels

        # plot Stochastic Oscilator
        n_k = 14
        n_d = 3
        K,D = self.stochastic(nK=n_k, nD=n_d, start=start_date, end=end_date)
        ax_sto = plt.subplot(nrows, 1, 6)
        K.plot(grid=True, label='%'+'K(%d)'%n_k, color='red')
        D.plot(grid=True, label='%'+'D(%d)'%n_d, color='darkgreen')
        bottom, top = ax_sto.get_ylim()
        ax_sto.set_yticks(np.round(np.linspace(bottom, top, num=4), decimals=0)) # reduce y-axis ticks
        ax_sto.fill_between(np.asarray(K.index), 80, 100, facecolor=fillcolor_red, alpha=0.5, linewidth=0.0)
        ax_sto.fill_between(np.asarray(K.index), 0, 20, facecolor=fillcolor_green, alpha=0.5, linewidth=0.0)
        plt.legend(fontsize='xx-small', loc='upper left')
        ax_sto.set_ylabel('FSTO')

        fig.suptitle(ticker)
        fig.autofmt_xdate()
        fig.show()
        return

    ### Insider Trade ###
    def get_insider_trade(self):
        """
        NOT IMPLEMENTED YET
        """
        # links: http://insidertrading.org/
        # and http://openinsider.com/search?q=AMD
        # TODO: download insider trade history
        return
