from stock_analysis.utils import *
from stock_analysis.symbol import *

class Index(object):
    """
    Base class of stock index.
    """
    def __init__(self, sym='^GSPC', name='Unknown', datapath='./data', components = DataFrame(), loaddata=False):
        self.name = name        # e.g. SP500
        self.sym = Symbol(sym, name=name, datapath=datapath, loaddata=False) # the index ticker, e.g. '^GSPC'
        self.datapath = os.path.normpath(datapath + '/' + name)
        self.datafile = self.datapath + '/components.csv'
        self.components = components # index 'Symbol'
        self.value_attribs = ['P/E', 'Price/Book', 'Price/Sales', 'Debt/Assets', 'ReturnOnCapital', 'ReceivablesTurnover', 'InventoryTurnover', 'AssetUtilization', 'OperatingProfitMargin', 'PriceIn52weekRange']
        if loaddata:
            self.sym.get_quotes()
            self.load_data(from_file=True)

    def get_compo_list(self):
        """
        Get all components in this index, stored as DataFrame, which should contain
        at least one column named 'Symbol'.
        """
        return self.components

    # Helper function for parallel-computing
    def _get_single_compo_stat(self, args):
        sym = args[0]
        quote = args[1].dropna(how='all') # DataFrame
        if quote.empty:
            return DataFrame()
        print('Processing ' + sym + ' ...') # FIXME: TEST ONLY
        stock = Symbol(sym, datapath=self.datapath+'/../', loaddata=False)
        stock.quotes = quote
        if not stock.quotes.empty:
            stock.get_stats(index=self.sym, exclude_name=True, exclude_dividend=True)
            stat = stock.stats
        else:
            print('Appending empty stats for ' + sym)
            stat = DataFrame()
        return stat

    def _get_compo_stats(self, pquotes):
        """
        pquotes: Pandas Panel of stocks' quotes from DataReader.
        """
        # calc additional stats
        add_stats = DataFrame()
        num_cores = mp.cpu_count()
        pool = ThreadPool(num_cores)
        args = [] # a list of 2-tuples
        for sym in pquotes.items:
            args.append( (sym, pquotes[sym]) )
        stats = pool.map(self._get_single_compo_stat, args)
        for s in stats:
            add_stats = add_stats.append(s)
        return add_stats

    def _get_chunk_stats(self, args):
        """
        Internal function to process stocks in batch.
        """
        iStart = args[0]
        iEnd = args[1]
        [start_date, end_date] = parse_start_end_date(None, None)
        print('Chunk %d - %d' %(iStart, iEnd)) # FIXME: TEST ONLY
        chunk_stats = self.components[iStart:iEnd]
        sym_list = self.components[iStart:iEnd].index.tolist()

        try:
            pquotes = web.DataReader(sym_list, 'yahoo', start_date, end_date)
        except:
            print('Chunk(%d - %d): failed to download history quotes from Yahoo Finance, try Google Finance now...' %(iStart, iEnd))
            try:
                pquotes = web.DataReader(sym_list, 'google', start_date, end_date)
            except:
                print('!!!Error: chunk(%d - %d): failed to download history quotes!!!' %(iStart, iEnd))
                return DataFrame()

        # items - symbols; major_axis - time; minor_axis - Open to Adj Close
        pquotes = pquotes.transpose(2,1,0)
        if len(pquotes.items) == 0:
            print('!!!Error: invalid history quotes for chunk %d - %d.' %(iStart, iEnd))
            return DataFrame()

        print('Total # of symbols in this chunk: %d' %len(pquotes.items)) # FIXME: TEST ONLY
        add_stats = self._get_compo_stats(pquotes)
        chunk_stats = chunk_stats.join(add_stats)
        return chunk_stats

    def get_stats(self, save=True, chunk=100):
        """
        Calculate all components' statistics in batch.
        """
        [start_date, end_date] = parse_start_end_date(None, None)
        self.components = DataFrame() # reset data
        self.get_compo_list()
        if self.sym.quotes.empty:
            self.sym.get_quotes()

        if len(self.components) <= chunk:
            args = (0,len(self.components))
            self.components = self._get_chunk_stats(args)
            return self.components

        # multiprocessing - process stocks chunk-by-chunk
        num_chunks = int(np.ceil(len(self.components)/chunk))
        num_procs = min(mp.cpu_count(), num_chunks)
        pool = mp.Pool(processes=num_procs)
        steps = np.round(np.linspace(0, len(self.components), num_chunks)).astype(int)
        args = [(steps[i-1], steps[i]-1) for i in range(1,len(steps))]
        stats = pool.map(self._get_chunk_stats, args)

        chunk_stats = DataFrame()
        for s in stats:
            chunk_stats = chunk_stats.append(s)
        self.components = chunk_stats

        # Replace inf by NaN
        self.components.replace([np.inf, -np.inf], np.nan, inplace=True)

        if save and not self.components.empty:
            self.save_data()
        return self.components

    def _get_financials_by_chunk(self, args):
        """
        Download financial data for specified chunk.
        args: a tuple of (istart,iend)
        """
        (istart, iend) = args
        comp_index = self.components.index
        # download financials
        browser=webdriver.Chrome()
        for sym in comp_index[istart:iend]:
            print('Chunk %s-%s: downloading financial data for %s' %(comp_index[istart], comp_index[iend], sym))
            stock = Symbol(sym)
            if 'Exchange' in self.components.columns:
                exch = self.components['Exchange'][sym]
                if type(exch) == pd.Series:
                    # unexpected duplicates, e.g. AMOV
                    exch = exch.iloc[0]
                if type(exch) == str:
                    stock.exch = exch
            stock.get_financials(browser=browser)
            stock.save_financial_data()
        browser.close()
        return

    def get_financials(self, update_list=True, sym_start=str(), sym_end=str(), num_procs=10):
        """
        Download financial data for stocks.

        update_list: True for update index component list, otherwise False.
        sym_start: the first stock symbol in this index
        sym_end: the last stock symbol in the index
        num_procs: number of processes for financial data download in parallel
        """
        if self.components.empty or update_list:
            self.get_compo_list(update_list=True)
        # slice symbols
        comp_index = self.components.index
        istart = 0
        iend = len(comp_index)
        if len(sym_start) > 0 and sym_start in comp_index:
            istart = comp_index.get_loc(sym_start)
        if len(sym_end) > 0 and sym_end in comp_index:
            iend = comp_index.get_loc(sym_end)
        if istart > iend:
            (istart, iend) = (iend, istart) # make sure end is greater than start
        # download financials
        pool = mp.Pool(processes=num_procs)
        steps = np.round(np.linspace(istart, iend, num_procs+1)).astype(int)
        args = [(steps[i-1], steps[i]-1) for i in range(1,len(steps))]
        stats = pool.map(self._get_financials_by_chunk, args)
        return

    def _get_tickers_of_sector_industry(self, secind, key):
        """
        Filter out all components in the same sector/industry.
        secind: 'Sector' or 'Industry'
        key: keyword, sector/industry name
        """
        if type(key) != str:
            print('Error: _get_sector_industry: keyword must be string.')
            return None
        if self.components.empty:
            self.get_stats()
        m = self.components[secind] == key
        symbols = self.components.where(m, np.nan)
        return symbols.dropna(axis=0, how='all') # drop all NaN rows

    def get_tickers_of_sector(self, sector):
        """
        Filter out all components in the same sector.
        sector: sector name, string
        """
        return self._get_tickers_of_sector_industry('Sector', sector)

    def get_tickers_of_industry(self, industry):
        """
        Filter out all components in the same industry.
        industry: industry name, string
        """
        return self._get_tickers_of_sector_industry('Industry', industry)

    def _get_sector_industry_top(self, secind=str(), siname='', columns=[], percent=0.2):
        """
        Get the top performers of each sector/industry.

        secind: 'Sector' or 'Industry'
        siname: sector/industry name
        columns: a list of attributes, e.g. ['P/E', 'Price/Book', 'Price/Sales']
        percent: the top percentage, [0-1]. E.g. percent=0.2 means top 20%
        """
        if self.components.empty:
            self.get_stats()

        if len(columns) == 0:
            columns = self.value_attribs

        if len(secind) > 0 and len(siname) > 0:
            symbols = self._get_tickers_of_sector_industry(secind, siname)
        else:
            symbols = self.components
        if symbols.empty:
            return DataFrame()

        orig_symbols = symbols[columns]
        symbols = symbols[columns]
        # Columns with values the smaller the better
        descend_cols = ['P/E', 'Price/Book', 'Price/Sales', 'Debt/Assets', 'Debt/Assets Momentum', 'PEG', 'Forward P/E', 'Price/Book', 'PriceIn52weekRange']
        asccols = list(set(columns) - set(descend_cols))
        descols = list(set(columns) & set(descend_cols))

        # pre-processing data
        #
        # suppress noises
        symbols /= symbols.median()
        upper_limit = symbols.median() * 20
        m = (symbols <= upper_limit)
        symbols = symbols.where(m, np.nan).replace(np.nan, upper_limit)
        lower_limit = -np.abs(upper_limit)
        m = (symbols >= lower_limit)
        symbols = symbols.where(m, np.nan).replace(np.nan, lower_limit)
        # normalization
        col_max = symbols.max()
        col_min = symbols.min()
        symbols = (symbols - col_min) / (col_max - col_min)

        # calc scores for each symbol based on columns
        scores = symbols[asccols].sum(axis=1) + (1 - symbols[descols]).sum(axis=1)
        scores = np.exp(scores) / np.sum(np.exp(scores)) # softmax
        scores.name = 'Score'
        symbols = orig_symbols.join(scores)

        # find the top by score
        top_nums = int(len(symbols) * percent) + 1
        tops = symbols.sort_values('Score', ascending=False).iloc[:top_nums]
        return tops

    def get_industry_tops(self, industries=list(), stocks=list(), columns=list(), percent=0.2, saveto = None):
        """
        Get the top performers of each industry.

        industries: a list of industry names
        stocks: a list of tickers, which can be used to get their industries
        columns: a list of attributes, e.g. ['P/E', 'Price/Book', 'Price/Sales']
        percent: the top percentage, [0-1]. E.g. percent=0.2 means top 20%
        saveto: save to file
        """
        if self.components.empty:
            self.get_stats()

        if type(industries) == str:
            industries = [industries]
        if len(industries) == 0:
            if len(stocks) > 0:
                inds = set()
                for s in stocks:
                    inds.add(self.components['Industry'].loc[s])
                industries = list(inds)
            else:
                industries = self.components['Industry'].drop_duplicates()

        tops = DataFrame()
        for ind in industries:
            # append sec/ind median as comparison
            ind_median = self.get_industry_median(columns=columns, industry=ind)
            ind_median = ind_median.join(pd.Series([np.nan], name='Score'))
            tops = tops.append(ind_median)
            tops = tops.append(self._get_sector_industry_top('Industry', siname=ind, columns=columns, percent=percent))

        if saveto != None and not tops.empty:
            f = os.path.normpath(self.datapath + '/' + saveto)
            tops.to_csv(f)
        return tops

    def _get_sector_industry_mean(self, secind, key, item=str()):
        """
        Get mean value in the same sector/industry.
        secind: 'Sector' or 'Industry'
        item: a specific sector/industry name
        key: keyword, e.g. 'P/E', 'Price/Book', 'Price/Sales'
        """
        if type(key) == str:
            key = [key]
        tags = [secind] + key
        lines = []
        if len(item) > 0:
            stocks = self._get_tickers_of_sector_industry(secind, item)
            lines.append([item] + stocks[key].mean().tolist())
        else:
            # dump all items
            uniques = set(self.components[secind].tolist())
            for item in uniques:
                stocks = self._get_tickers_of_sector_industry(secind, item)
                if len(stocks) == 0:
                    continue
                lines.append([item] + stocks[key].mean().tolist())
        stats = DataFrame(lines, columns=tags)
        stats = stats.drop_duplicates()
        stats = stats.set_index(secind)
        return stats

    def _get_sector_industry_median(self, secind, key=list(), item=str()):
        """
        Get mean value in the same sector/industry.
        secind: 'Sector' or 'Industry'
        item: a specific sector/industry name
        key: keyword, e.g. 'P/E', 'Price/Book', 'Price/Sales'
        """
        if len(key) == 0:
            key = self.value_attribs
        tags = [secind] + key
        lines = []
        if len(item) > 0:
            stocks = self._get_tickers_of_sector_industry(secind, item)
            lines.append([item] + stocks[key].median().tolist())
        else:
            # dump all items
            uniques = set(self.components[secind].tolist())
            for item in uniques:
                stocks = self._get_tickers_of_sector_industry(secind, item)
                if len(stocks) == 0:
                    continue
                lines.append([item] + stocks[key].median().tolist())
        stats = DataFrame(lines, columns=tags)
        stats = stats.drop_duplicates()
        stats = stats.set_index(secind)
        return stats

    def get_sector_median(self, columns=list(), sector=str()):
        """
        Get mean value in the same sector.
        columns: (list of) keyword, e.g. ['P/E', 'Price/Book', 'Price/Sales']
        sector: a specific sector name
        e.g. nasdaq.get_sector_mean(['P/E', 'Price/Book', 'Price/Sales'])
        """
        return self._get_sector_industry_median('Sector', columns, sector)

    def get_industry_median(self, columns=list(), industry=str()):
        """
        Get mean value in the same industry.
        columns: (list of) keyword, e.g. ['P/E', 'Price/Book', 'Price/Sales']
        industry: a specific industry name

        e.g.
            nasdaq.get_industry_median(['P/E', 'Price/Book', 'Price/Sales'])
            value_attribs = ['P/E', 'Price/Book', 'Price/Sales', 'Debt/Assets', 'ReceivablesTurnover', 'InventoryTurnover', 'AssetUtilization', 'OperatingProfitMargin']
            nasdaq.get_industry_median(value_attribs, item='RETAIL: Building Materials')
        """
        return self._get_sector_industry_median('Industry', columns, industry)

    def _compare_to_sector_industry(self, stocks, secind, columns=[], how='median'):
        """
        Compare a list of stocks to their sectors/industries.

        stocks: a list of stock tickers
        secind: 'Sector' or 'Industry'
        columns: a list of attributes to be compared
        how: 'median', 'top'
        """
        if type(stocks) != list:
            print('Error: a list of tickers is expected.')
            return
        if len(columns) == 0:
            columns = self.value_attribs

        tags = ['Items'] + columns
        if how == 'top':
            tags += ['Score']

        sidict = dict() # 'Sector/Industry' : [[]]
        for s in stocks:
            if s not in self.components.index:
                print('Error: %s not found in %s' %(s.sym, self.name))
                continue
            stock = self.components.loc[s]
            sec_ind = stock.loc[secind]
            line = [s] + stock.loc[columns].tolist()
            if sec_ind not in sidict.keys():
                sidict[sec_ind] = [line] # list of lists
            else:
                sidict[sec_ind].append(line)

        # TODO: add compare to top
        lines = []
        for sec_ind in sidict:
            lines.append([sec_ind] + self._get_sector_industry_median(secind, columns, item=sec_ind).iloc[-1].tolist())
            for l in sidict[sec_ind]:
                lines.append(l)
        stats = DataFrame(lines, columns=tags)
        stats = stats.drop_duplicates()
        stats = stats.set_index(tags[0])
        print(stats.to_string())
        return

    def compare_to_sector(self, stocks, columns=[]):
        """
        Compare a list of stocks to their sectors.

        stocks: a list of stock tickers
        columns: a list of attributes to be compared
        """
        return self._compare_to_sector_industry(stocks, 'Sector', columns)

    def compare_to_industry(self, stocks, columns=[]):
        """
        Compare a list of stocks to their industries.

        stocks: a list of stock tickers
        columns: a list of attributes to be compared
        """
        return self._compare_to_sector_industry(stocks, 'Industry', columns)

    def compare_stocks(self, stocks, columns=None):
        """
        Compare stocks using given attributes.

        stocks: a list of stock tickers
        columns: a list of attributes to be compared

        For example,
            columns=sp500.components.columns[3:].tolist()
            columns=nasdaq.components.columns[4:].tolist()
        """
        if type(stocks) != list:
            print('Error: a list of tickers is expected.')
            return
        if columns == None:
            not_needed = ['Name', 'ADR TSO', 'Sector', 'Industry', 'Summary Quote']
            columns = self.components.columns.drop(not_needed, errors='ignore').tolist()
        if type(columns) != list:
            print('Error: a list of attribute is expected.')
            return
        ret = self.components.loc[stocks, columns].transpose()
        print(ret.to_string())
        return

    def filter_by_sort(self, columns, n=-1, saveto=None):
        """
        Find out the common top n components according to the given columns(str, list or dict).

        n: number of the top components for each columns
        columns: str, list or dict. By default all given columns will be sorted by descending order.
                 To specify different orders for different columns, a dict can be used.
        For example:
            cheap={'AvgQuarterlyReturn':False, 'LastQuarterReturn':True, 'PriceIn52weekRange':True}
            reliable={'MedianQuarterlyReturn':False, 'AvgQuarterlyReturn':False, 'MedianYearlyReturn':False, 'AvgYearlyReturn':False, 'YearlyRelativeGrowth':False}
            buy={'AvgQuarterlyReturn':False, 'MedianQuarterlyReturn':False, 'PriceIn52weekRange':True, 'AvgFSTOLastMonth':True}
        where 'True' means ascending=True, and 'False' means ascending=False.
        """
        components = self.components
        if n <= 0:
            n = int(len(components)/2)
        if n <= 0:
            print('Error: components empty, run get_stats() first.')
            return DataFrame()
    
        if type(columns) == str or type(columns) == list:
            cols = str2list(columns)
            orders = [False]*len(cols) # by default ascending = False
        elif type(columns) == dict:
            cols = list(columns.keys())
            orders = list(columns.values())
        else:
            print('Error: unsupported columns type.')
            return DataFrame()
        if len(cols) == 0:
            return DataFrame()
        elif len(cols) == 1:
            return components.sort_values(cols[0], ascending=orders[0])[:n] # slicing
    
        # multiple columns
        comp = components.sort_values(cols[0], ascending=orders[0])
        common = comp.index[:n]
        for col,order in zip(cols[1:], orders[1:]):
            comp = components.sort_values(col, ascending=order)
            common = common.append(comp.index[:n])
            common_list = common.get_duplicates()
            if len(common_list) < 1:
                return DataFrame() # Nothing in common
            common = pd.Index(common_list)

        components = components.loc[common]
        if saveto != None and len(components) > 0:
            f = os.path.normpath(self.datapath + '/' + saveto)
            components.to_csv(f)
        return components # DataFrame of common stocks

    def filter_by_compare(self, rules, saveto=None):
        """
        Filtering stocks by comparing columns.

        rules: list of three-tuple, where the tuples are like (attribute1, oper, attribute2 or expression).

        For example:
            rules = [('EPSEstimateCurrentYear', '>', 'EPS'), ('PriceIn52weekRange', '<=', 0.7)]
            rules = [('LastMonthReturn', '>', 0), ('LastMonthReturn', '>', 'LastQuarterReturn'), ('LastQuarterReturn', '>', 'HalfYearReturn'), ('RelativeGrowthLastMonth', '>', 'RelativeGrowthLastQuarter'), ('RelativeGrowthLastQuarter', '>','RelativeGrowthHalfYear'), ('LastQuarterReturn', '>=', ('HalfYearReturn', '*=', 1.2)), ('AvgQuarterlyReturn', '>', 0.03)]
            rules = [('LastMonthReturn', '>', 0), ('LastQuarterReturn', '>', 'HalfYearReturn'), ('RelativeGrowthLastQuarter', '>','RelativeGrowthHalfYear'), ('LastQuarterReturn', '>=', ('HalfYearReturn', '*=', 1.2)), ('AvgQuarterlyReturn', '>', 0.03)]
        """
        if type(rules) == tuple:
            rules = [rules]
    
        ops = { '>': operator.gt,
                '<': operator.lt,
                '>=': operator.ge,
                '<=': operator.le,
                '==': operator.eq,
                '!=': operator.ne,
                '+=': operator.iadd,
                '-=': operator.isub,
                '*=': operator.imul,
                '/=': operator.itruediv,
                '%=': operator.imod,
                '**=': operator.ipow,
                '//=': operator.ifloordiv }
        operate = lambda x,oper,y: ops[oper](x, y)
    
        stats = self.components.replace(np.nan, 0) # np.nan is not comparable
        for rule in rules:
            if len(rule) != 3:
                print('Error: invalid rule ' + str(rule))
                continue # not 3-tuple
            if len(stats) <= 0:
                break
            if (type(rule[0]) != str) or (rule[0] not in stats.columns):
                print('Error: invalid keyword ' + rule[0])
                continue
            if (type(rule[2]) == str) and (rule[2] in stats.columns):
                cut = stats[rule[2]]
            elif (type(rule[2]) == tuple) and (len(rule[2]) == 3):
                # an embeded 3-tuple rule
                r = rule[2]
                cut = operate(stats[r[0]], r[1], r[2])
            else:
                cut = rule[2]
            inp = stats[rule[0]]
            stats = stats.where(operate(inp, rule[1], cut), np.nan).dropna(axis=0, how='all')

        if saveto != None and len(stats) > 0:
            f = os.path.normpath(self.datapath + '/' + saveto)
            stats.to_csv(f)

        return stats

    def load_data(self, from_file=True):
        if from_file:
            self.sym.load_data()
            if os.path.isfile(self.datafile):
                self.components = pd.read_csv(self.datafile)
                self.components = self.components.set_index('Symbol')
        else:
            self.sym.get_quotes()
            self.get_stats()
        return self.components

    def save_data(self):
        if not os.path.isdir(self.datapath):
            os.makedirs(self.datapath)
        self.sym.save_data()
        if len(self.components) > 0:
            self.components.to_csv(self.datafile)
        return

def get_index_components_from_wiki(link, params):
    """
    Download S&P index components.

        link:   url to the wiki page.
        params: a dict of <label:column_idx> - keys are labels to be used in the DataFrame,
             and values are the indices of columns in table. The following keys are needed:
                ['Symbol', 'Name', 'Sector', 'Industry']
    
    Return a DataFrame of the index.
    """
    tags = ['Symbol', 'Name', 'Sector', 'Industry']
    page = urlopen(link)
    soup = BeautifulSoup(page, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    st = list()
    for row in table.find_all('tr'):
        col = row.find_all('td')
        if len(col) > 0 and len(col) >= max(params.values()):
            if col[params['Symbol']].string == None:
                continue
            symbol = str(col[params['Symbol']].string.strip())
            if col[params['Name']].string != None:
                name = str(col[params['Name']].string.strip())
            else:
                name = 'n/a'
            if col[params['Sector']].string != None:
                sector = str(col[params['Sector']].string.strip()).lower().replace(' ', '_')
            else:
                sector = 'n/a'
            if col[params['Industry']].string != None:
                sub_industry = str(col[params['Industry']].string.strip()).lower().replace(' ', '_')
            else:
                sub_industry = 'n/a'
            st.append([symbol, name, sector, sub_industry])
    components = DataFrame(st, columns=tags)
    components = components.drop_duplicates()
    components = components.set_index('Symbol')
    return components

class SP500(Index):
    """
    S&P 500 index
    """
    def __init__(self, datapath='./data', loaddata=False):
        super(self.__class__, self).__init__(sym='^GSPC', name='SP500', datapath=datapath, loaddata=loaddata)

    def get_compo_list(self):
        """
        Get S&P 500 components. Ported from http://www.thealgoengineer.com/2014/download_sp500_data/.
        S&P 500 table format:
        Ticker symbol	| Security | SEC filings | GICS Sector | GICS Sub Industry | Address of Headquarters | Date first added | CIK
        """
        link = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        params={'Symbol':0, 'Name':1, 'Sector':3, 'Industry':4}
        self.components = get_index_components_from_wiki(link, params)
        return self.components

class SP400(Index):
    """
    S&P 400 index
    """
    def __init__(self, datapath='./data', loaddata=False):
        super(self.__class__, self).__init__(sym='^GSPC', name='SP400', datapath=datapath, loaddata=loaddata)

    def get_compo_list(self):
        """
        S&P 500 table format:
        Ticker Symbol | Company | GICS Economic Sector | GICS Sub-Industry | SEC Filings
        """
        link = "http://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        params={'Symbol':0, 'Name':1, 'Sector':2, 'Industry':3}
        self.components = get_index_components_from_wiki(link, params)
        return self.components


class DJIA(Index):
    """
    Dow Jones Industrial Average
    """
    def __init__(self, datapath='./data', loaddata=False):
        super(self.__class__, self).__init__(sym='^DJI', name='DowJones', datapath=datapath, loaddata=loaddata)

    def get_compo_list(self):
        """
        Company | Exchange | Symbol | Industry | Date Added  | Notes
        """
        link = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
        params={'Symbol':2, 'Name':0, 'Sector':3, 'Industry':3}
        self.components = get_index_components_from_wiki(link, params)
        return self.components

class NASDAQ100(Index):
    """
    NASDAQ-100
    """
    def __init__(self, datapath='./data', loaddata=False):
        super(self.__class__, self).__init__(sym='^NDX', name='NASDAQ-100', datapath=datapath, loaddata=loaddata)

    def get_compo_list(self):
        # link: https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average
        # TODO:
        return

class NASDAQ(Index):
    """
    NASDAQ
    """
    def __init__(self, datapath='./data', loaddata=False):
        super(self.__class__, self).__init__(sym='^IXIC', name='NASDAQ', datapath=datapath, loaddata=loaddata)

    def get_compo_list(self, update_list=False):
        if not os.path.isdir(self.datapath):
            os.makedirs(self.datapath)
        companylist = self.datapath + '/companylist.csv'

        self.components = DataFrame() # Reset components
        if os.path.isfile(companylist) and not update_list:
            self.components = self.components.from_csv(companylist)
            return self.components

        for exch in ['NASDAQ', 'NYSE', 'AMEX']:
            # Download company list
            f = open(companylist,'wb')
            link = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=' + exch + '&render=download'
            response = urlopen(link)
            comp_list=response.read()
            f.write(comp_list) # save csv file
            f.close()

            # Insert exchange column
            compdf = DataFrame()
            compdf = compdf.from_csv(companylist)
            exch_list = [STR_TO_EXCH_SYM[exch]] * len(compdf)
            exch_col = pd.Series(exch_list, name='Exchange', index=compdf.index)
            compdf = compdf.join(exch_col)

            if self.components.empty:
                self.components = compdf
            else:
                self.components = self.components.append(compdf)

        # delete columns
        self.components.drop('LastSale', axis=1, inplace=True)
        self.components.drop('MarketCap', axis=1, inplace=True)
        self.components.drop('IPOyear', axis=1, inplace=True)
        self.components.dropna(axis=1, inplace=True)

        # remove unwanted chars from Symbol
        symbols = self.components.index.str.strip()
        symbols = pd.Series(symbols, name='Symbol')
        self.components.reset_index(inplace=True)
        self.components.drop('Symbol', axis=1, inplace=True)
        self.components = self.components.join(symbols)
        self.components = self.components.drop_duplicates(subset=['Symbol']) # drop duplicated symbols
        self.components.set_index('Symbol', inplace=True)

        # remove spaces from Sector and Industry
        sector = self.components['Sector'].str.strip()
        industry = self.components['Industry'].str.replace(' ','')
        self.components.drop('Sector', axis=1, inplace=True)
        self.components.drop('Industry', axis=1, inplace=True)
        self.components = self.components.join(sector)
        self.components = self.components.join(industry)

        self.components.sort_index(inplace=True) # sort symbols alphabetically

        self.components.to_csv(companylist)

        return self.components

class Russell2000(Index):
    """
    Russell 2000
    """
    def __init__(self, datapath='./data', loaddata=False):
        super(self.__class__, self).__init__(sym='^GSPC', name='Russell2000', datapath=datapath, loaddata=loaddata)

    def get_compo_list(self, update_list=False):
        if not os.path.isdir(self.datapath):
            os.makedirs(self.datapath)
        companylist = self.datapath + '/companylist.csv'

        # TODO: download/generate Russell 2000 tickers
        if os.path.isfile(companylist):
            self.components = self.components.from_csv(companylist)

        return self.components

