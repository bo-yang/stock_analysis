import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

from stock_analysis.symbol import *

class Index(object):
    """
    Base class of stock index.
    """
    def __init__(self, sym, name='Unknown', datapath='./data', loaddata=False):
        self.name = name        # e.g. SP500
        self.sym = Symbol(sym, name=name, datapath=datapath, loaddata=False) # the index ticker, e.g. '^GSPC'
        self.datapath = os.path.normpath(datapath + '/' + name)
        self.datafile = self.datapath + '/components.csv'
        self.components = DataFrame() # index 'Symbol'
        if loaddata:
            self.sym.get_quotes()
            self.load_data(from_file=True)

    def _string_to_float(self, data):
        if data == '-' or data.upper() == 'N/A' or data.upper() == 'NA':
            return data #float(-99999999.99)
        else:
            return float(data.replace(',','')) # remove ',' in numbers

    def get_compo_list(self):
        """
        Get all components in this index, stored as DataFrame, which should contain
        at least one column named 'Symbol'.
        """
        return self.components

    # Helper function for parallel-computing
    def _get_single_compo_stat(self, args):
        sym = args[0]
        quote = args[1].dropna() # DataFrame
        yahoo_stat = args[2]
        if quote.empty:
            return DataFrame()
        print('Processing ' + sym + ' ...') # FIXME: TEST ONLY
        stock = Symbol(sym, datapath=self.datapath, loaddata=False)
        stock.quotes = quote
        if not stock.quotes.empty:
            stock.stats = yahoo_stat
            stat = stock.get_additional_stats(exclude_dividend=True) # additional stats
            stat = stat.join(stock.diverge_stats(index=self.sym)) # add columns of SMA stats
            stat = stat.join(stock.trend_stats())
        else:
            print('Appending empty stats for ' + sym)
            stat = DataFrame()
        return stat

    def _get_compo_stats(self, pquotes, yahoo_stats):
        """
        pquotes: Pandas Panel of stocks' quotes from DataReader.
        """
        # calc additional stats
        add_stats = DataFrame()
        num_cores = mp.cpu_count()
        pool = ThreadPool(num_cores)
        args = [] # a list of 3-tuples
        for sym in pquotes.items:
            if sym in yahoo_stats.index:
                args.append( (sym, pquotes[sym], yahoo_stats.loc[sym].to_frame().transpose()) )
            else:
                args.append( (sym, pquotes[sym], DataFrame()) )
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
        tmp_stats = self.components[iStart:iEnd]
        sym_list = self.components[iStart:iEnd].index.tolist()
        pquotes = web.DataReader(sym_list, "yahoo", start_date, end_date)
        # items - symbols; major_axis - time; minor_axis - Open to Adj Close
        pquotes = pquotes.transpose(2,1,0)
        if len(pquotes.items) == 0:
            print('Error: failed to get history quotes for chunk  %d - %d.' %(iStart, iEnd))
            return DataFrame()
        print('# chunk symbols %d' %len(pquotes.items)) # FIXME: TEST ONLY
        yahoo_stats = get_symbol_yahoo_stats(pquotes.items.tolist())
        if yahoo_stats.empty:
            time.sleep(2)
            yahoo_stats = get_symbol_yahoo_stats(pquotes.items.tolist()) # try again
        if yahoo_stats.empty:
            print('Error: failed to download yahoo stats for chunk %d - %d.' %(iStart, iEnd))
            return DataFrame()
        add_stats = self._get_compo_stats(pquotes, yahoo_stats)
        tmp_stats = tmp_stats.join(add_stats)
        tmp_stats = tmp_stats.join(yahoo_stats)
        return tmp_stats

    def get_stats(self, save=True):
        """
        Calculate all components' statistics in batch.
        """
        [start_date, end_date] = parse_start_end_date(None, None)
        self.components = DataFrame() # reset data
        self.get_compo_list()
        if self.sym.quotes.empty:
            self.sym.get_quotes()

        chunk = 300 # len(SP500) == 505
        if len(self.components) <= chunk:
            args = (0,len(self.components))
            self.components = self._get_chunk_stats(args)
            return self.components

        # multiprocessing - process stocks chunk-by-chunk
        num_chunks = int(np.ceil(len(self.components)/chunk))
        num_procs = min(mp.cpu_count(), num_chunks)
        pool = mp.Pool(processes=num_procs)
        steps = np.round(np.linspace(0, len(self.components), num_chunks)).astype(int)
        args = [(steps[i-1], steps[i]) for i in range(1,len(steps))]
        stats = pool.map(self._get_chunk_stats, args)

        chunk_stats = DataFrame()
        for s in stats:
            chunk_stats = chunk_stats.append(s)
        self.components = chunk_stats

        if save and not self.components.empty:
            self.save_data()
        return self.components

    def get_sector(self, sector):
        """
        Filter out all components in the same sector.
        sector: str
        """
        if type(sector) != str:
            print('Error: sector must be string.')
            return None
        if self.components.empty:
            self.get_stats()
        m = self.components['Sector'] == sector
        symbols = self.components.where(m, np.nan)
        return symbols.dropna(axis=0, how='all') # drop all NaN rows

    def get_sector_medians(self, saveto=None):
        """
        Calculate industry medians for each column.
        """
        if self.components.empty:
            self.get_stats()
        medians = list()
        all_sectors = self.components['Sector'].drop_duplicates()
        for sec in all_sectors:
            symbols = self.get_sector(sec)
            if symbols.empty:
                continue
            row = list()
            idx = round(len(symbols)/2)
            for col in symbols.columns:
                if col == 'Name' or col == 'Industry':
                    row.append(symbols.sort_values('1-Year Return').ix[idx][col]) # this is a trick
                elif col == 'Sector':
                    row.append(sec)
                else:
                    row.append(symbols.sort_values(col).ix[idx][col])
            medians.append(row) # each sector
        medians_df = DataFrame(medians, columns=self.components.columns)
        medians_df = medians_df.set_index('Sector')
        if saveto != None and len(medians) > 0:
            medians_df.to_csv(saveto)
        return medians_df

    def find_top_components(self, columns, n=-1, saveto=None):
        """
        Find out the common top n components according to the given columns(str, list or dict).

        n: number of the top components for each columns
        columns: str, list or dict. By default all given columns will be sorted by descending order.
                 To specify different orders for different columns, a dict can be used.
        For example:
            columns = {'Price In 52-week Range':True, '1-Year Return':False, '1-Year Diverge SP500':False}
            columns={'1-Year Return':False, 'Median Quarterly Return':False, 'Price In 52-week Range':True}
            columns={'Avg Quarterly Return':False, 'Median Quarterly Return':False, 'Price In 52-week Range':True}
        where 'True' means ascending=True, and 'False' means ascending=False.
        """
        if n <= 0:
            n = int(len(self.components)/2)
        if n <= 0:
            print('Error: components empty, run get_stats() first.')
            return None

        if type(columns) == str or type(columns) == list:
            cols = convert_string_to_list(columns)
            orders = [False]*len(cols) # by default ascending = False
        elif type(columns) == dict:
            cols = list(columns.keys())
            orders = list(columns.values())
        else:
            print('Error: unsupported columns type.')
            return None
        if len(cols) == 0:
            return None
        elif len(cols) == 1:
            return self.components.sort_values(cols[0], ascending=orders[0])[:n] # slicing

        # multiple columns
        comp = self.components.sort_values(cols[0], ascending=orders[0])
        common = comp.index[:n]
        for col,order in zip(cols[1:], orders[1:]):
            comp = self.components.sort_values(col, ascending=order)
            common = common.append(comp.index[:n])
            common_list = common.get_duplicates()
            if len(common_list) < 1:
                return None # Nothing in common
            common = pd.indexes.base.Index(common_list)
        if saveto != None and len(common) > 0:
            self.components.loc[common].to_csv(saveto)
        return self.components.loc[common] # DataFrame of common stocks

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
        #self.sym.get_quotes(sym='^GSPC') # use SP500 as a reference

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

    def get_compo_list(self):
        if not os.path.isdir(self.datapath):
            os.makedirs(self.datapath)
        companylist = self.datapath + '/companylist.csv'

        # Exchange NASDAQ
        f = open(companylist,'wb')
        link = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download'
        response = urlopen(link)
        nasdaq=response.read()
        f.write(nasdaq) # save csv file
        f.close()

        # Exchange NYSE
        f = open(companylist,'ab')
        link = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download'
        response = urlopen(link)
        nyse=response.read()
        f.write(nyse) # save csv file
        f.close()

        # Exchange AMC
        f = open(companylist,'ab')
        link = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=AMC&render=download'
        response = urlopen(link)
        amc=response.read()
        f.write(amc) # save csv file
        f.close()

        self.components = self.components.from_csv(companylist)

        # delete columns
        self.components.drop('LastSale', axis=1, inplace=True)
        self.components.drop('MarketCap', axis=1, inplace=True)
        self.components.drop('ADR TSO', axis=1, inplace=True)
        self.components.drop('IPOyear', axis=1, inplace=True)
        self.components.dropna(axis=1, inplace=True)

        # remove unwanted chars from Symbol
        symbols = [''] * len(self.components)
        for i in range(0, len(self.components)):
            symbols[i] = self.components.index[i].strip()
        symbols = pd.Series(symbols, name='Symbol')
        self.components.reset_index(inplace=True)
        self.components.drop('Symbol', axis=1, inplace=True)
        self.components = self.components.join(symbols)
        self.components.set_index('Symbol', inplace=True)
        self.components = self.components.drop_duplicates()

        return self.components

