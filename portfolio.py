from stock_analysis.utils import *
from stock_analysis.symbol import *

class Portfolio(object):
    def __init__(self, symfile='portfolio/symbols.csv'):
        """ Expected symbol file format
            Symbol,Share
            CSCO,83
            $cash$,2139.14
            NOC,28.32
        The first column is the symbols, and the second column for shares you own.
        A special symbol '$cash$' for the total number of cash you are holding. And
        its price is always 1 by default.
        """
        if os.path.isfile(symfile):
            self.symbols = pd.read_csv(symfile)
            self.symbols.set_index('Symbol', inplace=True)
        else:
            raise Exception('File %s not exist.' %symfile)

    def show(self, sort_by='Percent', ascending=False, update=False):
        """ Show the summary of the portfolio.
        Output format:
            Symbol	Share	Price	Value	Percent
        where 'Value' is the total value of a symbol, and 'Percent' is the percentage
        in portfolio of this symbol.
        """
        if update or not sort_by in self.symbols.columns:
            prices = pd.Series(data=np.zeros(len(self.symbols)), index=self.symbols.index, name='Price')
            values = pd.Series(data=np.zeros(len(self.symbols)), index=self.symbols.index, name='Value')
            percent = pd.Series(data=np.zeros(len(self.symbols)), index=self.symbols.index, name='Percent')
            for s in self.symbols.index:
                if s.lower() != 'cash':
                    prices[s] = yf.YahooFinancials(s).get_current_price()
                    print('Price of %s is %f' % (s, prices[s]))
                else:
                    prices[s] = 1
                values[s] = prices[s] * self.symbols['Share'][s]
            percent = values / float(sum(values))
            self.symbols['Price'] = prices
            self.symbols['Value'] = values
            self.symbols['Percent'] = percent
        self.symbols.sort_values(sort_by, ascending=ascending, inplace=True)
        print(self.symbols)
        print('Total: $%.02f' % sum(values))
