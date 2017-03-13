from stock_analysis.utils import *
from stock_analysis.symbol import *
from stock_analysis.index import *

# True - the larger the better, False - the smaller the better
rank_tags_growth = {'MedianQuarterlyReturn':True, 'AvgQuarterlyReturn':True, 'RevenueMomentum':True, 'ProfitMarginMomentum':True, 'Debt/Assets Momentum':False, 'EPSGrowth':True, 'PEG':False, 'Forward P/E':False, 'Price/Book':False, 'PriceIn52weekRange':False, 'EarningsYield':True, 'ReturnOnCapital':True}
rank_tags_value = {'EarningsYield':True, 'ReturnOnCapital':True, 'EPSGrowth':True}

def ranking(stocks, tags=rank_tags_value, rank='range', saveto=None):
    """
    Make a table and compare stocks based on key factors.
    stocks: Pandas DataFrame of stock stats or Index
    rank:
        'range' - score stocks based on range of tag values
        'sort' - sort each stock and sum up the tag indices
    """
    if type(stocks) == pd.DataFrame:
        symbols = stocks
    elif issubclass(type(stocks), Index):
        symbols = stocks.components
    else:
        print('Error: ranking: unsupported type %s' %type(stocks))
        return DataFrame()

    table = DataFrame()
    if rank == 'range':
        table = ranking_by_range(symbols, tags, saveto=saveto)
    elif rank == 'sort':
        table = ranking_by_sort(symbols, tags, saveto=saveto)
    else:
        print('Error: unsupported ranking method %s' %rank)

    if saveto != None and len(table) > 0:
        table.to_csv(saveto)
    return table

def ranking_by_range(symbols, tags, saveto=None):
    """
    Make a table and compare stocks based on key factors.
    For each column, the stocks are given a score between [0 - 10], and the total scores are computed for each column.
    """
    table = DataFrame()
    for t in tags.keys():
        maxrange = symbols[t].max() - symbols[t].min()
        symbols[t].replace(np.nan, maxrange/2, inplace=True) # replace NaN with mean
        # TODO: suppress noise
        if tags[t]:
            # the larger the better
            col = np.round((symbols[t] - symbols[t].min()) / maxrange * 10)
        else:
            # the smaller the better
            col = np.round((symbols[t].max() - symbols[t]) / maxrange * 10)
        if table.empty:
            table = table.append(col).transpose()
        else:
            table = table.join(col)

    total = pd.Series(table.transpose().sum(), name='Total', index=table.index)
    table = table.join(total)
    table.sort_values('Total', ascending=False, inplace=True)

    return table

def ranking_by_sort(symbols, tags, saveto=None):
    """
    Ranking stocks by sorting.
    """
    #roc = symbols['EarningsYield']
    #ey = symbols['ReturnOnCapital']
    table = DataFrame()
    rank = pd.Series()
    for t in tags.keys():
        col = symbols[t].replace(np.nan, 0) # replace NaN with mean
        col = col.sort_values(ascending=tags[t])
        rank_col = pd.Series(np.arange(len(col)), name=t+'Score', index=col.index)
        rank_col = rank_col.sort_index()
        if table.empty:
            table = table.append(col.sort_index()).transpose()
            table = table.join(rank_col)
            rank = rank_col
        else:
            table = table.join(col.sort_index())
            table = table.join(rank_col)
            rank = rank.sort_index() + rank_col

    total = pd.Series(rank, name='Total', index=table.index)
    table = table.join(total)
    table.sort_values('Total', ascending=False, inplace=True)

    return table

def filter_by_sort(stocks, columns, n=-1, saveto=None):
    """
    Find out the common top n components according to the given columns(str, list or dict).

    n: number of the top components for each columns
    columns: str, list or dict. By default all given columns will be sorted by descending order.
             To specify different orders for different columns, a dict can be used.
    For example:
        cheap={'AvgQuarterlyReturn':False, 'LastQuarterReturn':True, 'PriceIn52weekRange':True}
        reliable={'MedianQuarterlyReturn':False, 'AvgQuarterlyReturn':False, 'MedianYearlyReturn':False, 'AvgYearlyReturn':False, 'YearlyDivergeIndex':False}
        reliable=['HalfYearDivergeIndex', '1YearDivergeIndex','2YearDivergeIndex', '3YearDivergeIndex','YearlyDivergeIndex']
        buy={'AvgQuarterlyReturn':False, 'MedianQuarterlyReturn':False, 'PriceIn52weekRange':True, 'AvgFSTOLastMonth':True}
    where 'True' means ascending=True, and 'False' means ascending=False.
    """
    if type(stocks) == pd.DataFrame:
        components = stocks
    elif issubclass(type(stocks), Index):
        components = stocks.components
    else:
        print('Error: ranking: unsupported type %s' %type(stocks))
        return DataFrame()

    if n <= 0:
        n = int(len(components)/2)
    if n <= 0:
        print('Error: components empty, run get_stats() first.')
        return None

    if type(columns) == str or type(columns) == list:
        cols = str2list(columns)
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
        return components.sort_values(cols[0], ascending=orders[0])[:n] # slicing

    # multiple columns
    comp = components.sort_values(cols[0], ascending=orders[0])
    common = comp.index[:n]
    for col,order in zip(cols[1:], orders[1:]):
        comp = components.sort_values(col, ascending=order)
        common = common.append(comp.index[:n])
        common_list = common.get_duplicates()
        if len(common_list) < 1:
            return None # Nothing in common
        common = pd.Index(common_list)

    if saveto != None and len(common) > 0:
        if issubclass(type(stocks), Index):
            f = os.path.normpath(stocks.datapath + '/' + saveto)
        else:
            f = saveto
        components.loc[common].to_csv(f)
    return components.loc[common] # DataFrame of common stocks

def filter_by_compare(stocks, rules, saveto=None):
    """
    Filtering stocks by comparing columns.

    rules: list of three-tuple, where the tuples are like (attribute1, oper, attribute2 or expression).

    For example:
        rules = [('EPSEstimateCurrentYear', '>', 'EPS'), ('PriceIn52weekRange', '<=', 0.7)]
        rules = [('LastMonthReturn', '>', 0), ('LastMonthReturn', '>', 'LastQuarterReturn'), ('LastQuarterReturn', '>', 'HalfYearReturn'), ('RelativeGrowthLastMonth', '>', 'RelativeGrowthLastQuarter'), ('RelativeGrowthLastQuarter', '>','RelativeGrowthHalfYear'), ('LastQuarterReturn', '>=', ('HalfYearReturn', '*=', 1.2)), ('AvgQuarterlyReturn', '>', 0.03)]
        rules = [('LastMonthReturn', '>', 0), ('LastQuarterReturn', '>', 'HalfYearReturn'), ('RelativeGrowthLastQuarter', '>','RelativeGrowthHalfYear'), ('LastQuarterReturn', '>=', ('HalfYearReturn', '*=', 1.2)), ('AvgQuarterlyReturn', '>', 0.03)]
        rules = [('AvgQuarterlyReturn', '>', 0.04), ('MedianQuarterlyReturn', '>', 0.04), ('PriceIn52weekRange', '<=', 0.8)]
        rules = [('EPS', '>=', 0), ('MarketCap', '>', 2), ('AvgMonthlyReturn', '>', 0.007), ('AvgQuarterlyReturn', '>', 0.04), ('MedianQuarterlyReturn', '>', 0.04)]
    """
    if type(stocks) == pd.DataFrame:
        components = stocks
    elif issubclass(type(stocks), Index):
        components = stocks.components
    else:
        print('Error: ranking: unsupported type %s' %type(stocks))
        return DataFrame()

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

    stats = components
    for rule in rules:
        if len(rule) != 3:
            continue # not 3-tuple
        if len(stats) <= 0:
            break
        if (type(rule[0]) != str) or (rule[0] not in stats.columns):
            continue
        if (type(rule[2]) == str) and (rule[2] in stats.columns):
            cut = stats[rule[2]]
        elif (type(rule[2]) == tuple) and (len(rule[2]) == 3):
            r = rule[2]
            cut = operate(stats[r[0]], r[1], r[2])
        else:
            cut = rule[2]
        inp = stats[rule[0]]
        stats = stats.where(operate(inp, rule[1], cut), np.nan).dropna(axis=0, how='all')

    if saveto != None and len(stats) > 0:
        if issubclass(type(stocks), Index):
            f = os.path.normpath(stocks.datapath + '/' + saveto)
        else:
            f = saveto
        stats.to_csv(f)

    return stats

def strategy_growth():
    """
    Calculate stats of stock growth.

    Winning Strategy:
    1. EPS growth estimate for the current year is greater than or equal to last year's actual growth.
    2. Percentage change in last year EPS should be greater than or equal to zero
    Above two criteria point to flat earning or a growth trend over the years.

    3. Percentage change in price over 4 weeks greater than the perecntage change in price over 12 weeks.
    4. Percentage change in price over 12 weeks greater than percentage change in price over 24 weeks.
    Above two criteria show that price of the stock is increasing consistently over the said timeframes.

    5. Percentage price change for 4 weeks relative to the S&P 500 greater than the percentage price change for 12 weeks relative to the S&P 500.
    6. Percentage price change for 12 weeks relative to the S&P 500 greater than the percentage price change for 24 weeks relatiev to the S&P 500.
    Above two criteria make sure the price gains gets even stronger as it compares to the index.

    7. Percentage price change for 12 weeks is 20% higher than or equal to the percentage price change for 24 weeks, but it should not exceed 100%. A 20% increase in the price of a stock from the breakout point gives cues of an impending uptrend. But a jump over 100% indicates that there is limited scope for further upside and that the stock might be due for a reversal.

    Average 20-day Volume greater than or equal to 50,000 - high trading volume implies that the stocks have adequate liquidity.
    """
    rules = [('EPSEstimateCurrentYear', '>', 'LastYearGrowth'), ('LastMonthReturn', '>', 'LastQuarterReturn'), ('LastQuarterReturn', '>', 'HalfYearReturn'), ('RelativeGrowthLastMonth', '>', 'RelativeGrowthLastQuarter'), ('RelativeGrowthLastQuarter', '>','RelativeGrowthHalfYear'), ('LastQuarterReturn', '>=', ('HalfYearReturn', '*=', 1.2)), ('AvgQuarterlyReturn', '>', 0.03)]
    #TODO:
    return

def price_table(stocks):
    """
    Calc discounted price table.
    """
    if type(stocks) != list:
        print('Error: ranking: unsupported type %s' %type(stocks))
        return DataFrame()
    [start_date, end_date] = parse_start_end_date(None, None)
    pquotes = web.DataReader(stocks, "yahoo", start_date, end_date)
    prices = pquotes['Adj Close'].iloc[-1]
    table = DataFrame(prices.values, index=prices.index, columns=['Price'])
    lables = ['95%', '90%', '85%', '80%', '75%', '70%', '65%', '60%', '50%', '40%']
    for percent in lables:
        col = pd.Series(prices * str2num(percent), name=percent)
        table = table.join(col)
    return table
