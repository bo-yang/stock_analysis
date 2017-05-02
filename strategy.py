from stock_analysis.utils import *
from stock_analysis.symbol import *
from stock_analysis.index import *

# True - the larger the better, False - the smaller the better
rank_tags_growth = {'MedianQuarterlyReturn':True, 'AvgQuarterlyReturn':True, 'PriceIn52weekRange':False}
rank_tags_reliable = {'MedianQuarterlyReturn':False, 'AvgQuarterlyReturn':False, 'MedianYearlyReturn':False, 'AvgYearlyReturn':False, 'YearlyDivergeIndex':False}
rank_tags_value = {'EarningsYield':True, 'ReturnOnCapital':True}
rank_tags_hybrid = {'EarningsYield':True, 'ReturnOnCapital':True, 'EPSGrowth':True, 'AvgQuarterlyReturn':True,'PriceIn52weekRange':False}
rank_tags_hybrid2 = {'MedianQuarterlyReturn':True, 'AvgQuarterlyReturn':True, 'RevenueMomentum':True, 'ProfitMarginMomentum':True, 'Debt/Assets Momentum':False, 'EPSGrowth':True, 'PEG':False, 'Forward P/E':False, 'Price/Book':False, 'PriceIn52weekRange':False, 'EarningsYield':True, 'ReturnOnCapital':True}

blacklist = ['WINS', 'ENIC', 'LEXEB', 'LAUR', 'BGCA', 'AEK', 'MBT', 'VIP', 'BMA', 'EOCC', 'SID', 'HNP', 'PDP', 'GGAL',
        'CPA', 'CEA', 'VALE', 'MFG', 'TKC', 'ZNH', 'GATX', 'AGNCP', 'BFR', 'KEP', 'YIN', 'GMLP', 'YRD', 'SHI', 'PAM',
        'OEC']

def value_analysis(index):
    """
    Value analysis for the given Index.
    """
    if not issubclass(type(index), Index):
        print('Error: only Index type is supported.')
        return DataFrame()

    if type(index) == NASDAQ:
        rule = [('EPS', '>', 0), ('MarketCap', '>', 1)]
    else:
        rule = [('EPS', '>', 0), ('MarketCap', '>', 0)]
    stocks_value = index.filter_by_compare(rule)

    # Append stocks that MarketCap not available
    rule = [('MarketCap', '==', 0), ('Volume', '>', 0), ('EPS', '>', 0)]
    stocks_value = stocks_value.append(index.filter_by_compare(rule))

    # Drop foreign companies
    if 'ADR TSO' in stocks_value.columns:
        m = stocks_value['ADR TSO'] == 'n/a'
        stocks_value = stocks_value.where(m, np.nan).dropna(how='all')
    stocks_value.drop(blacklist, inplace=True, errors='ignore')

    # Drop smaller financial/utitlity/transportation companies
    sectors = ['Finance', 'Public Utilities', 'Transportation', 'Energy']
    for sec in sectors:
        rule = [('Sector', '==', sec), ('MarketCap', '<', 8)]
        small_companies = index.filter_by_compare(rule)
        stocks_value.drop(small_companies.index, inplace=True, errors='ignore')

    # Drop small retailers and REITs
    industries = ['Clothing/Shoe/Accessory Stores', 'Department/Specialty Retail Stores', 'Real Estate Investment Trusts']
    for indust in industries:
        rule = [('Industry', '==', indust), ('MarketCap', '<', 5)]
        to_be_dropped = index.filter_by_compare(rule)
        stocks_value.drop(to_be_dropped.index, inplace=True, errors='ignore')

    # Ranking stocks
    stock_rank = ranking(stocks_value, tags=rank_tags_value, rank='sort')
    stock_rank = stock_rank.join(index.components.loc[stock_rank.index][['AvgQuarterlyReturn', 'PriceIn52weekRange', 'Sector', 'Industry']])
    saveto = 'data/%s_value.csv' %index.name
    stock_rank.to_csv(saveto)
    #print(stock_rank.to_string())
    return stock_rank

def efficiency_level(stocks, saveto=None):
    """
    Efficiency level measures a company’s capability to transform its available input to output - a company with a favorable efficiency level is expected to provide impressive returns. Four ratios are used to find efficient companies that have the potential to provide impressive returns.

    Receivables Turnover = (12-month sales) / (4-quarter average receivables), a high ratio indicates that the company efficiently collects its accounts receivables or has quality customers.

    Inventory Turnover = (12-month cost of goods sold (COGS)) / (a 4-quarter average inventory), a high value of the ratio indicates a low level of inventory relative to COGS, while a low ratio signals that the company has excess inventory.

    Asset Utilization = (12-month sales) / (last 4-quarter average of total assets), the higher the ratio, the greater is the chance that the company is utilizing its assets efficiently.

    Operating profit margin, which is simply operating income over the past 12 months divided by sales over the same period, indicates how well a company is controlling its operating expenses.

    Only consider those companies that have higher ratios than their respective industry averages.

    Example:
        nasdaq_value = value_analysis(nasdaq)
        stocks = efficiency_level(nasdaq.components.loc[nasdaq_value.index[:800]], saveto='data/stocks_value.csv')
        stocks = efficiency_level(sp500.components.loc[sp500_value.index[:100]], saveto='data/stocks_value.csv')
    """
    rule = {'ReceivablesTurnover':True, 'InventoryTurnover':True, 'AssetUtilization':True, 'OperatingProfitMargin':True}

    return ranking(stocks, tags=rule, rank='range', saveto=saveto)

def grow_and_value(index, ref_index):
    """
    Filter out fast growing stocks with value analysis.
    index: Index to be analyzed, e.g. sp500/sp400/NASDAQ
    ref_index: reference index for value analysis, can be the same as index
    """
    if not issubclass(type(index), Index) or not issubclass(type(ref_index), Index):
        print('Error: only Index type is supported.')
        return DataFrame()

    rules = [('AvgQuarterlyReturn', '>', 0.05), ('MedianQuarterlyReturn', '>', 0.03)]
    index_grow = filter_by_compare(index, rules)
    ref_value = value_analysis(ref_index)
    index_value_grow = ref_value.loc[index_grow.index][['Total', 'AvgQuarterlyReturn', 'PriceIn52weekRange']].dropna()

    return index_value_grow

def turnover_and_value(index):
    """
    Filter out stocks that bounce back or grow faster.
    """
    if not issubclass(type(index), Index):
        print('Error: only Index type is supported.')
        return DataFrame()
    rules = [('LastQuarterDivergeIndex', '>', 0), ('LastQuarterDivergeIndex', '>=', ('HalfYearDivergeIndex', '*=', 0.6)), ('LastQuarterDivergeIndex', '>=', ('1YearDivergeIndex', '*=', 0.4))]
    index_grow = filter_by_compare(index, rules)
    index_value = value_analysis(index)
    stocks = index_value.loc[index_grow.index][['Total', 'AvgQuarterlyReturn', 'PriceIn52weekRange']].dropna()
    return stocks

def ranking(stocks, tags=rank_tags_hybrid2, rank='range', saveto=None):
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
        table = ranking_by_range(symbols, tags)
    elif rank == 'sort':
        table = ranking_by_sort(symbols, tags)
    else:
        print('Error: unsupported ranking method %s' %rank)

    if saveto != None and len(table) > 0:
        table.to_csv(saveto)
    return table

def ranking_by_range(symbols, tags):
    """
    Make a table and compare stocks based on key factors.
    For each column, the stocks are given a score between [0 - 100], and the total scores are computed for each column.
    """
    table = DataFrame()
    for t in tags.keys():
        col_max = symbols[t].max()
        col_min = symbols[t].min()
        maxrange = col_max - col_min
        symbols[t].replace(np.nan, maxrange/2, inplace=True) # replace NaN by mean
        if tags[t]:
            # the larger the better
            col = np.round((symbols[t] - col_min) / maxrange * 100)
        else:
            # the smaller the better
            col = np.round((col_max - symbols[t]) / maxrange * 100)
        if table.empty:
            table = table.append(col).transpose()
        else:
            table = table.join(col)

    total = pd.Series(table.transpose().sum(), name='Total', index=table.index)
    table = table.join(total)
    table.sort_values('Total', ascending=False, inplace=True)
    return table

def ranking_by_sort(symbols, tags):
    """
    Ranking stocks by sorting. Total scores - the lower the better.
    """
    #roc = symbols['EarningsYield']
    #ey = symbols['ReturnOnCapital']
    table = DataFrame()
    rank = pd.Series()
    for t in tags.keys():
        col_max = symbols[t].max()
        col_min = symbols[t].min()
        maxrange = col_max - col_min
        col = symbols[t].replace(np.nan, maxrange/2) # replace NaN by mean
        col = col.sort_values(ascending=(not tags[t]))
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

    total = pd.Series(rank/len(tags), name='Total', index=table.index)
    table = table.join(total)
    table.sort_values('Total', ascending=True, inplace=True)
    return table

def filter_by_sort(stocks, columns, n=-1, saveto=None):
    """
    Find out the common top n components according to the given columns(str, list or dict).

    n: number of the top components for each columns
    columns: str, list or dict.
    
    By default all given columns will be sorted by *descending* order.
             To specify different orders for different columns, a dict can be used.

    For example:
        cheap={'AvgQuarterlyReturn':False, 'LastQuarterReturn':True, 'PriceIn52weekRange':True}
        reliable={'MedianQuarterlyReturn':False, 'AvgQuarterlyReturn':False, 'MedianYearlyReturn':False, 'AvgYearlyReturn':False, 'YearlyDivergeIndex':False}
        reliable=['HalfYearDivergeIndex', '1YearDivergeIndex','2YearDivergeIndex', '3YearDivergeIndex','YearlyDivergeIndex']
        buy={'AvgQuarterlyReturn':False, 'MedianQuarterlyReturn':False, 'PriceIn52weekRange':True, 'AvgFSTOLastMonth':True}
        value = {'EarningsYield':False, 'ReturnOnCapital':False, 'EPSGrowth':False}
        score = {'AvgQuarterlyReturn':False, 'Total': True}
    where 'True' means ascending=True, and 'False' means ascending=False.
    """
    if type(stocks) == pd.DataFrame:
        index = Index()
        index.components = stocks
    elif issubclass(type(stocks), Index):
        index = stocks
    else:
        print('Error: ranking: unsupported type %s' %type(stocks))
        return DataFrame()

    stocks = index.filter_by_sort(columns, n, saveto=None)

    if saveto != None and len(stocks) > 0:
        stocks.to_csv(saveto)
    return stocks # DataFrame of common stocks

def filter_by_compare(stocks, rules, saveto=None):
    """
    Filtering stocks by comparing columns.

    stocks: DataFrame or Index
    rules: list of three-tuple, where the tuples are like (attribute1, oper, attribute2 or expression).

    For example:
        rules = [('EPSEstimateCurrentYear', '>', 'EPS'), ('PriceIn52weekRange', '<=', 0.7)]
        rules = [('LastMonthReturn', '>', 0), ('LastMonthReturn', '>', 'LastQuarterReturn'), ('LastQuarterReturn', '>', 'HalfYearReturn'), ('RelativeGrowthLastMonth', '>', 'RelativeGrowthLastQuarter'), ('RelativeGrowthLastQuarter', '>','RelativeGrowthHalfYear'), ('LastQuarterReturn', '>=', ('HalfYearReturn', '*=', 1.2)), ('AvgQuarterlyReturn', '>', 0.03)]
        rules = [('LastMonthReturn', '>', 0), ('LastQuarterReturn', '>', 'HalfYearReturn'), ('RelativeGrowthLastQuarter', '>','RelativeGrowthHalfYear'), ('LastQuarterReturn', '>=', ('HalfYearReturn', '*=', 1.2)), ('AvgQuarterlyReturn', '>', 0.03)]
        rules = [('AvgQuarterlyReturn', '>', 0.03), ('MedianQuarterlyReturn', '>', 0.03), ('PriceIn52weekRange', '<=', 0.8)]
        rules = [('EPS', '>=', 0), ('MarketCap', '>', 2), ('AvgMonthlyReturn', '>', 0.007), ('AvgQuarterlyReturn', '>', 0.04), ('MedianQuarterlyReturn', '>', 0.04)]
        rules = [('EPS', '>=', 0), ('MarketCap', '>', 2)]
    """
    if type(stocks) == pd.DataFrame:
        index = Index()
        index.components = stocks
    elif issubclass(type(stocks), Index):
        index = stocks
    else:
        print('Error: ranking: unsupported type %s' %type(stocks))
        return DataFrame()

    stats = index.filter_by_compare(rules, saveto=None)

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
