# stock_analysis

This is a suite of basic stock analysis methods collected from Internet. Due to my limited understanding of stocks and financial analysis, there's no guarantee for the correctness of technical/fundamental analysis implementations.

This library is implemented based on `pandas` and `numpy`. And it also requires the following libraries:
- `pandas_datareader` for downloading history data from Yahoo Finance.
- `bs4` for `BeautifulSoup`
- `multiprocessing` for multiprocessing
- `yahoo_finance` for downloading stock statistics from YQL
- `selenium` to download financial data from Google Finance

File organization:
- `symbol.py`: class for a stock symbol(equity).
- `index.py`: classes for a stock exchange index.
- `strategy.py`: colleciton of stock analysis strategies.
- `utils.py`: misc functions.

The basic usage of this library is:

```python
from stock_analysis import *
nasdaq = NASDAQ()   # define an index
nasdaq.get_financials() # download financial data from Google Finance, a bit slow
nasdaq.get_stats() # compute equity statistic features

# for value analysis
nasdaq_value1 = value_analysis(nasdaq)
nasdaq_value2 = value_ranking(nasdaq)
# for growth analysis
nasdaq_growth = fast_grow_stocks(nasdaq)
# combination of growth and value analysis
stocks = grow_and_value(nasdaq)

# Or ranking based on other attribtues
rank_tags_hybrid = {'EarningsYield':True, 'ReturnOnCapital':True, 'EPSGrowth':True, 'AvgQuarterlyReturn':True,'PriceIn52weekRange':False}
nasdaq_hybrid = ranking(nasdaq, tags=rank_tags_hybrid)
```
For addtional explanation of the code, please refer to [My First Taste of Computational Stock Analysis](http://www.bo-yang.net/2017/03/24/my-first-taste-of-stock-analysis).

Any questions/suggestions please send e-mail to bonny95@gmail.com.
