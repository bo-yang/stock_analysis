# stock_analysis

This is a suite of basic stock analysis methods collected from Internet. Due to my limited understanding of stocks and financial analysis, there's no guarantee for the correctness of technical/fundamental analysis implementations.

This library is implemented based on `pandas` and `numpy`. And it also requires the following libraries:
- `pandas_datareader` for downloading history data from Yahoo Finance.
- `bs4` for `BeautifulSoup`
- `multiprocessing` for multiprocessing
- `yahoo_finance` for downloading stock statistics from YQL
- `selenium` to download financial data from Google Finance

The basic usage of this library is:

```python
from stock_analysis import *
sp500 = SP500()   # define an index
sp500.get_financials() # download financial data Google Finance, a bit slow
sp500.get_stats() # calculate key statistic features
sp500_value = value_analysis(sp500)  # do the value analysis
# Or ranking based on other attribtues
rank_tags_hybrid = {'EarningsYield':True, 'ReturnOnCapital':True, 'EPSGrowth':True, 'AvgQuarterlyReturn':True,'PriceIn52weekRange':False}
sp500_hybrid = ranking(sp500, tags=rank_tags_hybrid)
```
For other APIs, please refer to the code.

Any suggestions please send e-mail to bonny95@gmail.com.
