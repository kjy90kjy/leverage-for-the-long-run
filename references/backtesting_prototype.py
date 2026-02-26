import multiprocessing as mp

import numpy as np
import pandas as pd
from backtesting import Backtest
from backtesting import Strategy
from backtesting.lib import crossover

mp.set_start_method("fork")
# csv = "QQQ_3x.csv"
csv = "^NDX_3x.csv"
# csv = "TQQQ.csv"
data = pd.read_csv(csv, index_col="Date", parse_dates=["Date"], date_format="%Y-%m-%d")
data = data.loc[data.index >= "2002-01-01"]


def sma(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values including current.
    """
    return pd.Series(values).rolling(n, closed="right").mean()


class SmaCross(Strategy):
    n1 = 1
    n2 = 200
    __sma1 = None
    __sma2 = None

    def init(self):
        self.__sma1 = self.I(sma, self.data.Close, self.n1)
        self.__sma2 = self.I(sma, self.data.Close, self.n2)

    def next(self):
        if crossover(self.__sma1, self.__sma2):
            self.buy()
        elif crossover(self.__sma2, self.__sma1):
            self.position.close()


bt = Backtest(data, SmaCross, cash=10_000_000, commission=0.002, trade_on_close=True)
stats = bt.run()
print(stats)
# bt.plot(plot_drawdown=True)

stats["_equity_curve"]["Equity"] = np.log(
    stats["_equity_curve"]["Equity"] / stats["_equity_curve"].iloc[0]["Equity"]
)
# bt.plot(results=stats, plot_drawdown=True, relative_equity=False)
# import sys
#
# sys.exit()
from backtesting.lib import plot_heatmaps

for maximize in [
    "# Trades",
    "Return (Ann.) [%]",
    "Return [%]",
    "Max. Drawdown [%]",
    "Avg. Drawdown [%]",
    "Worst Trade [%]",
]:
    stats, heatmap = bt.optimize(
        n1=range(1, 50, 1),
        n2=range(10, 300, 10),
        constraint=lambda p: p.n1 < p.n2,
        return_heatmap=True,
        # random_state=0,
        # max_tries=100,
        maximize=maximize,
    )
    plot_heatmaps(heatmap, agg="mean")
