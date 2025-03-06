# strategy.py
import logging
import numpy as np
import pandas as pd
import backtrader as bt
from typing import Dict, List, Tuple
import config
from data_utils import is_last_business_day_of_month, is_first_business_day_of_month

class PortfolioDataObserver(bt.Observer):
    """
    포트폴리오 가치, 날짜 및 수익률을 추적하는 커스텀 관측자 클래스
    """
    lines = ('portfolio_value',)
    plotinfo = dict(plot=True, subplot=True, plotname='Portfolio Value')

    # IMPROVEMENT: 중복기록 방지를 위해 마지막 관측된 날짜를 저장
    def __init__(self):
        self.last_observed_date = None

    def next(self):
        current_date = self._owner.data.datetime.datetime().date()
        if self.last_observed_date == current_date:
            return  # 같은 날짜에 이미 기록함

        self.last_observed_date = current_date
        # 원본 코드 유지: 첫 번째 데이터(feed)에 대해서만 기록
        if self._owner.datas.index(self._owner.data) == 0:
            self.lines.portfolio_value[0] = self._owner.broker.getvalue()

class MonthlyRebalanceStrategy(bt.Strategy):
    params = (
        ('touchstone', None),
        ('ma_periods', None),
        ('wma_periods', None),
        ('ma_type', None),
        ('asset_allocation', None),
        ('aggressive_allocation', None),
        ('moderate_allocation', None),
        ('defensive_allocation', None),
        ('mid_defensive_allocation', None),
        ('riskfreerate', None),
        ('use_market_regime', None),
        ('fractional_shares', None),
        ('cash_buffer_percent', None),
    )

    def __init__(self):
        self.logger = logging.getLogger('backtest.strategy')
        self._load_configuration()

        # 기준 티커
        self.touch = self.getdatabyname(self.p.touchstone)

        # 일간 MA 인디케이터 (daily)
        self.mas = {
            period: bt.indicators.SMA(self.touch.close, period=period, plotname=f"MA_{period}")
            for period in self.p.ma_periods
        }

        # 주간 모드에서 쓸 WMA를 위해 lines 참조 (FinanceDataReaderData에서 'wma_10', 'wma_30'이 이미 정의됨)
        # self.touch.wma_10, self.touch.wma_30 로 각 Bar에 해당하는 값 확인 가능
        # => _determine_allocation_weekly()에서 사용

        self._market_regime = None
        self._last_date = None
        self._current_month = None
        self._current_year = None
        self._prev_month_close_prices = {}
        self._assets = list(self.p.asset_allocation.keys())
        
        # 관측자 추가 (중복기록 방지 로직 포함)
        self.pf_observer = PortfolioDataObserver()
        
        self._validate_data()
        self.logger.info(f"Market regime usage: {'Enabled' if self.p.use_market_regime else 'Disabled'}")
        self.logger.info(f"Moving average type: {self.p.ma_type}")
        self.logger.info(f"Fractional shares: {'Allowed' if self.p.fractional_shares else 'Not allowed'}")

    def _load_configuration(self):
        if self.p.touchstone is None:
            self.p.touchstone = config.config.get("TOUCHSTONE")
        if self.p.ma_periods is None:
            self.p.ma_periods = config.config.get("MA_PERIODS")
        if self.p.wma_periods is None:
            self.p.wma_periods = config.config.get("WMA_PERIODS")
        if self.p.ma_type is None:
            self.p.ma_type = config.config.get("MA_TYPE")
        if self.p.asset_allocation is None:
            self.p.asset_allocation = config.config.get("ASSET_ALLOCATION")
        if self.p.aggressive_allocation is None:
            self.p.aggressive_allocation = config.config.get("AGGRESSIVE_ALLOCATION")
        if self.p.moderate_allocation is None:
            self.p.moderate_allocation = config.config.get("MODERATE_ALLOCATION")
        if self.p.defensive_allocation is None:
            self.p.defensive_allocation = config.config.get("DEFENSIVE_ALLOCATION")
        if self.p.mid_defensive_allocation is None:
            self.p.mid_defensive_allocation = config.config.get("MID_DEFENSIVE_ALLOCATION")
        if self.p.riskfreerate is None:
            self.p.riskfreerate = config.config.get("RISK_FREE_RATE")
        if self.p.use_market_regime is None:
            self.p.use_market_regime = config.config.get("USE_MARKET_REGIME")
        if self.p.fractional_shares is None:
            self.p.fractional_shares = config.config.get("FRACTIONAL_SHARES")
        if self.p.cash_buffer_percent is None:
            self.p.cash_buffer_percent = config.config.get("CASH_BUFFER_PERCENT")

    def _validate_data(self):
        for asset in self._assets:
            try:
                d = self.getdatabyname(asset)
                self.logger.info(f"Asset data loaded for: {asset}")
            except Exception as e:
                self.logger.error(f"Data missing for asset {asset}: {e}")
                raise ValueError(f"Data missing for asset: {asset}")

    def get_target_allocation(self, current_date) -> Dict[str, float]:
        if not self.p.use_market_regime:
            self.logger.info(f"[{current_date}] Market regime disabled; using base allocation.")
            self._market_regime = "Neutral"
            return self.p.asset_allocation
        
        current_price = self.touch.close[0]
        if self.p.ma_type == 'weekly':
            return self._determine_allocation_weekly(current_date, current_price)
        else:
            return self._determine_allocation_daily(current_date, current_price)
    
    def _determine_allocation_daily(self, current_date, current_price) -> Dict[str, float]:
        ma_values = {}
        for period in self.p.ma_periods:
            ma_values[period] = self.mas[period][0]

        self.logger.info(f"[{current_date}] {self.p.touchstone} Price: {current_price:.2f}, MA values: {ma_values}")
        ma_short = ma_values[self.p.ma_periods[0]]
        ma_mid = ma_values[self.p.ma_periods[1]]
        ma_mid2 = ma_values[self.p.ma_periods[2]]
        ma_long = ma_values[self.p.ma_periods[3]]
        
        if current_price > ma_short and ma_short > ma_mid and ma_mid > ma_long:
            self._market_regime = "Aggressive"
            allocation = self.p.aggressive_allocation
        elif current_price > ma_mid and ma_mid > ma_long:
            self._market_regime = "Moderate"
            allocation = self.p.moderate_allocation
        elif current_price < ma_mid2 and current_price > ma_long:
            self._market_regime = "MidDefensive"
            allocation = self.p.mid_defensive_allocation
        elif current_price < ma_long:
            self._market_regime = "Defensive"
            allocation = self.p.defensive_allocation
        else:
            self._market_regime = "Neutral"
            allocation = self.p.asset_allocation
            
        self.logger.info(f"Determined market regime: {self._market_regime}, Allocation: {allocation}")
        return allocation

    def _determine_allocation_weekly(self, current_date, current_price) -> Dict[str, float]:
        """
        IMPROVEMENT:
        기존에는 self.touch.p.dataname['WMA_XX'].iloc[0]로 첫 행만 사용했으나,
        FinanceDataReaderData의 lines(wma_10, wma_30)에 매 Bar가 대응되는 값이 들어옴.
        따라서 매번 self.touch.wma_10[0], self.touch.wma_30[0]을 사용.
        """
        try:
            wma_short = self.touch.wma_10[0]
            wma_long = self.touch.wma_30[0]
        except Exception as e:
            # 컬럼이 없거나 NaN일 수 있음
            self.logger.warning(f"[{current_date}] Weekly MA lines not found or invalid: {e}. Using base allocation.")
            self._market_regime = "Neutral"
            return self.p.asset_allocation

        self.logger.info(f"[{current_date}] {self.p.touchstone} Price: {current_price:.2f}, "
                         f"Weekly MA short={wma_short:.2f}, long={wma_long:.2f}")
        
        if current_price > wma_short and wma_short > wma_long:
            self._market_regime = "Aggressive"
            allocation = self.p.aggressive_allocation
        elif current_price < wma_short and current_price > wma_long:
            self._market_regime = "Moderate"
            allocation = self.p.moderate_allocation
        elif current_price < wma_long:
            self._market_regime = "Defensive"
            allocation = self.p.defensive_allocation
        else:
            self._market_regime = "Neutral"
            allocation = self.p.asset_allocation
            
        self.logger.info(f"Determined weekly market regime: {self._market_regime}, Allocation: {allocation}")
        return allocation

    def next(self):
        current_date = bt.num2date(self.touch.datetime[0]).date()
        current_value = self.broker.getvalue()
        
        if self._last_date is None:
            self._last_date = current_date
            self._current_month = current_date.month
            self._current_year = current_date.year
            self.logger.info(f"Start date: {current_date}, Initial value: {current_value:.2f}")
            return
            
        if is_last_business_day_of_month(current_date):
            self._record_month_end_prices(current_date)
            
        if current_date != self._last_date:
            if current_date.month != self._current_month or current_date.year != self._current_year:
                self._current_month = current_date.month
                self._current_year = current_date.year
                
            if is_first_business_day_of_month(current_date):
                try:
                    allocation = self.get_target_allocation(current_date)
                    self._rebalance_portfolio(current_date, allocation)
                except Exception as e:
                    self.logger.error(f"Rebalancing failed: {e}")
                    
            self._last_date = current_date
    
    def _record_month_end_prices(self, current_date):
        for asset in self._assets:
            try:
                d = self.getdatabyname(asset)
                self._prev_month_close_prices[asset] = d.close[0]
                self.logger.info(f"Recorded {asset} month-end close: {d.close[0]:.2f}")
            except Exception as e:
                self.logger.error(f"Failed to record close price for {asset}: {e}")

    def _rebalance_portfolio(self, current_date, target_allocation: Dict[str, float]):
        self.logger.info(f"=== {current_date} Rebalancing Start ===")
        total_value = self.broker.getvalue()
        cash_buffer = total_value * (self.p.cash_buffer_percent / 100.0)
        available_value = total_value - cash_buffer
        self.logger.info(f"Total value: {total_value:.2f}, Cash buffer: {cash_buffer:.2f}, Available: {available_value:.2f}")
        
        target_values = {}
        target_shares = {}
        current_shares = {}
        adjustments = {}
        
        for asset in self._assets:
            d = self.getdatabyname(asset)
            pos = self.getposition(d)
            current_shares[asset] = pos.size if pos else 0
            prev_close = self._prev_month_close_prices.get(asset)
            
            if prev_close is None:
                self.logger.warning(f"No previous month close for {asset}; skipping.")
                continue
                
            alloc_ratio = target_allocation.get(asset, 0)
            target_value = available_value * alloc_ratio
            target_values[asset] = target_value
            
            if self.p.fractional_shares:
                shares = target_value / prev_close
            else:
                shares = int(target_value / prev_close)
                
            target_shares[asset] = shares
            adjustments[asset] = shares - current_shares[asset]
            
            self.logger.info(f"{asset}: Target Ratio {alloc_ratio:.4f}, Target Value {target_value:.2f}, "
                             f"Target Shares {shares}, Current Shares {current_shares[asset]}, Adjustment {adjustments[asset]}")
        
        sell_orders = []
        for asset, adj in adjustments.items():
            if adj < 0:
                d = self.getdatabyname(asset)
                self.sell(data=d, size=abs(adj))
                sell_orders.append((asset, abs(adj)))
                self.logger.info(f"{current_date}: Sell {abs(adj)} shares of {asset}")
                
        current_cash = self.broker.getcash()
        estimated_cash = current_cash
        
        for asset, shares in sell_orders:
            prev_close = self._prev_month_close_prices.get(asset, 0)
            sell_value = shares * prev_close
            commission = sell_value * config.config.get("COMMISSION")
            estimated_cash += sell_value - commission
            
        self.logger.info(f"Current cash: {current_cash:.2f}, Estimated cash after sales: {estimated_cash:.2f}")
        
        buy_orders = []
        for asset, adj in adjustments.items():
            if adj > 0:
                d = self.getdatabyname(asset)
                prev_close = self._prev_month_close_prices.get(asset, 0)
                purchase_cost = adj * prev_close
                commission = purchase_cost * config.config.get("COMMISSION")
                required_cash = purchase_cost + commission
                
                if required_cash > estimated_cash:
                    max_shares = estimated_cash / (prev_close * (1 + config.config.get("COMMISSION")))
                    if not self.p.fractional_shares:
                        max_shares = int(max_shares)
                    if max_shares > 0:
                        self.logger.warning(f"Adjusting buy order for {asset}: {adj} -> {max_shares}")
                        adj = max_shares
                    else:
                        self.logger.error(f"Insufficient cash to buy {asset}.")
                        continue
                        
                if adj > 0:
                    self.buy(data=d, size=adj)
                    buy_cost = adj * prev_close
                    buy_commission = buy_cost * config.config.get("COMMISSION")
                    estimated_cash -= (buy_cost + buy_commission)
                    buy_orders.append((asset, adj))
                    self.logger.info(f"{current_date}: Buy {adj} shares of {asset}")
                    
        total_sell = sum(self._prev_month_close_prices.get(asset, 0) * shares for asset, shares in sell_orders)
        total_buy = sum(self._prev_month_close_prices.get(asset, 0) * shares for asset, shares in buy_orders)
        
        self.logger.info(f"Total Sell Value: {total_sell:.2f}, Total Buy Value: {total_buy:.2f}")
        self.logger.info(f"Rebalancing complete. Portfolio value: {total_value:.2f}, Estimated remaining cash: {estimated_cash:.2f}")
        self.logger.info(f"=== {current_date} Rebalancing End ===")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        dt = self.data.datetime.datetime()
        if order.status == order.Completed:
            order_type = 'BUY' if order.isbuy() else 'SELL'
            self.logger.info(f"Order Completed [{dt}]: {order_type} {order.executed.size:.4f} shares of "
                             f"{order.data._name} @ {order.executed.price:.2f}, "
                             f"Commission: {order.executed.comm:.2f}, Total: {order.executed.value:.2f}")
        elif order.status == order.Canceled:
            self.logger.warning(f"Order Canceled [{dt}]: {order.data._name}")
        elif order.status == order.Margin:
            self.logger.error(f"Margin Error [{dt}]: {order.data._name}")
        elif order.status == order.Rejected:
            self.logger.error(f"Order Rejected [{dt}]: {order.data._name}")
        else:
            self.logger.warning(f"Unknown order status [{dt}]: {order.Status[order.status]}")

    def notify_trade(self, trade):
        if trade.isclosed:
            dt = self.data.datetime.datetime()
            self.logger.info(f"Trade Closed [{dt}]: {trade.data._name}, PnL: {trade.pnl:.2f}, Net PnL: {trade.pnlcomm:.2f}")

    def get_portfolio_dates(self):
        datetime_array = self.touch.datetime.array
        value_array = self.pf_observer.lines.portfolio_value.array
        
        if len(value_array) == 2 * len(datetime_array):
            values = list(value_array[::2][-len(datetime_array):])
            if len(values) < len(datetime_array):
                min_length = min(len(datetime_array), len(values))
                datetime_array = datetime_array[-min_length:]
                values = values[-min_length:]
            return [bt.num2date(dt).date() for dt in datetime_array]
        elif len(value_array) > len(datetime_array):
            ratio = len(value_array) // len(datetime_array)
            if ratio > 1:
                self.logger.info(f"Observer array ratio: {ratio}:1. Adjusting date array.")
                values = list(value_array[::ratio][-len(datetime_array):])
            else:
                min_length = min(len(datetime_array), len(value_array))
                values = list(value_array[-min_length:])
                datetime_array = datetime_array[-min_length:]
            return [bt.num2date(dt).date() for dt in datetime_array]
        else:
            min_length = min(len(datetime_array), len(value_array))
            datetime_array = datetime_array[-min_length:]
            values = value_array[-min_length:]
            return [bt.num2date(dt).date() for dt in datetime_array]

    def get_portfolio_values(self):
        datetime_array = self.touch.datetime.array
        value_array = self.pf_observer.lines.portfolio_value.array
        
        if len(value_array) == 2 * len(datetime_array):
            values = list(value_array[::2][-len(datetime_array):])
        elif len(value_array) > len(datetime_array):
            ratio = len(value_array) // len(datetime_array)
            if ratio > 1:
                values = list(value_array[::ratio][-len(datetime_array):])
            else:
                min_length = min(len(datetime_array), len(value_array))
                values = list(value_array[-min_length:])
        else:
            min_length = min(len(datetime_array), len(value_array))
            values = list(value_array[-min_length:])

        nan_count = sum(1 for v in values if np.isnan(v))
        zero_count = sum(1 for v in values if v == 0)
        neg_count = sum(1 for v in values if v < 0)
        
        if nan_count > 0 or zero_count > 0 or neg_count > 0:
            self.logger.warning(f"Portfolio values contain: {nan_count} NaNs, {zero_count} zeros, {neg_count} negative values")
        
        if len(values) > 0:
            first_val = values[0]
            mid_val = values[len(values)//2] if len(values) > 1 else None
            last_val = values[-1]
            self.logger.info(f"Portfolio value samples: first={first_val}, mid={mid_val}, last={last_val}")
        
        non_nan_values = [v for v in values if not np.isnan(v)]
        if non_nan_values:
            last_valid = non_nan_values[0]
            for i in range(len(values)):
                if np.isnan(values[i]):
                    values[i] = last_valid
                else:
                    last_valid = values[i]
        
        return values
    
    def get_daily_returns(self):
        values = self.get_portfolio_values()
        returns = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                returns.append((values[i] / values[i-1]) - 1)
            else:
                returns.append(0)
        return returns

    def compute_annual_metrics(self) -> dict:
        values = self.get_portfolio_values()
        dates = self.get_portfolio_dates()
        
        if not values or not dates or len(values) != len(dates):
            self.logger.warning(f"Inconsistent data for annual metrics. Dates: {len(dates)}, Values: {len(values)}")
            return {}
            
        df = pd.DataFrame({'Date': dates, 'Value': values})
        df['Date'] = pd.to_datetime(df['Date'])
        nan_count = df['Value'].isna().sum()
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} NaN values in portfolio data")
            df['Value'].fillna(method='ffill', inplace=True)
            df['Value'].fillna(method='bfill', inplace=True)
        
        zero_count = (df['Value'] == 0).sum()
        if zero_count > 0:
            self.logger.warning(f"Found {zero_count} zero values in portfolio data")
            min_non_zero = df['Value'][df['Value'] > 0].min() if not df[df['Value'] > 0].empty else 100000
            df.loc[df['Value'] == 0, 'Value'] = min_non_zero
        
        df['Year'] = df['Date'].dt.year
        
        annual_results = {}
        for year, group in df.groupby('Year'):
            group = group.sort_values('Date')
            if len(group) < 2:
                self.logger.warning(f"Year {year} has insufficient data points: {len(group)}")
                continue
                
            start_val = group['Value'].iloc[0]
            end_val = group['Value'].iloc[-1]
            
            if np.isnan(start_val) or np.isnan(end_val) or start_val <= 0 or end_val <= 0:
                self.logger.warning(f"Year {year} has invalid values: start={start_val}, end={end_val}")
                continue
                
            annual_return = (end_val / start_val) - 1
            daily_ret = group['Value'].pct_change(fill_method=None).dropna()
            
            if daily_ret.empty:
                self.logger.warning(f"Year {year} has no valid daily returns")
                continue
                
            std_dev = daily_ret.std(ddof=1) * np.sqrt(252) if not daily_ret.empty else np.nan
            sharpe = (annual_return - self.p.riskfreerate) / std_dev if std_dev and std_dev > 0 else np.nan
            
            downside = daily_ret[daily_ret < 0]
            downside_std = downside.std(ddof=1) * np.sqrt(252) if not downside.empty else np.nan
            sortino = (annual_return - self.p.riskfreerate) / downside_std if downside_std and downside_std > 0 else np.nan
            
            values_arr = group['Value'].values
            cummax = np.maximum.accumulate(values_arr)
            drawdowns = (values_arr - cummax) / cummax
            mdd = abs(drawdowns.min()) if len(drawdowns) > 0 else np.nan
            
            annual_results[year] = {
                'Return': annual_return * 100,
                'MDD': mdd * 100,
                'Sharpe': sharpe,
                'Sortino': sortino
            }
            
        return annual_results

    def stop(self):
        portfolio_values = self.get_portfolio_values()
        portfolio_dates = self.get_portfolio_dates()
        
        if not portfolio_values or len(portfolio_values) < 2:
            self.logger.error("Insufficient portfolio data for performance metrics.")
            return
            
        start_val = portfolio_values[0]
        end_val = portfolio_values[-1]
        
        if np.isnan(start_val) or np.isnan(end_val) or start_val <= 0 or end_val <= 0:
            self.logger.error(f"Invalid portfolio values: start={start_val}, end={end_val}")
            valid_values = [v for v in portfolio_values if not np.isnan(v) and v > 0]
            if len(valid_values) >= 2:
                start_val = valid_values[0]
                end_val = valid_values[-1]
                self.logger.info(f"Using alternative values: start={start_val}, end={end_val}")
            else:
                self.logger.error("Cannot find valid start and end values.")
                return

        daily_returns = self.get_daily_returns()
        if not daily_returns:
            self.logger.error("No daily returns data for performance metrics.")
            return
            
        std_dev = np.std(daily_returns, ddof=1) * np.sqrt(252)
        start_date = portfolio_dates[0]
        end_date = portfolio_dates[-1]
        
        years = (end_date - start_date).days / 365.25
        cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else 0
        
        negative_returns = [r for r in daily_returns if r < 0]
        downside_dev = np.std(negative_returns, ddof=1) * np.sqrt(252) if negative_returns else 0
        
        risk_free = self.p.riskfreerate
        sortino = (cagr - risk_free) / downside_dev if downside_dev > 0 else np.nan
        sharpe = (cagr - risk_free) / std_dev if std_dev > 0 else np.nan
        
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - peak) / peak
        max_drawdown = abs(np.min(drawdown)) * 100
        
        self.logger.info('-' * 50)
        self.logger.info('Overall Backtest Results:')
        self.logger.info('-' * 50)
        self.logger.info(f"CAGR: {cagr * 100:.2f}%")
        self.logger.info(f"MDD: {max_drawdown:.2f}%")
        self.logger.info(f"Sharpe Ratio: {sharpe:.4f}")
        self.logger.info(f"Sortino Ratio: {sortino:.4f}" if not np.isnan(sortino) else "Sortino Ratio: N/A")
        self.logger.info(f"Total Return: {((end_val / start_val) - 1) * 100:.2f}%")
        self.logger.info(f"Annual Volatility: {std_dev * 100:.2f}%")
        self.logger.info(f"Duration: {years:.2f} years ({start_date} ~ {end_date})")
        
        if hasattr(self.analyzers, 'sharpe'):
            sharpe_ratio = self.analyzers.sharpe.get_analysis().get('sharperatio', None)
            if sharpe_ratio is not None:
                self.logger.info(f"Backtrader Sharpe Ratio: {sharpe_ratio:.4f}")
                
        if hasattr(self.analyzers, 'drawdown'):
            drawdown_info = self.analyzers.drawdown.get_analysis()
            max_dd = drawdown_info.get('max', {}).get('drawdown', None)
            if max_dd is not None:
                self.logger.info(f"Backtrader Max Drawdown: {max_dd:.2f}%")
                
        if hasattr(self.analyzers, 'returns'):
            returns_info = self.analyzers.returns.get_analysis()
            avg_return = returns_info.get('ravg', None)
            if avg_return is not None:
                self.logger.info(f"Backtrader Average Annual Return: {avg_return * 100:.2f}%")
                
        self.logger.info('-' * 50)
        
        annual_metrics = self.compute_annual_metrics()
        self.logger.info("Annual Performance Metrics:")
        for year, metrics in sorted(annual_metrics.items()):
            self.logger.info(f"{year} -> Return: {metrics['Return']:.2f}%, "
                             f"MDD: {metrics['MDD']:.2f}%, "
                             f"Sharpe: {metrics['Sharpe']:.4f}, "
                             f"Sortino: {metrics['Sortino']:.4f}")
        self.logger.info('-' * 50)
