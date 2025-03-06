#strategy.py
import logging
import numpy as np
import pandas as pd
import backtrader as bt
from typing import Dict, List, Tuple
import config
from data_utils import is_last_business_day_of_month, is_first_business_day_of_month

# MonthlyRebalanceStrategy는 월말 리밸런싱 전략을 구현합니다.
class MonthlyRebalanceStrategy(bt.Strategy):
    params = (
        ('touchstone', None),             # 기준 티커 (없으면 config에서 로드)
        ('ma_periods', None),             # 일간 이동평균 기간 리스트 (없으면 config에서 로드)
        ('wma_periods', None),            # 주간 이동평균 기간 리스트 (없으면 config에서 로드)
        ('ma_type', None),                # 이동평균 타입 ('daily' 또는 'weekly')
        ('asset_allocation', None),       # 기본 자산 배분 (없으면 config에서 로드)
        ('aggressive_allocation', None),  # 공격적 배분 (없으면 config에서 로드)
        ('moderate_allocation', None),    # 중립적 배분 (없으면 config에서 로드)
        ('defensive_allocation', None),   # 방어적 배분 (없으면 config에서 로드)
        ('mid_defensive_allocation', None), # 중간 방어적 배분 (없으면 config에서 로드)
        ('riskfreerate', None),           # 리스크 프리 레이트 (없으면 config에서 로드)
        ('use_market_regime', None),      # 시장 상태 활용 여부 (없으면 config에서 로드)
        ('fractional_shares', None),      # 소수점 주식 거래 허용 여부 (없으면 config에서 로드)
        ('cash_buffer_percent', None),    # 현금 버퍼 비율 (없으면 config에서 로드)
    )

    def __init__(self):
        self.logger = logging.getLogger('backtest.strategy')
        self._load_configuration()
        self.touch = self.getdatabyname(self.p.touchstone)
        self.mas = { period: bt.indicators.SMA(self.touch.close, period=period, plotname=f"MA_{period}") 
                     for period in self.p.ma_periods }
        self.market_regime = None
        self.last_date = None
        self.current_month = None
        self.current_year = None
        self.daily_returns = []
        self.portfolio_values = []
        self.portfolio_dates = []
        self.prev_month_close_prices = {}
        self.assets = list(self.p.asset_allocation.keys())
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
        for asset in self.assets:
            try:
                d = self.getdatabyname(asset)
                self.logger.info(f"Asset data loaded for: {asset}")
            except Exception as e:
                self.logger.error(f"Data missing for asset {asset}: {e}")
                raise ValueError(f"Data missing for asset: {asset}")

    def get_target_allocation(self, current_date) -> Dict[str, float]:
        if not self.p.use_market_regime:
            self.logger.info(f"[{current_date}] Market regime disabled; using base allocation.")
            self.market_regime = "Neutral"
            return self.p.asset_allocation
        current_price = self.touch.close[0]
        if self.p.ma_type == 'weekly':
            return self._determine_allocation_weekly(current_date, current_price)
        else:
            return self._determine_allocation_daily(current_date, current_price)
    
    def _determine_allocation_daily(self, current_date, current_price) -> Dict[str, float]:
        ma_values = { period: self.mas[f'MA_{period}'][0] for period in self.p.ma_periods }
        self.logger.info(f"[{current_date}] {self.p.touchstone} Price: {current_price:.2f}, MA values: {ma_values}")
        ma_short = ma_values[self.p.ma_periods[0]]
        ma_mid = ma_values[self.p.ma_periods[1]]
        ma_mid2 = ma_values[self.p.ma_periods[2]]
        ma_long = ma_values[self.p.ma_periods[3]]
        if current_price > ma_short and ma_short > ma_mid and ma_mid > ma_long:
            self.market_regime = "Aggressive"
            allocation = self.p.aggressive_allocation
        elif current_price > ma_mid and ma_mid > ma_long:
            self.market_regime = "Moderate"
            allocation = self.p.moderate_allocation
        elif current_price < ma_mid2 and current_price > ma_long:
            self.market_regime = "MidDefensive"
            allocation = self.p.mid_defensive_allocation
        elif current_price < ma_long:
            self.market_regime = "Defensive"
            allocation = self.p.defensive_allocation
        else:
            self.market_regime = "Neutral"
            allocation = self.p.asset_allocation
        self.logger.info(f"Determined market regime: {self.market_regime}, Allocation: {allocation}")
        return allocation

    def _determine_allocation_weekly(self, current_date, current_price) -> Dict[str, float]:
        """
        주간 이동평균을 기반으로 시장 상태를 판단합니다.
        DataFrame에서 'WMA_10'과 'WMA_30' 컬럼을 직접 읽어옵니다.
        """
        wma_values = {}
        for period in self.p.wma_periods:
            wma_col = f'WMA_{period}'
            try:
                # self.touch.p.dataname는 원본 DataFrame을 참조합니다.
                value = self.touch.p.dataname[wma_col].iloc[0]
                if pd.isna(value):
                    raise ValueError("NaN value")
                wma_values[period] = value
            except Exception as e:
                self.logger.warning(f"Failed to get {wma_col}: {e}. Using base allocation.")
                self.market_regime = "Neutral"
                return self.p.asset_allocation
        self.logger.info(f"[{current_date}] {self.p.touchstone} Price: {current_price:.2f}, Weekly MA: {wma_values}")
        wma_short = wma_values[self.p.wma_periods[0]]
        wma_long = wma_values[self.p.wma_periods[1]]
        if current_price > wma_short and wma_short > wma_long:
            regime = "Aggressive"
            allocation = self.p.aggressive_allocation
        elif current_price < wma_short and current_price > wma_long:
            regime = "Moderate"
            allocation = self.p.moderate_allocation
        elif current_price < wma_long:
            regime = "Defensive"
            allocation = self.p.defensive_allocation
        else:
            regime = "Neutral"
            allocation = self.p.asset_allocation
        self.logger.info(f"Determined weekly market regime: {regime}, Allocation: {allocation}")
        return allocation

    def next(self):
        """
        매일 호출되어 포트폴리오 가치, 일별 수익률을 기록하고,
        월말 종가 기록 및 월 첫 거래일 리밸런싱을 실행합니다.
        """
        current_date = bt.num2date(self.touch.datetime[0]).date()
        current_value = self.broker.getvalue()
        self.portfolio_values.append(current_value)
        self.portfolio_dates.append(current_date)
        if len(self.portfolio_values) > 1:
            daily_ret = (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1
            self.daily_returns.append(daily_ret)
        if self.last_date is None:
            self.last_date = current_date
            self.current_month = current_date.month
            self.current_year = current_date.year
            self.logger.info(f"Start date: {current_date}, Initial value: {self.portfolio_values[0]:.2f}")
            return
        if is_last_business_day_of_month(current_date):
            self._record_month_end_prices(current_date)
        if current_date != self.last_date:
            if current_date.month != self.current_month or current_date.year != self.current_year:
                self.current_month = current_date.month
                self.current_year = current_date.year
            if is_first_business_day_of_month(current_date):
                try:
                    allocation = self.get_target_allocation(current_date)
                    self._rebalance_portfolio(current_date, allocation)
                except Exception as e:
                    self.logger.error(f"Rebalancing failed: {e}")
            self.last_date = current_date

    def _record_month_end_prices(self, current_date):
        """
        월말에 각 자산의 종가를 기록합니다.
        
        :param current_date: 현재 날짜
        """
        for asset in self.assets:
            try:
                d = self.getdatabyname(asset)
                self.prev_month_close_prices[asset] = d.close[0]
                self.logger.info(f"Recorded {asset} month-end close: {d.close[0]:.2f}")
            except Exception as e:
                self.logger.error(f"Failed to record close price for {asset}: {e}")

    def _rebalance_portfolio(self, current_date, target_allocation: Dict[str, float]):
        """
        목표 배분에 따라 포트폴리오를 리밸런싱합니다.
        매도 주문 후 매수 주문을 실행하며, 잔액 및 수수료를 고려합니다.
        
        :param current_date: 리밸런싱 날짜
        :param target_allocation: 각 자산별 목표 배분 비율
        """
        self.logger.info(f"=== {current_date} Rebalancing Start ===")
        total_value = self.broker.getvalue()
        cash_buffer = total_value * (self.p.cash_buffer_percent / 100.0)
        available_value = total_value - cash_buffer
        self.logger.info(f"Total value: {total_value:.2f}, Cash buffer: {cash_buffer:.2f}, Available: {available_value:.2f}")
        target_values = {}
        target_shares = {}
        current_shares = {}
        adjustments = {}
        for asset in self.assets:
            d = self.getdatabyname(asset)
            pos = self.getposition(d)
            current_shares[asset] = pos.size if pos else 0
            prev_close = self.prev_month_close_prices.get(asset)
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
            prev_close = self.prev_month_close_prices.get(asset, 0)
            sell_value = shares * prev_close
            commission = sell_value * config.config.get("COMMISSION")
            estimated_cash += sell_value - commission
        self.logger.info(f"Current cash: {current_cash:.2f}, Estimated cash after sales: {estimated_cash:.2f}")
        buy_orders = []
        for asset, adj in adjustments.items():
            if adj > 0:
                d = self.getdatabyname(asset)
                prev_close = self.prev_month_close_prices.get(asset, 0)
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
        total_sell = sum(self.prev_month_close_prices.get(asset, 0) * shares for asset, shares in sell_orders)
        total_buy = sum(self.prev_month_close_prices.get(asset, 0) * shares for asset, shares in buy_orders)
        self.logger.info(f"Total Sell Value: {total_sell:.2f}, Total Buy Value: {total_buy:.2f}")
        self.logger.info(f"Rebalancing complete. Portfolio value: {total_value:.2f}, Estimated remaining cash: {estimated_cash:.2f}")
        self.logger.info(f"=== {current_date} Rebalancing End ===")

    def notify_order(self, order):
        """
        주문 상태 변경 시 호출되어 주문 완료, 취소, 거절 등을 로그에 기록합니다.
        
        :param order: 주문 객체
        """
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
        """
        거래 종료 시 호출되어 거래 결과를 로그에 기록합니다.
        
        :param trade: 거래 객체
        """
        if trade.isclosed:
            dt = self.data.datetime.datetime()
            self.logger.info(f"Trade Closed [{dt}]: {trade.data._name}, PnL: {trade.pnl:.2f}, Net PnL: {trade.pnlcomm:.2f}")

    def compute_annual_metrics(self) -> dict:
        """
        연도별 성과 지표(CAGR, MDD, Sharpe, Sortino 등)를 계산합니다.
        
        :return: 연도별 성과 지표를 포함하는 딕셔너리
        """
        if not self.portfolio_values or not self.portfolio_dates:
            self.logger.warning("Insufficient portfolio data for annual metrics.")
            return {}
        df = pd.DataFrame({'Date': self.portfolio_dates, 'Value': self.portfolio_values})
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        annual_results = {}
        for year, group in df.groupby('Year'):
            group = group.sort_values('Date')
            start_val = group['Value'].iloc[0]
            end_val = group['Value'].iloc[-1]
            annual_return = (end_val / start_val) - 1
            daily_ret = group['Value'].pct_change().dropna()
            std_dev = daily_ret.std(ddof=1) * np.sqrt(252) if not daily_ret.empty else np.nan
            sharpe = (annual_return - self.p.riskfreerate) / std_dev if std_dev and std_dev > 0 else np.nan
            downside = daily_ret[daily_ret < 0]
            downside_std = downside.std(ddof=1) * np.sqrt(252) if not downside.empty else np.nan
            sortino = (annual_return - self.p.riskfreerate) / downside_std if downside_std and downside_std > 0 else np.nan
            values = group['Value'].values
            cummax = np.maximum.accumulate(values)
            drawdowns = (values - cummax) / cummax
            mdd = abs(drawdowns.min()) if len(drawdowns) > 0 else np.nan
            annual_results[year] = {
                'Return': annual_return * 100,
                'MDD': mdd * 100,
                'Sharpe': sharpe,
                'Sortino': sortino
            }
        return annual_results

    def stop(self):
        """
        백테스트 종료 시 전체 성과 지표를 계산하고 로그에 기록합니다.
        """
        if not self.daily_returns:
            self.logger.error("No daily returns data for performance metrics.")
            return
        std_dev = np.std(self.daily_returns, ddof=1) * np.sqrt(252)
        start_val = self.portfolio_values[0]
        end_val = self.portfolio_values[-1]
        start_date = self.portfolio_dates[0]
        end_date = self.portfolio_dates[-1]
        years = (end_date - start_date).days / 365.25
        cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else 0
        negative_returns = [r for r in self.daily_returns if r < 0]
        downside_dev = np.std(negative_returns, ddof=1) * np.sqrt(252) if negative_returns else 0
        risk_free = self.p.riskfreerate
        sortino = (cagr - risk_free) / downside_dev if downside_dev > 0 else np.nan
        sharpe = (cagr - risk_free) / std_dev if std_dev > 0 else np.nan
        portfolio_array = np.array(self.portfolio_values)
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
