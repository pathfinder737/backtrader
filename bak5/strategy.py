#strategy.py
import logging
import numpy as np
import pandas as pd
import backtrader as bt
from typing import Dict, List, Tuple
import config
from data_utils import is_last_business_day_of_month, is_first_business_day_of_month

# 관측자(Observer) 클래스 정의: 포트폴리오 데이터 수집용
class PortfolioDataObserver(bt.Observer):
    """
    포트폴리오 가치, 날짜 및 수익률을 추적하는 커스텀 관측자 클래스
    백트레이더의 라인 기반 데이터 관리 시스템 활용
    """
    lines = ('portfolio_value',)  # 라인 정의
    plotinfo = dict(plot=True, subplot=True, plotname='Portfolio Value')
    
    def next(self):
        # 중요: 주 데이터 피드(첫 번째 데이터)에 대해서만 값을 기록하여 중복 기록 방지
        if self._owner.datas.index(self._owner.data) == 0:
            self.lines.portfolio_value[0] = self._owner.broker.getvalue()

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
        
        # 백트레이더 라인 시스템과 충돌하지 않도록 밑줄 접두사 사용
        self._market_regime = None
        self._last_date = None
        self._current_month = None
        self._current_year = None
        self._prev_month_close_prices = {}
        self._assets = list(self.p.asset_allocation.keys())
        
        # 관측자 추가
        self.pf_observer = PortfolioDataObserver()
        
        self._validate_data()
        self.logger.info(f"Market regime usage: {'Enabled' if self.p.use_market_regime else 'Disabled'}")
        self.logger.info(f"Moving average type: {self.p.ma_type}")
        self.logger.info(f"Fractional shares: {'Allowed' if self.p.fractional_shares else 'Not allowed'}")

    def _load_configuration(self):
        """설정값을 로드합니다."""
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
        """데이터 유효성을 검증합니다."""
        for asset in self._assets:
            try:
                d = self.getdatabyname(asset)
                self.logger.info(f"Asset data loaded for: {asset}")
            except Exception as e:
                self.logger.error(f"Data missing for asset {asset}: {e}")
                raise ValueError(f"Data missing for asset: {asset}")

    def get_target_allocation(self, current_date) -> Dict[str, float]:
        """현재 날짜에 맞는 목표 자산 배분을 계산합니다."""
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
        """일간 이동평균을 기반으로 자산 배분을 결정합니다."""
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
        """주간 이동평균을 기반으로 자산 배분을 결정합니다."""
        wma_values = {}
        for period in self.p.wma_periods:
            wma_col = f'WMA_{period}'
            try:
                # self.touch.p.dataname은 원본 DataFrame을 참조합니다.
                value = self.touch.p.dataname[wma_col].iloc[0]
                if pd.isna(value):
                    raise ValueError("NaN value")
                wma_values[period] = value
            except Exception as e:
                self.logger.warning(f"Failed to get {wma_col}: {e}. Using base allocation.")
                self._market_regime = "Neutral"
                return self.p.asset_allocation
                
        self.logger.info(f"[{current_date}] {self.p.touchstone} Price: {current_price:.2f}, Weekly MA: {wma_values}")
        wma_short = wma_values[self.p.wma_periods[0]]
        wma_long = wma_values[self.p.wma_periods[1]]
        
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
        """
        매일 호출되어 포트폴리오 변화를 추적하고,
        월말 종가 기록 및 월 첫 거래일 리밸런싱을 실행합니다.
        """
        current_date = bt.num2date(self.touch.datetime[0]).date()
        
        # current_value는 사용하지만 portfolio_values에 직접 추가하지 않음
        # 대신 PortfolioDataObserver에서 관리함
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
        """월말에 각 자산의 종가를 기록합니다."""
        for asset in self._assets:
            try:
                d = self.getdatabyname(asset)
                self._prev_month_close_prices[asset] = d.close[0]
                self.logger.info(f"Recorded {asset} month-end close: {d.close[0]:.2f}")
            except Exception as e:
                self.logger.error(f"Failed to record close price for {asset}: {e}")

    def _rebalance_portfolio(self, current_date, target_allocation: Dict[str, float]):
        """목표 배분에 따라 포트폴리오를 리밸런싱합니다."""
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
        """주문 상태 변경 시 호출되어 주문 완료, 취소, 거절 등을 로그에 기록합니다."""
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
        """거래 종료 시 호출되어 거래 결과를 로그에 기록합니다."""
        if trade.isclosed:
            dt = self.data.datetime.datetime()
            self.logger.info(f"Trade Closed [{dt}]: {trade.data._name}, PnL: {trade.pnl:.2f}, Net PnL: {trade.pnlcomm:.2f}")

    def get_portfolio_dates(self):
        """백테스트 기간 동안의 날짜 목록을 반환합니다."""
        datetime_array = self.touch.datetime.array
        value_array = self.pf_observer.lines.portfolio_value.array
        
        # 날짜 배열과 값 배열의 길이를 맞추는 로직
        if len(value_array) > len(datetime_array):
            ratio = len(value_array) // len(datetime_array)
            if ratio > 1:
                self.logger.info(f"Observer array ratio: {ratio}:1. Adjusting date array.")
            return [bt.num2date(dt).date() for dt in datetime_array]
        else:
            # 값 배열이 더 짧은 경우 (드문 경우)
            min_length = min(len(datetime_array), len(value_array))
            return [bt.num2date(dt).date() for dt in datetime_array[-min_length:]]
    
    def get_portfolio_values(self):
        """백테스트 기간 동안의 포트폴리오 가치 목록을 반환합니다."""
        datetime_array = self.touch.datetime.array
        value_array = self.pf_observer.lines.portfolio_value.array
        
        # 1:2 비율 처리
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

        # 데이터 검증
        nan_count = sum(1 for v in values if np.isnan(v))
        zero_count = sum(1 for v in values if v == 0)
        neg_count = sum(1 for v in values if v < 0)
        
        if nan_count > 0 or zero_count > 0 or neg_count > 0:
            self.logger.warning(f"Portfolio values contain: {nan_count} NaNs, {zero_count} zeros, {neg_count} negative values")
        
        # 중간 데이터 검사 - 처음, 중간, 끝 값을 확인
        if len(values) > 0:
            first_val = values[0]
            mid_val = values[len(values)//2] if len(values) > 1 else None
            last_val = values[-1] if len(values) > 0 else None
            self.logger.info(f"Portfolio value samples: first={first_val}, mid={mid_val}, last={last_val}")
        
        # NaN 값 처리 - 이전 유효한 값으로 대체
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
        """일별 수익률을 계산하여 반환합니다."""
        values = self.get_portfolio_values()
        # 여기서는 수정된 get_portfolio_values()를 사용하므로 별도 수정 불필요
        returns = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                returns.append((values[i] / values[i-1]) - 1)
            else:
                returns.append(0)
        return returns

    def compute_annual_metrics(self) -> dict:
        """연도별 성과 지표를 계산합니다."""
        values = self.get_portfolio_values()
        dates = self.get_portfolio_dates()
        
        if not values or not dates or len(values) != len(dates):
            self.logger.warning(f"Inconsistent data for annual metrics. Dates: {len(dates)}, Values: {len(values)}")
            return {}
            
        df = pd.DataFrame({'Date': dates, 'Value': values})
        df['Date'] = pd.to_datetime(df['Date'])
        
        # NaN 값 확인 및 처리
        nan_count = df['Value'].isna().sum()
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} NaN values in portfolio data")
            df['Value'].fillna(method='ffill', inplace=True)
            df['Value'].fillna(method='bfill', inplace=True)
        
        # 0 값 확인 및 처리
        zero_count = (df['Value'] == 0).sum()
        if zero_count > 0:
            self.logger.warning(f"Found {zero_count} zero values in portfolio data")
            # 0값을 이전/이후 값으로 대체
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
            
            # 유효성 검사
            if np.isnan(start_val) or np.isnan(end_val) or start_val <= 0 or end_val <= 0:
                self.logger.warning(f"Year {year} has invalid values: start={start_val}, end={end_val}")
                continue
                
            annual_return = (end_val / start_val) - 1
            
            # FutureWarning 수정
            daily_ret = group['Value'].pct_change(fill_method=None).dropna()
            
            # 추가 로깅으로 NaN 원인 디버깅
            if daily_ret.empty:
                self.logger.warning(f"Year {year} has no valid daily returns")
                continue
                
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
    
        portfolio_values = self.get_portfolio_values()
        portfolio_dates = self.get_portfolio_dates()
        
        if not portfolio_values or len(portfolio_values) < 2:
            self.logger.error("Insufficient portfolio data for performance metrics.")
            return
            
        # 처음과 마지막 값이 유효한지 확인
        start_val = portfolio_values[0]
        end_val = portfolio_values[-1]
        
        if np.isnan(start_val) or np.isnan(end_val) or start_val <= 0 or end_val <= 0:
            self.logger.error(f"Invalid portfolio values: start={start_val}, end={end_val}")
            # 유효한 값 찾기
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
        start_val = portfolio_values[0]
        end_val = portfolio_values[-1]
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