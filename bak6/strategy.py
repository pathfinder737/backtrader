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
        """
        전략 초기화: touchstone(SPY)에 대해 SMA 인디케이터를 생성하고,
        자산 목록을 _assets에 저장. 또한, 데이터 검증(_validate_data) 수행.
        """
        self.logger = logging.getLogger('backtest.strategy')
        self._load_configuration()

        # touchstone 데이터 feed
        self.touch = self.getdatabyname(self.p.touchstone)

        # SPY(=touchstone)에 대해서만 SMA 인디케이터 생성 (이미 기존 코드)
        self.mas = {
            period: bt.indicators.SMA(self.touch.close, period=period, plotname=f"MA_{period}")
            for period in self.p.ma_periods
        }

        # self.assets -> self._assets 로 통일
        self._assets = list(self.p.asset_allocation.keys())

        # 명시적으로 리스트 초기화
        self.portfolio_values = []
        self.portfolio_dates = []
        self.daily_returns = []

        self.market_regime = None
        self.last_date = None
        self.current_month = None
        self.current_year = None
        self.prev_month_close_prices = {}

        # 변경된 변수명에 맞춰 검증 함수 호출
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
        """
        구성된 자산(_assets)에 대해 데이터(feed)가 정상 로드되었는지 확인
        """
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
        주간 이동평균을 기반으로 시장 상태를 판단합니다.
        WMA 컬럼 미존재/NaN 값을 안전하게 처리합니다.
        """
        wma_values = {}
        for period in self.p.wma_periods:
            wma_col = f'WMA_{period}'
            try:
                df = self.touch.p.dataname
                
                # 1. 컬럼 존재 확인
                if wma_col not in df.columns:
                    self.logger.error(f"Column {wma_col} not found. Falling back to base allocation.")
                    return self.p.asset_allocation
                    
                # 2. 현재 데이터프레임 인덱스 가져오기 (백트레이더 내부 방식)
                idx = self.touch.idx
                if idx < 0 or idx >= len(df):
                    self.logger.error(f"Invalid index {idx} for {wma_col}")
                    return self.p.asset_allocation
                    
                # 3. NaN 값 안전하게 처리
                value = df.iloc[idx].get(wma_col)
                if pd.isna(value):
                    self.logger.warning(f"NaN value for {wma_col} at index {idx}. Using base allocation.")
                    return self.p.asset_allocation
                    
                wma_values[period] = value
                
            except Exception as e:
                self.logger.error(f"Error accessing {wma_col}: {str(e)}")
                self.logger.error(traceback.format_exc())
                return self.p.asset_allocation
                
        # 값을 모두 가져왔는지 확인
        if len(wma_values) != len(self.p.wma_periods):
            return self.p.asset_allocation
            
        # 이하 계산 로직은 기존과 동일
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
            
        self.logger.info(f"Weekly market regime: {regime}, Allocation: {allocation}")
        self.market_regime = regime
        return allocation

    def next(self):
        """
        매일 호출되어 포트폴리오 가치, 일별 수익률 기록 및 리밸런싱 실행
        예외 처리와 안전 체크를 추가했습니다.
        """
        try:
            current_date = bt.num2date(self.touch.datetime[0]).date()
            current_value = self.broker.getvalue()
            
            # 유효한 경우만 기록
            if not pd.isna(current_value) and current_value > 0:
                self.portfolio_values.append(current_value)
                self.portfolio_dates.append(current_date)
                
                # 일별 수익률 계산 (2번째 값부터)
                if len(self.portfolio_values) > 1:
                    prev_value = self.portfolio_values[-2]
                    daily_ret = (current_value / prev_value) - 1
                    self.daily_returns.append(daily_ret)
            else:
                self.logger.warning(f"Invalid portfolio value on {current_date}: {current_value}")
                return  # 유효하지 않은 값은 처리 중단
            
            # 첫 번째 실행인 경우 초기화
            if self.last_date is None:
                self.last_date = current_date
                self.current_month = current_date.month
                self.current_year = current_date.year
                self.logger.info(f"Start date: {current_date}, Initial value: {self.portfolio_values[0]:.2f}")
                return
            
            # 월말 종가 기록
            if is_last_business_day_of_month(current_date):
                self._record_month_end_prices(current_date)
            
            # 날짜 변경 확인 및 리밸런싱
            if current_date != self.last_date:
                if current_date.month != self.current_month or current_date.year != self.current_year:
                    self.current_month = current_date.month
                    self.current_year = current_date.year
                    
                # 월 첫 거래일에 리밸런싱
                if is_first_business_day_of_month(current_date):
                    try:
                        allocation = self.get_target_allocation(current_date)
                        self._rebalance_portfolio(current_date, allocation)
                    except Exception as e:
                        self.logger.error(f"Rebalancing failed: {str(e)}")
                        self.logger.error(traceback.format_exc())
                        
                self.last_date = current_date
        except Exception as e:
            self.logger.error(f"Error in next() method: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _record_month_end_prices(self, current_date):
        for asset in self._assets:
            try:
                d = self.getdatabyname(asset)
                self._prev_month_close_prices[asset] = d.close[0]
                self.logger.info(f"Recorded {asset} month-end close: {d.close[0]:.2f}")
            except Exception as e:
                self.logger.error(f"Failed to record close price for {asset}: {e}")

    def _rebalance_portfolio(self, current_date, target_allocation: Dict[str, float]):
       
       
        """
        목표 배분에 따라 포트폴리오를 리밸런싱합니다.
        데이터 품질 검사와 오류 처리를 강화했습니다.
        
        :param current_date: 리밸런싱 날짜
        :param target_allocation: 각 자산별 목표 배분 비율
        """
        self.logger.info(f"=== {current_date} Rebalancing Start ===")
        
        # 포트폴리오 총 가치 확인
        total_value = self.broker.getvalue()
        if pd.isna(total_value) or total_value <= 0:
            self.logger.error(f"Invalid portfolio value: {total_value}. Skipping rebalancing.")
            return
            
        # 캐시 버퍼 계산
        cash_buffer = total_value * (self.p.cash_buffer_percent / 100.0)
        available_value = total_value - cash_buffer
        
        self.logger.info(f"Total value: {total_value:.2f}, Cash buffer: {cash_buffer:.2f}, Available: {available_value:.2f}")
        
        # 목표 금액 및 주식 수 계산
        target_values = {}
        target_shares = {}
        current_shares = {}
        adjustments = {}
        
        # 이전 월말 종가 확인
        missing_prices = []
        
        for asset in self.assets:
            # 현재 포지션 확인
            d = self.getdatabyname(asset)
            pos = self.getposition(d)
            current_shares[asset] = pos.size if pos else 0
            
            # 월말 종가 확인
            prev_close = self.prev_month_close_prices.get(asset)
            if prev_close is None or pd.isna(prev_close) or prev_close <= 0:
                missing_prices.append(asset)
                self.logger.warning(f"Missing or invalid previous month close for {asset}: {prev_close}")
                continue
                
            # 배분 비율 확인
            alloc_ratio = target_allocation.get(asset, 0)
            if alloc_ratio <= 0:
                self.logger.info(f"Asset {asset} has zero allocation, will be sold if owned")
                target_values[asset] = 0
                target_shares[asset] = 0
                adjustments[asset] = -current_shares[asset]
                continue
                
            # 목표 금액과 주식 수 계산
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
        
        # 종가 데이터가 없는 경우 처리
        if missing_prices and missing_prices == self.assets:
            self.logger.error("Missing prices for all assets, cannot rebalance")
            return
        elif missing_prices:
            self.logger.warning(f"Missing prices for {missing_prices}, only rebalancing other assets")
        
        # 매도 주문 처리
        sell_orders = []
        for asset, adj in adjustments.items():
            if adj < 0:
                d = self.getdatabyname(asset)
                self.sell(data=d, size=abs(adj))
                sell_orders.append((asset, abs(adj)))
                self.logger.info(f"{current_date}: Sell {abs(adj)} shares of {asset}")
        
        # 예상 현금 계산
        current_cash = self.broker.getcash()
        if pd.isna(current_cash) or current_cash < 0:
            self.logger.error(f"Invalid cash value: {current_cash}")
            current_cash = 0
            
        estimated_cash = current_cash
        
        for asset, shares in sell_orders:
            prev_close = self.prev_month_close_prices.get(asset, 0)
            if prev_close > 0:
                sell_value = shares * prev_close
                commission = sell_value * config.config.get("COMMISSION")
                estimated_cash += sell_value - commission
        
        self.logger.info(f"Current cash: {current_cash:.2f}, Estimated cash after sales: {estimated_cash:.2f}")
        
        # 매수 주문 처리
        buy_orders = []
        
        # 가용 현금이 있는지 확인
        if pd.isna(estimated_cash) or estimated_cash <= 0:
            self.logger.error(f"No cash available for purchases: {estimated_cash}")
            return
        
        # 매수 순서 결정 (큰 금액부터 처리하여 캐시 고갈 방지)
        buy_adjustments = {asset: adj for asset, adj in adjustments.items() if adj > 0}
        sorted_buys = sorted(buy_adjustments.items(), 
                            key=lambda x: x[1] * self.prev_month_close_prices.get(x[0], 0), 
                            reverse=True)
        
        for asset, adj in sorted_buys:
            if adj <= 0:
                continue
                
            d = self.getdatabyname(asset)
            prev_close = self.prev_month_close_prices.get(asset, 0)
            
            if prev_close <= 0:
                self.logger.warning(f"Invalid price for {asset}: {prev_close}, skipping purchase")
                continue
                
            purchase_cost = adj * prev_close
            commission = purchase_cost * config.config.get("COMMISSION")
            required_cash = purchase_cost + commission
            
            if required_cash > estimated_cash:
                # 가용 자금 부족시 주식 수 조정
                max_shares = estimated_cash / (prev_close * (1 + config.config.get("COMMISSION")))
                if not self.p.fractional_shares:
                    max_shares = int(max_shares)
                    
                if max_shares > 0:
                    self.logger.warning(f"Adjusting buy order for {asset}: {adj} -> {max_shares} due to insufficient cash")
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
        
        # 리밸런싱 결과 요약
        total_sell = sum(self.prev_month_close_prices.get(asset, 0) * shares for asset, shares in sell_orders)
        total_buy = sum(self.prev_month_close_prices.get(asset, 0) * shares for asset, shares in buy_orders)
        
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

        """
        연도별 성과 지표(CAGR, MDD, Sharpe, Sortino 등)를 계산합니다.
        NaN 값을 필터링하여 유효한 데이터만 사용합니다.
        
        :return: 연도별 성과 지표를 포함하는 딕셔너리
        """
        if not self.portfolio_values or not self.portfolio_dates:
            self.logger.warning("Insufficient portfolio data for annual metrics.")
            return {}
            
        # NaN 값 필터링
        dates = np.array(self.portfolio_dates)
        values = np.array(self.portfolio_values)
        valid_indices = ~np.isnan(values)
        
        nan_count = len(values) - np.sum(valid_indices)
        if nan_count > 0:
            self.logger.warning(f"Filtering {nan_count} NaN values from annual metrics data")
            if np.sum(valid_indices) == 0:
                self.logger.error("No valid portfolio values for annual metrics calculation")
                return {}
            dates = dates[valid_indices]
            values = values[valid_indices]
        
        # 유효한 데이터로 DataFrame 생성
        df = pd.DataFrame({'Date': dates, 'Value': values})
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        
        annual_results = {}
        for year, group in df.groupby('Year'):
            group = group.sort_values('Date')
            
            # 충분한 데이터가 있는지 확인
            if len(group) < 2:
                self.logger.warning(f"Insufficient data points for year {year}")
                continue
                
            start_val = group['Value'].iloc[0]
            end_val = group['Value'].iloc[-1]
            
            # 유효한 값인지 확인
            if pd.isna(start_val) or pd.isna(end_val) or start_val <= 0:
                self.logger.warning(f"Invalid values for year {year}: start={start_val}, end={end_val}")
                continue
                
            annual_return = (end_val / start_val) - 1
            
            # 일간 수익률 계산
            daily_ret = group['Value'].pct_change().dropna()
            
            # 충분한 데이터가 있는지 확인
            if len(daily_ret) <= 1:
                std_dev = np.nan
                sharpe = np.nan
                sortino = np.nan
                mdd = np.nan
            else:
                # 표준편차 계산 (연간화)
                std_dev = daily_ret.std(ddof=1) * np.sqrt(252) if not daily_ret.empty else np.nan
                
                # Sharpe 비율 계산
                sharpe = (annual_return - self.p.riskfreerate) / std_dev if not pd.isna(std_dev) and std_dev > 0 else np.nan
                
                # Sortino 비율 계산을 위한 하방 변동성
                downside = daily_ret[daily_ret < 0]
                downside_std = downside.std(ddof=1) * np.sqrt(252) if not downside.empty and len(downside) > 1 else np.nan
                sortino = (annual_return - self.p.riskfreerate) / downside_std if not pd.isna(downside_std) and downside_std > 0 else np.nan
                
                # 최대 낙폭 계산
                values_array = group['Value'].values
                cummax = np.maximum.accumulate(values_array)
                drawdowns = (values_array - cummax) / cummax
                mdd = abs(drawdowns.min()) if len(drawdowns) > 0 else np.nan
            
            annual_results[year] = {
                'Return': annual_return * 100,
                'MDD': mdd * 100 if not pd.isna(mdd) else np.nan,
                'Sharpe': sharpe,
                'Sortino': sortino
            }
        
        return annual_results

    def stop(self):
        """
        백테스트 종료 시 전체 성과 지표를 계산하고 로그에 기록합니다.
        NaN 값을 필터링하고 유효한 데이터만 사용합니다.
        """
        # 유효한 데이터 필터링
        valid_values = []
        valid_dates = []
        valid_returns = []
        
        for i, val in enumerate(self.portfolio_values):
            if i < len(self.portfolio_dates) and not pd.isna(val) and val > 0:
                valid_values.append(val)
                valid_dates.append(self.portfolio_dates[i])
        
        for ret in self.daily_returns:
            if not pd.isna(ret) and np.isfinite(ret):
                valid_returns.append(ret)
        
        # 유효한 데이터가 없는 경우
        if not valid_values:
            self.logger.error("No valid portfolio values for performance metrics.")
            return
        
        if len(valid_values) < 2:
            self.logger.error("Insufficient valid portfolio values for performance metrics.")
            return
        
        # 성과 지표 계산
        std_dev = np.std(valid_returns, ddof=1) * np.sqrt(252) if valid_returns else np.nan
        start_val = valid_values[0]
        end_val = valid_values[-1]
        start_date = valid_dates[0]
        end_date = valid_dates[-1]
        
        years = (end_date - start_date).days / 365.25
        cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else 0
        
        negative_returns = [r for r in valid_returns if r < 0]
        downside_dev = np.std(negative_returns, ddof=1) * np.sqrt(252) if negative_returns and len(negative_returns) > 1 else np.nan
        
        risk_free = self.p.riskfreerate
        sortino = (cagr - risk_free) / downside_dev if not pd.isna(downside_dev) and downside_dev > 0 else np.nan
        sharpe = (cagr - risk_free) / std_dev if not pd.isna(std_dev) and std_dev > 0 else np.nan
        
        # 최대 낙폭 계산
        portfolio_array = np.array(valid_values)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - peak) / peak
        max_drawdown = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else np.nan
        
        # 결과 로깅
        self.logger.info('-' * 50)
        self.logger.info('Overall Backtest Results:')
        self.logger.info('-' * 50)
        self.logger.info(f"CAGR: {cagr * 100:.2f}%")
        self.logger.info(f"MDD: {max_drawdown:.2f}%")
        self.logger.info(f"Sharpe Ratio: {sharpe:.4f}")
        self.logger.info(f"Sortino Ratio: {sortino:.4f}" if not pd.isna(sortino) else "Sortino Ratio: N/A")
        self.logger.info(f"Total Return: {((end_val / start_val) - 1) * 100:.2f}%")
        self.logger.info(f"Annual Volatility: {std_dev * 100:.2f}%")
        self.logger.info(f"Duration: {years:.2f} years ({start_date} ~ {end_date})")
        self.logger.info(f"Data Points: {len(valid_values)} valid out of {len(self.portfolio_values)} total")
        
        # Backtrader 분석기 결과
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
        
        # 연간 성과 지표 (NaN 필터링된 compute_annual_metrics 버전 사용)
        annual_metrics = self.compute_annual_metrics()
        self.logger.info("Annual Performance Metrics:")
        for year, metrics in sorted(annual_metrics.items()):
            self.logger.info(f"{year} -> Return: {metrics['Return']:.2f}%, "
                            f"MDD: {metrics['MDD']:.2f}%, "
                            f"Sharpe: {metrics['Sharpe']:.4f}, "
                            f"Sortino: {metrics['Sortino']:.4f}")
        self.logger.info('-' * 50)