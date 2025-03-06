import logging
import traceback  # 반드시 추가!
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
        super().__init__()
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
        전략 초기화
        """
        self.logger = logging.getLogger('backtest.strategy')
        self._load_configuration()

        # touchstone 데이터 (ex: SPY)
        self.touch = self.getdatabyname(self.p.touchstone)

        # SPY 이동평균 인디케이터
        self.mas = {
            period: bt.indicators.SMA(self.touch.close, period=period, plotname=f"MA_{period}")
            for period in self.p.ma_periods
        }

        # 자산 목록 (터치스톤 제외한 투자자산)
        self._assets = list(self.p.asset_allocation.keys())

        # 포트폴리오 기록용
        self.portfolio_values = []
        self.portfolio_dates = []
        self.daily_returns = []

        # 시장 상태/리밸런싱 지원용
        self.market_regime = None
        self.last_date = None
        self.current_month = None
        self.current_year = None
        self._prev_month_close_prices = {}

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
        _assets에 대해 backtrader data가 있는지 확인
        """
        for asset in self._assets:
            try:
                d = self.getdatabyname(asset)
                self.logger.info(f"Asset data loaded for: {asset}")
            except Exception as e:
                self.logger.error(f"Data missing for asset {asset}: {e}")
                raise ValueError(f"Data missing for asset: {asset}")

    def get_target_allocation(self, current_date) -> Dict[str, float]:
        """
        시장 상태(마켓 레짐)에 따라 자산배분 비율 결정
        """
        if not self.p.use_market_regime:
            self.logger.info(f"[{current_date}] Market regime disabled; using base allocation.")
            return self.p.asset_allocation

        current_price = self.touch.close[0]
        if self.p.ma_type == 'weekly':
            return self._determine_allocation_weekly(current_date, current_price)
        else:
            return self._determine_allocation_daily(current_date, current_price)

    def _determine_allocation_daily(self, current_date, current_price) -> Dict[str, float]:
        """
        일간 이동평균(sma)을 기준으로 시장 상태 결정
        """
        ma_short = self.mas[self.p.ma_periods[0]][0]
        ma_mid = self.mas[self.p.ma_periods[1]][0]
        ma_mid2 = self.mas[self.p.ma_periods[2]][0]
        ma_long = self.mas[self.p.ma_periods[3]][0]

        if current_price > ma_short and ma_short > ma_mid and ma_mid > ma_long:
            regime = "Aggressive"
            alloc = self.p.aggressive_allocation
        elif current_price > ma_mid and ma_mid > ma_long:
            regime = "Moderate"
            alloc = self.p.moderate_allocation
        elif current_price < ma_mid2 and current_price > ma_long:
            regime = "MidDefensive"
            alloc = self.p.mid_defensive_allocation
        elif current_price < ma_long:
            regime = "Defensive"
            alloc = self.p.defensive_allocation
        else:
            regime = "Neutral"
            alloc = self.p.asset_allocation

        self.logger.info(f"[{current_date}] Price={current_price:.2f}, MA(short={ma_short:.2f}, mid={ma_mid:.2f}, mid2={ma_mid2:.2f}, long={ma_long:.2f}) => {regime}")
        return alloc

    def _determine_allocation_weekly(self, current_date, current_price) -> Dict[str, float]:
        """
        주간 이동평균(wma)을 기반으로 시장 상태 결정
        """
        try:
            df = self.touch.p.dataname
            wma_values = {}
            for period in self.p.wma_periods:
                wma_col = f'WMA_{period}'
                if wma_col not in df.columns:
                    self.logger.warning(f"{wma_col} not found in DataFrame => Using base allocation.")
                    return self.p.asset_allocation

                bar_idx = len(df) - 1  # 마지막행 가정 (데모용)
                value = df.iloc[bar_idx].get(wma_col, np.nan)
                if pd.isna(value):
                    self.logger.warning(f"WMA_{period} is NaN => Using base allocation.")
                    return self.p.asset_allocation
                wma_values[period] = value

            wma_short = wma_values[self.p.wma_periods[0]]
            wma_long = wma_values[self.p.wma_periods[1]]

            if current_price > wma_short and wma_short > wma_long:
                regime = "Aggressive"
                alloc = self.p.aggressive_allocation
            elif current_price < wma_short and current_price > wma_long:
                regime = "Moderate"
                alloc = self.p.moderate_allocation
            elif current_price < wma_long:
                regime = "Defensive"
                alloc = self.p.defensive_allocation
            else:
                regime = "Neutral"
                alloc = self.p.asset_allocation

            self.logger.info(f"[{current_date}] Weekly regime => {regime}")
            return alloc

        except Exception as e:
            self.logger.error(f"Error in _determine_allocation_weekly: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self.p.asset_allocation

    def next(self):
        """
        매일 호출
        """
        try:
            current_date = bt.num2date(self.touch.datetime[0]).date()
            current_value = self.broker.getvalue()

            if not pd.isna(current_value) and current_value > 0:
                self.portfolio_values.append(current_value)
                self.portfolio_dates.append(current_date)

                if len(self.portfolio_values) > 1:
                    prev_val = self.portfolio_values[-2]
                    daily_ret = (current_value / prev_val) - 1
                    self.daily_returns.append(daily_ret)

            else:
                self.logger.warning(f"Invalid portfolio value on {current_date}: {current_value}")
                return

            if self.last_date is None:
                # 첫 next()에서 초기화
                self.last_date = current_date
                self.current_month = current_date.month
                self.current_year = current_date.year
                self.logger.info(f"Start date: {current_date}, initial portfolio={current_value:.2f}")
                return

            # 월말(마지막영업일) -> 종가기록
            if is_last_business_day_of_month(current_date):
                self._record_month_end_prices(current_date)

            # 월초 리밸런싱
            if current_date != self.last_date:
                if (current_date.month != self.current_month) or (current_date.year != self.current_year):
                    self.current_month = current_date.month
                    self.current_year = current_date.year

                if is_first_business_day_of_month(current_date):
                    try:
                        alloc = self.get_target_allocation(current_date)
                        self._rebalance_portfolio(current_date, alloc)
                    except Exception as e:
                        self.logger.error(f"Rebalancing failed: {str(e)}")
                        self.logger.error(traceback.format_exc())

                self.last_date = current_date

        except Exception as e:
            self.logger.error(f"Error in next() method: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _record_month_end_prices(self, current_date):
        """
        월말 종가 기록 (터치스톤 제외 자산들 포함)
        """
        for asset in self._assets:
            try:
                d = self.getdatabyname(asset)
                self._prev_month_close_prices[asset] = d.close[0]
                self.logger.info(f"Month-end close for {asset}: {d.close[0]:.2f}")
            except Exception as e:
                self.logger.error(f"Failed to record close price for {asset}: {str(e)}")

    def _rebalance_portfolio(self, current_date, target_allocation: Dict[str, float]):
        """
        목표 배분비율에 맞춰 매도->매수 리밸런싱
        """
        self.logger.info(f"=== {current_date} Rebalancing Start ===")

        total_value = self.broker.getvalue()
        if pd.isna(total_value) or total_value <= 0:
            self.logger.error(f"Invalid portfolio value({total_value}), skip rebalancing.")
            return

        cash_buffer = total_value * (self.p.cash_buffer_percent / 100.0)
        available_value = total_value - cash_buffer

        self.logger.info(f"Portfolio value={total_value:.2f}, cash buffer={cash_buffer:.2f}, available={available_value:.2f}")

        target_shares = {}
        adjustments = {}
        current_shares = {}
        missing_prices = []

        for asset in self._assets:
            d = self.getdatabyname(asset)
            pos = self.getposition(d)
            current_size = pos.size if pos else 0
            current_shares[asset] = current_size

            prev_close = self._prev_month_close_prices.get(asset)
            if (prev_close is None) or pd.isna(prev_close) or (prev_close <= 0):
                missing_prices.append(asset)
                continue

            alloc_ratio = target_allocation.get(asset, 0)
            if alloc_ratio <= 0:
                # 이 자산은 0% 배분 => 전량 매도
                adjustments[asset] = -current_size
            else:
                # 배분 금액
                target_val = available_value * alloc_ratio
                if self.p.fractional_shares:
                    shares = target_val / prev_close
                else:
                    shares = int(target_val / prev_close)
                adjustments[asset] = shares - current_size
            self.logger.info(f"{asset}: prev_close={prev_close:.2f}, current_size={current_size}, alloc_ratio={alloc_ratio:.2f}, adjust={adjustments[asset]}")

        if missing_prices:
            if len(missing_prices) == len(self._assets):
                self.logger.error("All assets missing month-end price => cannot rebalance.")
                return
            else:
                self.logger.warning(f"Missing price for these assets => won't trade them: {missing_prices}")

        # 매도부터
        sell_list = [(a, adj) for a, adj in adjustments.items() if adj < 0]
        for asset, adj in sell_list:
            self.logger.info(f"{current_date}: SELL {asset}, shares={abs(adj)}")
            self.sell(data=self.getdatabyname(asset), size=abs(adj))

        # 현금 추정
        current_cash = self.broker.getcash()
        if pd.isna(current_cash):
            current_cash = 0

        self.logger.info(f"Cash after SELL: {current_cash:.2f}")

        # 매수
        buy_list = [(a, adj) for a, adj in adjustments.items() if adj > 0]
        # 큰 금액부터
        buy_list.sort(key=lambda x: x[1] * self._prev_month_close_prices.get(x[0], 0), reverse=True)

        for asset, adj in buy_list:
            prev_close = self._prev_month_close_prices.get(asset, 0)
            if prev_close <= 0:
                self.logger.warning(f"{asset}: invalid prev_close => skip buy")
                continue

            cost = adj * prev_close
            commission = cost * config.config.get("COMMISSION")
            needed = cost + commission
            if needed > self.broker.getcash():
                # 자금 부족 => 조정
                if self.p.fractional_shares:
                    new_shares = (self.broker.getcash() / (prev_close*(1+config.config.get("COMMISSION"))))
                else:
                    new_shares = int(self.broker.getcash() / (prev_close*(1+config.config.get("COMMISSION"))))

                if new_shares > 0:
                    self.logger.warning(f"{asset}: adjusting shares {adj} -> {new_shares} due to insufficient cash")
                    adj = new_shares
                else:
                    self.logger.error(f"No cash left to buy {asset}.")
                    continue

            self.logger.info(f"{current_date}: BUY {asset}, shares={adj}")
            self.buy(data=self.getdatabyname(asset), size=adj)

        self.logger.info(f"=== {current_date} Rebalancing End ===")

    def notify_order(self, order):
        # 체결/거부 로그
        dt = self.data.datetime.datetime()
        if order.status in [order.Submitted, order.Accepted]:
            return
        elif order.status == order.Completed:
            side = 'BUY' if order.isbuy() else 'SELL'
            self.logger.info(f"{dt} OrderCompleted: {side} {order.executed.size} {order.data._name} @ {order.executed.price:.2f}, Comm={order.executed.comm:.2f}")
        elif order.status == order.Canceled:
            self.logger.warning(f"{dt} OrderCanceled: {order.data._name}")
        elif order.status == order.Margin:
            self.logger.error(f"{dt} MarginError: {order.data._name}")
        elif order.status == order.Rejected:
            self.logger.error(f"{dt} OrderRejected: {order.data._name}")
        else:
            self.logger.warning(f"{dt} UnknownOrderStatus: {order.Status[order.status]}")

    def notify_trade(self, trade):
        if trade.isclosed:
            dt = self.data.datetime.datetime()
            self.logger.info(f"{dt} TradeClosed: {trade.data._name}, PnL={trade.pnl:.2f}, NetPnL={trade.pnlcomm:.2f}")

    def get_portfolio_dates(self):
        """
        Observer로 기록된 portfolio_value와 날짜를 동기화해서 반환
        """
        return self.portfolio_dates
    

    def get_portfolio_values(self):
        """
        Observer에 쌓인 portfolio_value를 꺼내어 list로 반환
        """
        return self.portfolio_values
    

    def get_daily_returns(self):
        vals = self.get_portfolio_values()
        rets = []
        for i in range(1, len(vals)):
            if vals[i-1] != 0:
                rets.append((vals[i]/vals[i-1]) - 1)
            else:
                rets.append(0.0)
        return rets

    def compute_final_metrics(self):
        """
        전체 백테스트 기간에 대한 최종 메트릭 계산
        """
        valid_indices = [i for i, v in enumerate(self.portfolio_values) if not pd.isna(v) and v > 0]
        if not valid_indices:
            self.logger.warning("No valid portfolio values to compute final metrics.")
            return {}
        initial = self.portfolio_values[valid_indices[0]]
        final = self.portfolio_values[valid_indices[-1]]
        start_date = self.portfolio_dates[valid_indices[0]]
        end_date = self.portfolio_dates[valid_indices[-1]]
        years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25

        total_profit = final - initial
        cagr = (final / initial) ** (1 / years) - 1 if years > 0 else np.nan

        # MDD 계산: 누적 최대값 대비 최저값의 하락폭
        arr = np.array([self.portfolio_values[i] for i in valid_indices])
        cummax = np.maximum.accumulate(arr)
        drawdowns = (arr - cummax) / cummax
        mdd = abs(drawdowns.min() * 100)

        # 일별 수익률 기반 표준편차, Sharpe, Sortino 계산
        daily_returns = np.array(self.daily_returns)
        if len(daily_returns) > 1:
            std_annual = np.std(daily_returns, ddof=1) * np.sqrt(252)
            sharpe = ((final / initial - 1) - self.p.riskfreerate) / std_annual if std_annual > 0 else np.nan
            negative = daily_returns[daily_returns < 0]
            downside_std = np.std(negative, ddof=1) * np.sqrt(252) if len(negative) > 1 else np.nan
            sortino = ((final / initial - 1) - self.p.riskfreerate) / downside_std if downside_std > 0 else np.nan
        else:
            std_annual, sharpe, sortino = np.nan, np.nan, np.nan

        return {
        'TotalAsset': round(final, 2),
        'TotalProfit': round(total_profit, 2),
        'CAGR': round(cagr * 100, 2),
        'MDD': round(mdd, 2),
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'StdDev': round(std_annual, 2)
    }

    def compute_annual_metrics(self) -> dict:
        """
        연도별 총자산, 총이익, CAGR, MDD, Sharpe, Sortino, 표준편차 계산
        """
        if not self.portfolio_values or not self.portfolio_dates:
            self.logger.warning("No portfolio data => annual metrics not computed.")
            return {}

        df = pd.DataFrame({'Date': self.portfolio_dates, 'Value': self.portfolio_values})
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year

        results = {}
        for year, grp in df.groupby('Year'):
            grp = grp.sort_values('Date')
            if len(grp) < 2:
                continue
            start_val = grp['Value'].iloc[0]
            end_val = grp['Value'].iloc[-1]
            total_profit = end_val - start_val
            cagr = (end_val / start_val) - 1  # 한 해에 해당하므로 연간 수익률과 동일
            daily_ret = grp['Value'].pct_change().dropna()
            std_annual = daily_ret.std(ddof=1) * np.sqrt(252) if len(daily_ret) >= 2 else np.nan
            sharpe = (cagr - self.p.riskfreerate) / std_annual if (std_annual > 0) else np.nan
            negative = daily_ret[daily_ret < 0]
            downside_std = negative.std(ddof=1) * np.sqrt(252) if len(negative) > 1 else np.nan
            sortino = (cagr - self.p.riskfreerate) / downside_std if (downside_std > 0) else np.nan

            # MDD 계산
            arr_vals = grp['Value'].values
            cummax = np.maximum.accumulate(arr_vals)
            dd = (arr_vals - cummax) / cummax
            mdd = abs(dd.min() * 100)

            results[year] = {
                'TotalAsset': round(end_val,2),
                'TotalProfit': round(total_profit, 2),
                'CAGR': round(cagr * 100, 2),
                'MDD': round(mdd, 2),
                'Sharpe': round(sharpe, 2),
                'Sortino': round(sortino, 2),
                'StdDev': round(std_annual, 2)
            }
        return results
    
    def stop(self):
        """
        백테스트 종료 후 최종결과 로그
        """
        # 필요하면 전체 통계 log
        self.logger.info("Strategy stop() called.")
