# strategy.py
# 백테스팅에 사용될 월말 리밸런싱 전략 클래스를 정의합니다.

import logging
import numpy as np
import pandas as pd
import backtrader as bt
import config
from date_data_utils import get_last_business_day, get_first_business_day

class MonthlyRebalanceStrategy(bt.Strategy):
    """
    월말 가격 기준으로 리밸런싱 결정을 하고 다음 달 첫 거래일에 실행하는 전략 클래스입니다.
    이 전략은 여러 이동평균(MA21, MA50, MA200)에 따라 자산 배분을 조정합니다.
    """
    params = (
        ('asset_allocation', config.ASSET_ALLOCATION),             # 기본 자산배분
        ('aggressive_allocation', config.AGGRESSIVE_ALLOCATION),   # 공격적 배분 (SPY > MA21)
        ('moderate_allocation', config.MODERATE_ALLOCATION),       # 중립적 배분 (SPY > MA50)
        ('defensive_allocation', config.DEFENSIVE_ALLOCATION),     # 방어적 배분 (SPY < MA200)
        ('ma_periods', config.MA_PERIODS),                         # 이동평균 기간 리스트
        ('riskfreerate', config.RISK_FREE_RATE),                   # 리스크 프리 레이트
        ('use_market_regime', getattr(config, 'USE_MARKET_REGIME', True)),  # 시장 상태 활용 여부 (기본값: True)
        ('fractional_shares', getattr(config, 'FRACTIONAL_SHARES', False)),  # 소수점 주식 거래 허용 여부
    )

    def __init__(self):
        """
        전략 초기화 함수입니다. 여러 이동평균 지표 설정 및 자산 배분 설정을 진행합니다.
        """
        # 이동평균 계산은 config.TOUCHSTONE 기준으로 진행
        self.touch = self.getdatabyname(config.TOUCHSTONE)
        
        # 여러 이동평균 지표 생성
        self.mas = {}
        for period in self.p.ma_periods:
            self.mas[period] = bt.indicators.SMA(self.touch.close, period=period, 
                                               plotname=f"MA_{period}")
        
        self.market_regime = None
        
        # 자산배분은 config의 asset_allocation 사용
        self.asset_allocation = self.p.asset_allocation
        self.assets = list(self.asset_allocation.keys())
        
        # 리밸런싱 상태 관리
        self.last_date = None
        self.current_month = None
        self.current_year = None
        
        # 성과 추적 데이터
        self.daily_returns = []
        self.portfolio_values = []
        self.portfolio_dates = []
        self.prev_month_close_prices = {}
        
        # 자산 배분에 있지만 데이터가 없는 경우 확인
        for asset_name in self.asset_allocation.keys():
            try:
                d = self.getdatabyname(asset_name)
                logging.info(f"자산 확인: {asset_name} 데이터 로드됨")
            except Exception as e:
                logging.error(f"오류: 자산 배분에 포함된 {asset_name}의 데이터를 찾을 수 없습니다.")
                raise ValueError(f"자산 배분에 포함된 {asset_name}의 데이터가 로드되지 않았습니다. TICKERS 목록을 확인하세요.")
        
        # 시장 상태에 따른 자산 배분 설정
        logging.info(f"시장 상태 활용: {'활성화' if self.p.use_market_regime else '비활성화'}")
        logging.info(f"소수점 주식 거래: {'허용' if self.p.fractional_shares else '불가'}")

    def get_target_allocation(self, current_date):
        """
        이동평균선 간의 관계에 따른 목표 자산 배분을 반환합니다.
        
        Args:
            current_date: 현재 날짜
            
        Returns:
            자산별 배분 비율 딕셔너리
        """
        # 시장 상태 판단 안함 (USE_MARKET_REGIME = False)
        if not self.p.use_market_regime:
            logging.info(f"[{current_date}] 시장 상태 활용 옵션 비활성화: 기본 자산 배분 사용")
            return self.asset_allocation
        
        # 현재 가격과 이동평균선 값 가져오기
        current_price = self.touch.close[0]
        ma21 = self.mas[21][0]
        ma50 = self.mas[50][0]
        ma200 = self.mas[200][0]
        
        # 시장 상태 로깅
        logging.info(f"[{current_date}] {config.TOUCHSTONE} 가격: {current_price:.2f}")
        logging.info(f"MA21: {ma21:.2f}, MA50: {ma50:.2f}, MA200: {ma200:.2f}")
        
        # 변경된 로직: 이동평균선 간의 관계를 기준으로 시장 상태 판단
        # 1. MA21 > MA50 > MA200 인 경우 - 공격적 배분 (Aggressive)
        if current_price > ma21 and ma21 > ma50 and ma50 > ma200:
            self.market_regime = "Aggressive"
            logging.info(f"시장 상태: {self.market_regime} (MA21 > MA50 > MA200)")
            return self.p.aggressive_allocation
        
        # 2. MA50 > MA200 인 경우 - 중립적 배분 (Moderate)
        elif current_price > ma50 and ma50 > ma200:
            self.market_regime = "Moderate"
            logging.info(f"시장 상태: {self.market_regime} (MA50 > MA200)")
            return self.p.moderate_allocation
        
        # 3. MA50 < MA200 인 경우 - 방어적 배분 (Defensive)
        elif current_price < ma200:
            self.market_regime = "Defensive"
            logging.info(f"시장 상태: {self.market_regime} (MA50 < MA200)")
            return self.p.defensive_allocation
        
        # 4. 그 외 경우 - 기본 배분 (Neutral)
        else:
            self.market_regime = "Neutral"
            logging.info(f"시장 상태: {self.market_regime} (기타 조건)")
            return self.asset_allocation
        
    def next(self):
        """
        백테스팅의 각 단계에서 실행되는 메인 로직입니다.
        매일 포트폴리오 가치를 추적하고, 월 첫 거래일에 리밸런싱을 수행합니다.
        """
        current_date = bt.num2date(self.touch.datetime[0]).date()
        
        # 포트폴리오 가치 추적
        self.portfolio_values.append(self.broker.getvalue())
        self.portfolio_dates.append(current_date)
        
        # 일별 수익률 계산
        if len(self.portfolio_values) > 1:
            daily_return = (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1
            self.daily_returns.append(daily_return)
            
        # 초기화
        if self.last_date is None:
            self.last_date = current_date
            self.current_month = current_date.month
            self.current_year = current_date.year
            logging.info(f"백테스트 시작일: {current_date}, 초기 포트폴리오 가치: {self.portfolio_values[0]:.2f}")
            return

        # 월말 시, 각 자산의 종가 기록
        if current_date == get_last_business_day(current_date):
            for asset_name in self.assets:
                try:
                    d = self.getdatabyname(asset_name)
                    self.prev_month_close_prices[asset_name] = d.close[0]
                    logging.info(f"{asset_name} 월말({current_date}) 종가 기록: {d.close[0]:.2f}")
                except Exception as e:
                    logging.error(f"{asset_name} 종가 기록 실패: {str(e)}")

        # 날짜 변경 감지
        if current_date != self.last_date:
            # 월 또는 연도 변경 감지
            if current_date.month != self.current_month or current_date.year != self.current_year:
                self.current_month = current_date.month
                self.current_year = current_date.year
                
            # 매월 첫 거래일에 리밸런싱
            if current_date == get_first_business_day(current_date):
                try:
                    target_allocation = self.get_target_allocation(current_date)
                    self.rebalance_portfolio(current_date, target_allocation)
                except Exception as e:
                    logging.error(f"리밸런싱 실패: {str(e)}")
                    import traceback
                    logging.error(traceback.format_exc())

        self.last_date = current_date
    
    def rebalance_portfolio(self, current_date, target_allocation):
        """
        목표 자산 배분에 따라 포트폴리오를 리밸런싱합니다.
        매도 주문을 먼저 실행한 후 매수 주문을 실행하여 현금 흐름을 최적화합니다.
        
        Args:
            current_date: 현재 날짜
            target_allocation: 목표 자산 배분 비율
        """
        logging.info(f"=== {current_date} 리밸런싱 시작 ===")
        total_value = self.broker.getvalue()
        cash_buffer = total_value * 0.01  # 총 가치의 1%를 현금 버퍼로 유지
        available_value = total_value - cash_buffer
        
        logging.info(f"총 포트폴리오 가치: {total_value:.2f}, 현금 버퍼: {cash_buffer:.2f}")
        logging.info(f"리밸런싱 가용 가치: {available_value:.2f}")
        
        # 목표 주식 수 계산
        target_values = {}
        target_shares = {}
        current_shares = {}
        adjust_shares = {}
        
        # 1단계: 현재 포지션 확인 및 목표 포지션 계산
        for asset_name in self.assets:
            d = self.getdatabyname(asset_name)
            pos = self.getposition(d)
            current_shares[asset_name] = pos.size if pos else 0
            
            prev_close = self.prev_month_close_prices.get(asset_name, None)
            if prev_close is None:
                logging.warning(f"{asset_name}의 이전 월 종가 정보가 없습니다. 해당 자산 리밸런싱 건너뜀.")
                continue
                
            allocation = target_allocation.get(asset_name, 0)
            target_value = available_value * allocation
            target_values[asset_name] = target_value
            
            # 소수점 주식 거래 여부에 따라 주식 수 계산
            if self.p.fractional_shares:
                shares_to_buy = target_value / prev_close
            else:
                shares_to_buy = int(target_value / prev_close)
                
            target_shares[asset_name] = shares_to_buy
            adjust_shares[asset_name] = target_shares[asset_name] - current_shares[asset_name]
            
            logging.info(f"자산: {asset_name}, 목표 배분율: {allocation:.4f}, "
                        f"목표 금액: {target_value:.2f}, 목표 주식 수: {shares_to_buy:.4f}, "
                        f"현재 주식 수: {current_shares[asset_name]}, 조정 주식 수: {adjust_shares[asset_name]:.4f}")
        
        # 2단계: 매도 주문 먼저 실행
        sell_orders = []
        for asset_name, shares in adjust_shares.items():
            if shares < 0:  # 매도 필요
                d = self.getdatabyname(asset_name)
                abs_shares = abs(shares)
                self.sell(data=d, size=abs_shares)
                sell_orders.append((asset_name, abs_shares))
                logging.info(f"{current_date}: {asset_name} {abs_shares:.4f}주 매도 주문")
        
        # 3단계: 현재 현금 확인 및 매도 후 예상 현금 계산
        current_cash = self.broker.getcash()
        estimated_cash = current_cash
        
        for asset_name, shares in sell_orders:
            prev_close = self.prev_month_close_prices.get(asset_name, 0)
            sell_value = shares * prev_close
            commission = sell_value * config.COMMISSION
            estimated_cash += sell_value - commission
        
        logging.info(f"현재 현금: {current_cash:.2f}, 매도 후 추정 현금: {estimated_cash:.2f}")
        
        # 4단계: 매수 주문 실행 (현금 제약 고려)
        buy_orders = []
        for asset_name, shares in adjust_shares.items():
            if shares > 0:  # 매수 필요
                d = self.getdatabyname(asset_name)
                prev_close = self.prev_month_close_prices.get(asset_name, 0)
                
                # 수수료를 고려한 필요 현금 계산
                purchase_cost = shares * prev_close
                commission = purchase_cost * config.COMMISSION
                required_cash = purchase_cost + commission
                
                # 현금 제약 확인 및 조정
                if required_cash > estimated_cash:
                    # 가용 현금에 맞게 매수 수량 조정
                    max_shares = int(estimated_cash / (prev_close * (1 + config.COMMISSION)))
                    if max_shares > 0:
                        logging.warning(f"현금 부족으로 매수 수량 조정: {asset_name} {shares:.4f}주 -> {max_shares}주")
                        shares = max_shares
                    else:
                        logging.error(f"현금 부족으로 매수 불가: {asset_name}")
                        continue
                
                # 매수 주문 실행
                if shares > 0:
                    self.buy(data=d, size=shares)
                    buy_cost = shares * prev_close
                    buy_commission = buy_cost * config.COMMISSION
                    estimated_cash -= (buy_cost + buy_commission)
                    buy_orders.append((asset_name, shares))
                    logging.info(f"{current_date}: {asset_name} {shares:.4f}주 매수 주문")
        
        # 5단계: 결과 요약
        total_sell_value = sum(self.prev_month_close_prices.get(asset, 0) * shares for asset, shares in sell_orders)
        total_buy_value = sum(self.prev_month_close_prices.get(asset, 0) * shares for asset, shares in buy_orders)
        
        logging.info(f"총 매도 가치: {total_sell_value:.2f}, 총 매수 가치: {total_buy_value:.2f}")
        logging.info(f"리밸런싱 완료: 총 포트폴리오 가치: {total_value:.2f}, 추정 잔여 현금: {estimated_cash:.2f}")
        logging.info(f"=== {current_date} 리밸런싱 종료 ===")

    def notify_order(self, order):
        """주문 상태 알림 처리"""
        if order.status in [order.Submitted, order.Accepted]:
            # 주문 접수 상태는 로깅하지 않음
            return
            
        dt = self.data.datetime.datetime()
        
        if order.status in [order.Completed]:
            order_type = 'BUY' if order.isbuy() else 'SELL'
            logging.info(f"주문 완료 [{dt}]: {order_type} {order.executed.size:.4f}주 "
                        f"{order.data._name} @ {order.executed.price:.2f}, "
                        f"수수료: {order.executed.comm:.2f}, 총액: {order.executed.value:.2f}")
        elif order.status in [order.Canceled]:
            logging.warning(f"주문 취소됨 [{dt}]: {order.data._name}")
        elif order.status in [order.Margin]:
            logging.error(f"증거금 부족 [{dt}]: {order.data._name}")
        elif order.status in [order.Rejected]:
            logging.error(f"주문 거부됨 [{dt}]: {order.data._name}")
        else:
            logging.warning(f"주문 상태 알 수 없음 [{dt}]: {order.Status[order.status]}")

    def notify_trade(self, trade):
        """거래 완료 알림 처리"""
        if trade.isclosed:
            dt = self.data.datetime.datetime()
            logging.info(f"거래 종료 [{dt}]: {trade.data._name}, "
                        f"손익: {trade.pnl:.2f}, 순손익: {trade.pnlcomm:.2f}")

    def compute_annual_metrics(self):
        """연도별 성과 지표를 계산합니다."""
        # 연도별 성과 계산
        df = pd.DataFrame({
            'Date': self.portfolio_dates,
            'Value': self.portfolio_values
        })
        df['Date'] = pd.to_datetime(df['Date'])  # datetime 형식으로 변환
        df['Year'] = df['Date'].dt.year

        annual_results = {}
        for year, group in df.groupby('Year'):
            group = group.sort_values('Date')
            start_value = group['Value'].iloc[0]
            end_value = group['Value'].iloc[-1]
            
            # 연간 수익률
            annual_return = (end_value / start_value - 1)
            
            # 일별 수익률 계산 (해당 연도)
            group = group.reset_index(drop=True)
            daily_returns = group['Value'].pct_change().dropna()
            
            # 변동성 및 성과 지표
            std_dev = daily_returns.std(ddof=1) * np.sqrt(252) if not daily_returns.empty else np.nan
            sharpe = (annual_return - self.p.riskfreerate) / std_dev if std_dev and std_dev > 0 else np.nan
            
            downside = daily_returns[daily_returns < 0]
            downside_std = downside.std(ddof=1) * np.sqrt(252) if not downside.empty else np.nan
            sortino = (annual_return - self.p.riskfreerate) / downside_std if downside_std and downside_std > 0 else np.nan
            
            # 최대 낙폭(MDD) 계산
            values = group['Value'].values
            cummax = np.maximum.accumulate(values)
            drawdowns = (values - cummax) / cummax
            mdd = abs(drawdowns.min()) if len(drawdowns) > 0 else np.nan

            annual_results[year] = {
                'Return': annual_return * 100,  # CAGR에서 Annual Return으로 수정
                'MDD': mdd * 100,
                'Sharpe': sharpe,
                'Sortino': sortino
            }
        return annual_results

    def stop(self):
        """백테스트 종료 시 성과 분석 결과를 계산하고 출력합니다."""
        if not self.daily_returns:
            logging.error("성과 지표 계산에 필요한 일별 수익률 데이터가 없습니다.")
            return
            
        # 기본 지표 계산
        std_dev = np.std(self.daily_returns, ddof=1) * np.sqrt(252)
        start_value = self.portfolio_values[0]
        end_value = self.portfolio_values[-1]
        start_date = self.portfolio_dates[0]
        end_date = self.portfolio_dates[-1]
        years = (end_date - start_date).days / 365.25
        
        # CAGR 계산 (연평균 복리 수익률)
        cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else 0

        # 위험 조정 수익률 지표
        negative_returns = [r for r in self.daily_returns if r < 0]
        downside_deviation = np.std(negative_returns, ddof=1) * np.sqrt(252) if negative_returns else 0
        risk_free_rate = self.p.riskfreerate
        sortino = (cagr - risk_free_rate) / downside_deviation if downside_deviation > 0 else np.nan
        sharpe = (cagr - risk_free_rate) / std_dev if std_dev > 0 else np.nan
        
        # 최대 낙폭 계산
        portfolio_array = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - peak) / peak
        max_drawdown = abs(np.min(drawdown)) * 100

        # 내장 분석기 결과 활용
        analyzers = self.analyzers
        
        # 결과 출력
        logging.info('-' * 50)
        logging.info('전체 백테스트 결과:')
        logging.info('-' * 50)
        logging.info(f"CAGR: {cagr * 100:.2f}%")
        logging.info(f"MDD: {max_drawdown:.2f}%")
        logging.info(f"Sharpe Ratio: {sharpe:.4f}")
        logging.info(f"Sortino Ratio: {sortino:.4f}" if not np.isnan(sortino) else "Sortino Ratio: N/A")
        logging.info(f"총 수익률: {((end_value / start_value) - 1) * 100:.2f}%")
        logging.info(f"연간 표준편차: {std_dev * 100:.2f}%")
        logging.info(f"투자 기간: {years:.2f}년 ({start_date} ~ {end_date})")
        
        # 내장 분석기 결과 출력
        if hasattr(analyzers, 'sharpe'):
            sharpe_ratio = analyzers.sharpe.get_analysis().get('sharperatio', None)
            if sharpe_ratio is not None:
                logging.info(f"Backtrader Sharpe Ratio: {sharpe_ratio:.4f}")
        
        if hasattr(analyzers, 'drawdown'):
            drawdown_info = analyzers.drawdown.get_analysis()
            max_dd = drawdown_info.get('max', {}).get('drawdown', None)
            if max_dd is not None:
                logging.info(f"Backtrader Max Drawdown: {max_dd:.2f}%")
        
        if hasattr(analyzers, 'returns'):
            returns_info = analyzers.returns.get_analysis()
            avg_return = returns_info.get('ravg', None)
            if avg_return is not None:
                logging.info(f"Backtrader 평균 연간 수익률: {avg_return * 100:.2f}%")
        
        logging.info('-' * 50)

        # 연도별 성과 지표 계산 및 출력
        annual_metrics = self.compute_annual_metrics()
        logging.info("연도별 성과 지표:")
        for year, metrics in sorted(annual_metrics.items()):
            logging.info(f"{year} -> Return: {metrics['Return']:.2f}%, "
                        f"MDD: {metrics['MDD']:.2f}%, "
                        f"Sharpe: {metrics['Sharpe']:.4f}, "
                        f"Sortino: {metrics['Sortino']:.4f}")
        logging.info('-' * 50)