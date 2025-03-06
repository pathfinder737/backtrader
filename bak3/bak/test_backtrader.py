import os
import logging
import datetime
import traceback
import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import pandas_market_calendars as mcal
from pandas.tseries.offsets import BMonthEnd, BusinessDay
import config

# ------------------------------
# 기본 설정 로드 및 폴더/로그 설정
# ------------------------------
try:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)
except PermissionError:
    print(f"폴더 생성 권한이 없습니다: {config.LOG_DIR} 또는 {config.DATA_DIR}")
    raise
except Exception as e:
    print(f"폴더 생성 중 오류 발생: {str(e)}")
    raise

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, mode='a', encoding='utf-8'),  # 'w'에서 'a'로 변경하여 로그 누적
        logging.StreamHandler()
    ]
)

DATA_MODE = config.DATA_MODE  # 'online' 또는 'offline'
logging.info(f"데이터 모드 설정: {DATA_MODE.upper()}")

# ------------------------------
# NYSE 캘린더 및 헬퍼 함수 정의
# ------------------------------
nyse = mcal.get_calendar('NYSE')

def get_last_business_day(date):
    """
    주어진 날짜가 속한 월의 마지막 영업일을 반환합니다.
    
    Args:
        date: 날짜 객체 또는 문자열
        
    Returns:
        마지막 영업일(date 객체)
    """
    ts = pd.Timestamp(date)
    last_day_of_month = ts + BMonthEnd(0)
    # 15일 범위로 늘려 긴 휴일 기간도 고려
    schedule = nyse.schedule(start_date=last_day_of_month - pd.Timedelta(days=15), end_date=last_day_of_month)
    return schedule.index[-1].date() if not schedule.empty else (ts + BMonthEnd(0)).date()

def get_first_business_day(date):
    """
    주어진 날짜가 속한 월의 첫 영업일을 반환합니다.
    
    Args:
        date: 날짜 객체 또는 문자열
        
    Returns:
        첫 영업일(date 객체)
    """
    ts = pd.Timestamp(date).replace(day=1)
    # 15일 범위로 늘려 긴 휴일 기간도 고려
    schedule = nyse.schedule(start_date=ts, end_date=ts + pd.Timedelta(days=15))
    return schedule.index[0].date() if not schedule.empty else (ts + BusinessDay(0)).date()

# ------------------------------
# Backtrader 데이터 피드 정의
# ------------------------------
class FinanceDataReaderData(bt.feeds.PandasData):
    params = (
        ('open',  'Open'),
        ('high',  'High'),
        ('low',   'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
    )

# ------------------------------
# 데이터 다운로드 및 전처리 (offline 모드용)
# ------------------------------
def download_and_preprocess(ticker, start_date, end_date, force_download=False):
    """
    티커 데이터를 다운로드하고 전처리합니다.
    
    Args:
        ticker: 티커 심볼
        start_date: 시작 날짜
        end_date: 종료 날짜
        force_download: 기존 파일이 있어도 강제 다운로드 여부
        
    Returns:
        전처리된 데이터프레임
    
    Raises:
        ValueError: 필수 컬럼이 누락된 경우
        ConnectionError: 데이터 다운로드 실패 시
    """
    file_path = os.path.join(config.DATA_DIR, f"{ticker}.csv")
    
    # 기존 파일 확인 (force_download가 False이고 파일이 존재하는 경우)
    if not force_download and os.path.exists(file_path):
        logging.info(f"{ticker} 기존 데이터 파일을 사용합니다: {file_path}")
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logging.info(f"{ticker} 데이터 로드 완료: {len(df)} 행")
            return df
        except Exception as e:
            logging.warning(f"{ticker} 기존 파일 로드 실패, 다시 다운로드합니다: {str(e)}")
    
    # 파일이 없거나 로드 실패 시 다운로드
    logging.info(f"{ticker} 데이터를 다운로드합니다: {start_date} ~ {end_date}")
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            df = fdr.DataReader(ticker, start_date, end_date)
            if df.empty:
                raise ValueError(f"{ticker}에 대한 데이터가 없습니다.")
                
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            # 필수 컬럼 확인
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"{ticker} 데이터에 필수 컬럼 {missing_columns} 누락됨.")
                
            # 결측치 처리 (Volume은 0으로, 나머지는 선형 보간)
            if df.isnull().values.any():
                logging.info(f"{ticker} 데이터에 결측치가 발견되었습니다.")
                # Volume 컬럼은 0으로 채우기
                if 'Volume' in df.columns and df['Volume'].isnull().any():
                    logging.info(f"{ticker} Volume 데이터의 결측치를 0으로 채웁니다.")
                    df['Volume'].fillna(0, inplace=True)
                
                # 나머지 컬럼은 선형 보간
                cols_to_interpolate = [col for col in df.columns if col != 'Volume']
                if any(df[col].isnull().any() for col in cols_to_interpolate):
                    logging.info(f"{ticker} 가격 데이터의 결측치를 선형 보간법으로 채웁니다.")
                    df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method='linear')
                
                # 여전히 결측치가 있는지 확인
                if df.isnull().values.any():
                    logging.warning(f"{ticker} 데이터에 여전히 결측치가 있습니다. 앞/뒤 값으로 채웁니다.")
                    df.fillna(method='ffill', inplace=True)
                    df.fillna(method='bfill', inplace=True)
            
            # 전처리 완료된 데이터 CSV 저장
            df.to_csv(file_path)
            logging.info(f"{ticker} 전처리 완료 데이터를 {file_path}에 저장했습니다. ({len(df)} 행)")
            return df
            
        except Exception as e:
            retry_count += 1
            logging.warning(f"{ticker} 데이터 다운로드 시도 {retry_count}/{max_retries} 실패: {str(e)}")
            if retry_count >= max_retries:
                logging.error(f"{ticker} 데이터 다운로드 최대 시도 횟수 초과.")
                raise ConnectionError(f"{ticker} 데이터를 가져올 수 없습니다: {str(e)}")

# ------------------------------
# 월말 리밸런싱 전략 클래스
# ------------------------------
class MonthlyRebalanceStrategy(bt.Strategy):
    params = (
        ('asset_allocation', config.ASSET_ALLOCATION),  # 기본 자산배분
        ('riskfreerate', config.RISK_FREE_RATE),        # 리스크 프리 레이트
        ('ma_period', config.MA_PERIOD),                # 이동평균 기간 (config에서 가져옴)
        ('use_market_regime', getattr(config, 'USE_MARKET_REGIME', False)),  # 시장 상태 활용 여부
        ('fractional_shares', getattr(config, 'FRACTIONAL_SHARES', False)),  # 소수점 주식 거래 허용 여부
    )

    def __init__(self):
        # 이동평균 계산은 config.TOUCHSTONE 기준으로 진행
        self.touch = self.getdatabyname(config.TOUCHSTONE)
        self.ma = bt.indicators.SMA(self.touch.close, period=self.p.ma_period, 
                                     plotname=f"MA_{self.p.ma_period}")
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
        현재 시장 상태에 따른 목표 자산 배분을 반환합니다.
        
        Args:
            current_date: 현재 날짜
            
        Returns:
            자산별 배분 비율 딕셔너리
        """
        # 시장 상태 판단 - SPY(TOUCHSTONE)의 가격과 이동평균 비교
        if self.touch.close[0] > self.ma[0]:
            self.market_regime = "Positive"
        else:
            self.market_regime = "Negative"
        
        logging.info(f"[{current_date}] {config.TOUCHSTONE} 가격: {self.touch.close[0]:.2f}, "
                     f"MA({self.p.ma_period}): {self.ma[0]:.2f} => 시장 상태: {self.market_regime}")
        
        # 시장 상태 활용 옵션이 켜진 경우에만 변형된 자산 배분 적용
        if self.p.use_market_regime:
            if self.market_regime == "Positive":
                # 시장이 긍정적일 때는 기본 자산 배분 사용
                logging.info("시장 상태 긍정적: 기본 자산 배분 사용")
                return self.asset_allocation
            else:
                # 시장이 부정적일 때는 위험자산 비중 절반으로 감소
                adjusted_allocation = {}
                total_reduced = 0.0
                
                # 모든 자산의 비중을 절반으로 감소
                for asset in self.asset_allocation:
                    original = self.asset_allocation[asset]
                    reduced = original * 0.5  # 50%로 감소
                    adjusted_allocation[asset] = reduced
                    total_reduced += (original - reduced)
                
                # 안전자산(예: IAU)에 감소된 비중 추가 (안전자산 설정에 따라 조정 필요)
                if 'IAU' in adjusted_allocation:
                    adjusted_allocation['IAU'] += total_reduced
                    
                logging.info(f"시장 상태 부정적: 조정된 자산 배분 사용 {adjusted_allocation}")
                return adjusted_allocation
        else:
            # 시장 상태 무시, 기본 자산 배분 사용
            return self.asset_allocation

    def next(self):
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
                    logging.error(traceback.format_exc())

        self.last_date = current_date

    def rebalance_portfolio(self, current_date, target_allocation):
        """
        목표 자산 배분에 따라 포트폴리오를 리밸런싱합니다.
        
        Args:
            current_date: 현재 날짜
            target_allocation: 목표 자산 배분 비율
        """
        logging.info(f"=== {current_date} 리밸런싱 시작 ===")
        total_value = self.broker.getvalue()
        target_values = {}
        target_shares = {}
        total_used_cash = 0.0

        # 목표 주식 수 계산
        for asset_name in self.assets:
            prev_close = self.prev_month_close_prices.get(asset_name, None)
            if prev_close is None:
                logging.warning(f"No recorded previous month close price for {asset_name}. Skipping rebalance for this asset.")
                continue
                
            allocation = target_allocation[asset_name]
            target_value = total_value * allocation
            target_values[asset_name] = target_value
            
            # 소수점 주식 거래 여부에 따라 주식 수 계산
            if self.p.fractional_shares:
                shares_to_buy = target_value / prev_close
            else:
                shares_to_buy = int(target_value / prev_close)
                
            target_shares[asset_name] = shares_to_buy
            used_cash = shares_to_buy * prev_close
            total_used_cash += used_cash
            
            logging.info(f"자산: {asset_name}, 목표 배분율: {allocation:.4f}, "
                        f"목표 금액: {target_value:.2f}, 목표 주식 수: {shares_to_buy:.4f}, "
                        f"사용 금액: {used_cash:.2f}")

        # 현재 포지션과 필요한 조정 계산
        current_shares = {}
        adjust_shares = {}
        
        for asset_name in self.assets:
            d = self.getdatabyname(asset_name)
            pos = self.getposition(d)
            current_shares[asset_name] = pos.size if pos else 0
            
            # 해당 자산에 대한 목표 주식 수가 없으면 건너뜀
            if asset_name not in target_shares:
                continue
                
            adjust_shares[asset_name] = target_shares[asset_name] - current_shares[asset_name]

        # 주문 실행
        for asset_name, shares in adjust_shares.items():
            if abs(shares) < 0.0001:  # 매우 작은 조정은 무시 (소수점 주식 거래 시)
                continue
                
            d = self.getdatabyname(asset_name)
            
            if shares > 0:
                self.buy(data=d, size=shares)
                logging.info(f"{current_date}: {asset_name} {shares:.4f}주 매수 주문")
            elif shares < 0:
                self.sell(data=d, size=abs(shares))
                logging.info(f"{current_date}: {asset_name} {abs(shares):.4f}주 매도 주문")

        # 잔여 현금 로그
        remaining_cash = total_value - total_used_cash
        logging.info(f"리밸런싱 완료: 총 포트폴리오 가치: {total_value:.2f}, "
                    f"사용된 현금: {total_used_cash:.2f}, 잔여 현금: {remaining_cash:.2f}")
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

# ------------------------------
# Cerebro 엔진 초기화 및 데이터 로딩
# ------------------------------
def run_backtest():
    """
    백테스트를 실행하고 결과를 시각화합니다.
    
    Returns:
        백테스트 결과
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MonthlyRebalanceStrategy)

    start_date = config.START_DATE
    end_date = config.END_DATE
    tickers = config.TICKERS
    touchstone_data_loaded = False  # TOUCHSTONE으로 변수명 변경

    # ASSET_ALLOCATION에 포함된 모든 티커가 TICKERS에 있는지 확인
    allocation_tickers = set(config.ASSET_ALLOCATION.keys())
    missing_tickers = allocation_tickers - set(tickers)
    if missing_tickers:
        logging.error(f"자산 배분에 포함되었지만 TICKERS에 없는 티커: {missing_tickers}")
        raise ValueError(f"자산 배분에 포함된 모든 티커는 TICKERS 목록에 있어야 합니다. 누락된 티커: {missing_tickers}")
    
    for ticker in tickers:
        logging.info(f"{ticker} 데이터 처리 시작...")
        try:
            if DATA_MODE == 'online':
                # 온라인 모드: 실시간 다운로드
                df = download_and_preprocess(ticker, start_date, end_date, force_download=True)
            else:
                # 오프라인 모드: 캐시 활용
                df = download_and_preprocess(ticker, start_date, end_date, force_download=False)
                
            # 필수 컬럼 확인 (다운로드 함수에서 이미 확인하지만 한번 더 검증)
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.warning(f"{ticker} 데이터에 필수 컬럼 {missing_columns} 누락됨. 티커 건너뜀.")
                if ticker == config.TOUCHSTONE:
                    raise ValueError(f"{config.TOUCHSTONE}는 필수 티커입니다. 누락된 컬럼: {missing_columns}")
                continue

            # 날짜 변환
            start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            
            data = FinanceDataReaderData(
                dataname=df,
                fromdate=start_dt,
                todate=end_dt,
                name=ticker
            )
            cerebro.adddata(data, name=ticker)
            logging.info(f"{ticker} 데이터 추가 완료")
            
            if ticker == config.TOUCHSTONE:
                touchstone_data_loaded = True

        except ValueError as ve:
            logging.error(f"{ticker} 데이터 처리 실패: {str(ve)}")
            if ticker == config.TOUCHSTONE:
                raise ValueError(f"{config.TOUCHSTONE}는 필수 티커입니다. 오류: {str(ve)}")
                
        except ConnectionError as ce:
            logging.error(f"{ticker} 데이터 다운로드 실패: {str(ce)}")
            if ticker == config.TOUCHSTONE:
                raise ConnectionError(f"{config.TOUCHSTONE} 데이터 다운로드 실패. 백테스팅 중단.")
                
        except Exception as e:
            logging.error(f"{ticker} 데이터 처리 중 예상치 못한 오류: {str(e)}")
            logging.error(traceback.format_exc())
            if ticker == config.TOUCHSTONE:
                raise RuntimeError(f"{config.TOUCHSTONE} 데이터 처리 실패. 오류: {str(e)}")

    if not touchstone_data_loaded:
        raise ValueError(f"{config.TOUCHSTONE} 데이터 로드 실패. 백테스팅 중단.")

    # 초기 설정
    cerebro.broker.setcash(config.INITIAL_CASH)
    cerebro.broker.setcommission(commission=config.COMMISSION)

    # 분석 지표 추가
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=config.RISK_FREE_RATE)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

    # 백테스팅 실행
    initial_value = cerebro.broker.getvalue()
    logging.info('Initial portfolio value: {:,.2f}'.format(initial_value))
    
    try:
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        logging.info('Final portfolio value: {:,.2f}'.format(final_value))
        total_return = ((final_value / config.INITIAL_CASH) - 1) * 100
        logging.info(f"Total return: {total_return:.2f}%")
        
        # 결과 시각화
        visualize_results(results, initial_value, final_value)
        
        return results
        
    except Exception as e:
        logging.error(f"백테스트 실행 중 오류 발생: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def visualize_results(results, initial_value, final_value):
    """
    백테스트 결과를 시각화합니다.
    
    Args:
        results: 백테스트 결과
        initial_value: 초기 포트폴리오 가치
        final_value: 최종 포트폴리오 가치
    """
    if not results or len(results) == 0:
        logging.warning("시각화를 위한 백테스트 결과가 없습니다.")
        return
        
    strategy = results[0]  # 첫 번째 전략 인스턴스
    
    if not hasattr(strategy, 'portfolio_dates') or not strategy.portfolio_dates:
        logging.warning("포트폴리오 가치 데이터가 없어 그래프를 생성할 수 없습니다.")
        return
    
    # 타임스탬프를 파일명에 추가
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 포트폴리오 가치 그래프
    dates = strategy.portfolio_dates
    values = strategy.portfolio_values

    plt.figure(figsize=(12, 8))
    plt.plot(dates, values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    
    # x축 눈금 설정 (10개 내외로 제한)
    n = len(dates)
    step = max(1, n // 10)
    plt.xticks(dates[::step], [d.strftime('%Y-%m') for d in dates[::step]], rotation=45)
    
    # 그래프에 초기 및 최종 가치 표시
    plt.figtext(0.13, 0.02, f'Initial Value: ${initial_value:,.2f}', fontsize=10)
    plt.figtext(0.7, 0.02, f'Final Value: ${final_value:,.2f}', fontsize=10)
    plt.figtext(0.4, 0.02, f'Return: {((final_value/initial_value)-1)*100:.2f}%', fontsize=10)
    
    plt.tight_layout()
    
    # 결과 저장 (타임스탬프 포함)
    file_path = f'backtest_result_{timestamp}.png'
    plt.savefig(file_path)
    logging.info(f"포트폴리오 가치 그래프를 '{file_path}'에 저장했습니다.")
    
    # 그래프 표시 (필요 시 주석 처리)
    plt.show()
    
    # 추가 그래프 - 연간 수익률
    annual_metrics = strategy.compute_annual_metrics()
    if annual_metrics:
        years = sorted(annual_metrics.keys())
        returns = [annual_metrics[year]['Return'] for year in years]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(years, returns)
        
        # 막대 색상 설정 (양수는 파란색, 음수는 빨간색)
        for i, bar in enumerate(bars):
            bar.set_color('blue' if returns[i] >= 0 else 'red')
            
        plt.title('Annual Returns')
        plt.xlabel('Year')
        plt.ylabel('Return (%)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 각 막대 위에 수치 표시
        for i, v in enumerate(returns):
            plt.text(i, v + (5 if v >= 0 else -5), 
                     f'{v:.1f}%', 
                     ha='center', 
                     va='bottom' if v >= 0 else 'top',
                     fontsize=9)
        
        plt.tight_layout()
        annual_file_path = f'annual_returns_{timestamp}.png'
        plt.savefig(annual_file_path)
        logging.info(f"연간 수익률 그래프를 '{annual_file_path}'에 저장했습니다.")
        plt.show()

# ------------------------------
# 메인 실행 코드
# ------------------------------
if __name__ == "__main__":
    try:
        logging.info("백테스트 시작")
        results = run_backtest()
        logging.info("백테스트 완료")
    except Exception as e:
        logging.error(f"백테스트 실행 실패: {str(e)}")
        logging.error(traceback.format_exc())