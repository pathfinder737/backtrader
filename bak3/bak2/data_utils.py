# data_utils.py
# 날짜 유틸리티 및 데이터 처리 관련 함수들을 정의합니다.

import os
import logging
import datetime
import pandas as pd
import numpy as np
import backtrader as bt
import FinanceDataReader as fdr
import pandas_market_calendars as mcal
from pandas.tseries.offsets import BMonthEnd, BusinessDay, Week
from typing import Dict, List, Optional, Tuple, Union

import config

# ------------------------------
# NYSE 캘린더 초기화
# ------------------------------
try:
    nyse = mcal.get_calendar('NYSE')
except Exception as e:
    logging.warning(f"NYSE 캘린더 초기화 실패: {str(e)}. 기본 비즈니스 캘린더를 사용합니다.")
    nyse = None


# ------------------------------
# 날짜 유틸리티 함수
# ------------------------------
def get_last_business_day(date_obj: Union[str, datetime.date, datetime.datetime]) -> datetime.date:
    """
    주어진 날짜가 속한 월의 마지막 영업일을 반환합니다.
    
    Args:
        date_obj: 날짜 객체 또는 문자열
        
    Returns:
        마지막 영업일(date 객체)
    """
    try:
        ts = pd.Timestamp(date_obj)
        last_day_of_month = ts + BMonthEnd(0)
        
        if nyse:
            # 15일 범위로 늘려 긴 휴일 기간도 고려
            schedule = nyse.schedule(
                start_date=last_day_of_month - pd.Timedelta(days=15), 
                end_date=last_day_of_month
            )
            
            if not schedule.empty:
                return schedule.index[-1].date()
        
        # 캘린더가 없거나 스케줄이 비어있으면 기본 로직 사용
        return (ts + BMonthEnd(0)).date()
    
    except Exception as e:
        logging.error(f"마지막 영업일 계산 중 오류: {str(e)}")
        # 에러 발생 시 기본 BMonthEnd 사용
        ts = pd.Timestamp(date_obj)
        return (ts + BMonthEnd(0)).date()


def get_first_business_day(date_obj: Union[str, datetime.date, datetime.datetime]) -> datetime.date:
    """
    주어진 날짜가 속한 월의 첫 영업일을 반환합니다.
    
    Args:
        date_obj: 날짜 객체 또는 문자열
        
    Returns:
        첫 영업일(date 객체)
    """
    try:
        ts = pd.Timestamp(date_obj).replace(day=1)
        
        if nyse:
            # 15일 범위로 늘려 긴 휴일 기간도 고려
            schedule = nyse.schedule(
                start_date=ts, 
                end_date=ts + pd.Timedelta(days=15)
            )
            
            if not schedule.empty:
                return schedule.index[0].date()
        
        # 캘린더가 없거나 스케줄이 비어있으면 기본 로직 사용
        return (ts + BusinessDay(0)).date()
    
    except Exception as e:
        logging.error(f"첫 영업일 계산 중 오류: {str(e)}")
        # 에러 발생 시 기본 BusinessDay 사용
        ts = pd.Timestamp(date_obj).replace(day=1)
        return (ts + BusinessDay(0)).date()


def is_last_business_day_of_month(date_obj: Union[str, datetime.date, datetime.datetime]) -> bool:
    """
    주어진 날짜가 해당 월의 마지막 영업일인지 확인합니다.
    
    Args:
        date_obj: 날짜 객체 또는 문자열
        
    Returns:
        마지막 영업일이면 True, 아니면 False
    """
    date_to_check = pd.Timestamp(date_obj).date()
    last_business_day = get_last_business_day(date_obj)
    return date_to_check == last_business_day


def is_first_business_day_of_month(date_obj: Union[str, datetime.date, datetime.datetime]) -> bool:
    """
    주어진 날짜가 해당 월의 첫 영업일인지 확인합니다.
    
    Args:
        date_obj: 날짜 객체 또는 문자열
        
    Returns:
        첫 영업일이면 True, 아니면 False
    """
    date_to_check = pd.Timestamp(date_obj).date()
    first_business_day = get_first_business_day(date_obj)
    return date_to_check == first_business_day


# ------------------------------
# Backtrader 데이터 피드 정의
# ------------------------------
class FinanceDataReaderData(bt.feeds.PandasData):
    """
    FinanceDataReader에서 가져온 데이터를 Backtrader에서 사용할 수 있는 형식으로 변환하는 클래스입니다.
    """
    params = (
        ('open',  'Open'),
        ('high',  'High'),
        ('low',   'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
    )


# ------------------------------
# 데이터 다운로드 및 전처리 함수
# ------------------------------
def download_and_preprocess(ticker: str, 
                          start_date: str, 
                          end_date: str, 
                          force_download: bool = False,
                          data_dir: Optional[str] = None,
                          log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    티커 데이터를 다운로드하고 전처리합니다.
    
    Args:
        ticker: 티커 심볼
        start_date: 시작 날짜
        end_date: 종료 날짜
        force_download: 기존 파일이 있어도 강제 다운로드 여부
        data_dir: 데이터 저장 디렉토리 (기본값: config.DATA_DIR)
        log: 로깅에 사용할 로거
        
    Returns:
        전처리된 데이터프레임
    
    Raises:
        ValueError: 필수 컬럼이 누락된 경우
        ConnectionError: 데이터 다운로드 실패 시
    """
    # 로거 설정
    logger = log or logging.getLogger('backtest')
    
    # 데이터 디렉토리 설정
    directory = data_dir or config.DATA_DIR
    os.makedirs(directory, exist_ok=True)
    
    file_path = os.path.join(directory, f"{ticker}.csv")
    
    # 기존 파일 확인 (force_download가 False이고 파일이 존재하는 경우)
    if not force_download and os.path.exists(file_path):
        logger.info(f"{ticker} 기존 데이터 파일을 사용합니다: {file_path}")
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"{ticker} 데이터 로드 완료: {len(df)} 행")
            
            # 필수 컬럼 확인
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"{ticker} 기존 파일에 필수 컬럼 {missing_columns}가 누락되었습니다. 다시 다운로드합니다.")
                force_download = True
            else:
                return df
        except Exception as e:
            logger.warning(f"{ticker} 기존 파일 로드 실패, 다시 다운로드합니다: {str(e)}")
            force_download = True
    
    # 파일이 없거나 로드 실패 또는 강제 다운로드 시 다운로드 진행
    if force_download or not os.path.exists(file_path):
        logger.info(f"{ticker} 데이터를 다운로드합니다: {start_date} ~ {end_date}")
        
        # 재시도 로직
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 지수 백오프를 위한 대기 시간 계산
                wait_time = 2 ** retry_count if retry_count > 0 else 0
                
                # 재시도 시 대기
                if wait_time > 0:
                    logger.info(f"{ticker} 데이터 다운로드 재시도 ({retry_count}/{max_retries}) - {wait_time}초 후 시도")
                    import time
                    time.sleep(wait_time)
                
                # 데이터 다운로드
                df = fdr.DataReader(ticker, start_date, end_date)
                
                if df.empty:
                    raise ValueError(f"{ticker}에 대한 데이터가 없습니다.")
                
                # 인덱스가 DatetimeIndex가 아니면 변환
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # 필수 컬럼 확인
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"{ticker} 데이터에 필수 컬럼 {missing_columns} 누락됨.")
                
                # 결측치 처리
                if df.isnull().values.any():
                    logger.info(f"{ticker} 데이터에 결측치가 발견되었습니다.")
                    
                    # Volume 컬럼은 0으로 채우기
                    if 'Volume' in df.columns and df['Volume'].isnull().any():
                        logger.info(f"{ticker} Volume 데이터의 결측치를 0으로 채웁니다.")
                        df['Volume'].fillna(0, inplace=True)
                    
                    # 나머지 컬럼은 선형 보간
                    cols_to_interpolate = [col for col in df.columns if col != 'Volume']
                    if any(df[col].isnull().any() for col in cols_to_interpolate):
                        logger.info(f"{ticker} 가격 데이터의 결측치를 선형 보간법으로 채웁니다.")
                        df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method='linear')
                    
                    # 여전히 결측치가 있는지 확인
                    if df.isnull().values.any():
                        logger.warning(f"{ticker} 데이터에 여전히 결측치가 있습니다. 앞/뒤 값으로 채웁니다.")
                        df.fillna(method='ffill', inplace=True)
                        df.fillna(method='bfill', inplace=True)
                
                # 일간 이동평균 계산 추가
                ma_periods = config.get_config('MA_PERIODS', [21, 50, 150, 200])
                for period in ma_periods:
                    df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
                
                # 주간 데이터 및 이동평균 계산 추가
                weekly_df = calculate_weekly_data(df)
                
                # 주간 데이터를 일간 데이터에 매핑
                wma_periods = config.get_config('WMA_PERIODS', [10, 30])
                for period in wma_periods:
                    df = map_weekly_ma_to_daily(df, weekly_df, period)
                
                # 전처리 완료된 데이터 CSV 저장
                df.to_csv(file_path)
                logger.info(f"{ticker} 전처리 완료 데이터를 {file_path}에 저장했습니다. ({len(df)} 행)")
                return df
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"{ticker} 데이터 다운로드 시도 {retry_count}/{max_retries} 실패: {str(e)}")
                
                if retry_count >= max_retries:
                    logger.error(f"{ticker} 데이터 다운로드 최대 시도 횟수 초과.")
                    raise ConnectionError(f"{ticker} 데이터를 가져올 수 없습니다: {str(e)}")
    
    # 이 부분은 실행되지 않아야 함 (모든 경우가 위에서 처리됨)
    raise RuntimeError(f"{ticker} 데이터 처리 중 예상치 못한 오류가 발생했습니다.")


def calculate_weekly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    일간 데이터를 기반으로 주간 데이터를 계산합니다.
    
    Args:
        df: 일간 데이터 DataFrame
        
    Returns:
        주간 데이터 DataFrame
    """
    # 주 단위로 리샘플링 (마지막 거래일 기준)
    weekly_df = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    # 결측치 제거
    weekly_df = weekly_df.dropna()
    
    # 주간 이동평균 계산
    wma_periods = config.get_config('WMA_PERIODS', [10, 30])
    for period in wma_periods:
        weekly_df[f'WMA_{period}'] = weekly_df['Close'].rolling(window=period).mean()
    
    return weekly_df


def map_weekly_ma_to_daily(daily_df: pd.DataFrame, weekly_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    주간 이동평균을 일간 데이터에 매핑합니다.
    
    Args:
        daily_df: 일간 데이터 DataFrame
        weekly_df: 주간 데이터 DataFrame
        period: 이동평균 기간
        
    Returns:
        주간 이동평균이 추가된 일간 데이터 DataFrame
    """
    # 주간 데이터의 이동평균값 가져오기
    wma_values = weekly_df[f'WMA_{period}']
    
    # 각 일간 데이터에 대해 가장 가까운 주간 데이터의 값 찾기
    daily_df[f'WMA_{period}'] = None
    
    for date in daily_df.index:
        # 가장 최근의 주간 데이터 찾기 (해당 날짜 이전 또는 같은 날짜)
        try:
            week_end = date + pd.Timedelta(days=(6 - date.dayofweek))  # 해당 주의 일요일
            if week_end in weekly_df.index:
                daily_df.at[date, f'WMA_{period}'] = weekly_df.at[week_end, f'WMA_{period}']
            else:
                # 가장 가까운 이전 주의 데이터 찾기
                prev_weeks = weekly_df.index[weekly_df.index <= date]
                if len(prev_weeks) > 0:
                    closest_week = prev_weeks[-1]
                    daily_df.at[date, f'WMA_{period}'] = weekly_df.at[closest_week, f'WMA_{period}']
        except Exception:
            # 매핑 오류 시 NaN으로 유지
            pass
    
    # 매핑 후 남은 NaN 값들은 전방 채우기로 처리
    daily_df[f'WMA_{period}'] = daily_df[f'WMA_{period}'].fillna(method='ffill')
    
    return daily_df


def prepare_backtrader_data(tickers: List[str], 
                          start_date: str, 
                          end_date: str, 
                          data_mode: str = 'offline',
                          log: Optional[logging.Logger] = None) -> Dict[str, FinanceDataReaderData]:
    """
    여러 티커의 데이터를 로드하고 Backtrader 데이터 피드로 변환합니다.
    
    Args:
        tickers: 티커 목록
        start_date: 시작 날짜
        end_date: 종료 날짜
        data_mode: 'online'이면 강제 다운로드, 'offline'이면 캐시 사용
        log: 로깅에 사용할 로거
        
    Returns:
        티커별 Backtrader 데이터 피드 딕셔너리
    
    Raises:
        ValueError: 필수 티커 데이터 로드 실패 시
    """
    logger = log or logging.getLogger('backtest')
    force_download = (data_mode == 'online')
    touchstone = config.get_config('TOUCHSTONE')
    bt_data_feeds = {}
    
    try:
        # 날짜 형식 검사 및 변환
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        logger.error(f"날짜 형식 오류: {e}")
        raise ValueError(f"날짜 형식이 올바르지 않습니다. 'YYYY-MM-DD' 형식을 사용하세요.")
    
    # 각 티커 데이터 로드 및 변환
    for ticker in tickers:
        try:
            logger.info(f"{ticker} 데이터 로드 시작...")
            
            # 데이터 다운로드 및 전처리
            df = download_and_preprocess(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                force_download=force_download,
                log=logger
            )
            
            # Backtrader 데이터 피드 생성
            data_feed = FinanceDataReaderData(
                dataname=df,
                fromdate=start_dt,
                todate=end_dt,
                name=ticker
            )
            
            bt_data_feeds[ticker] = data_feed
            logger.info(f"{ticker} Backtrader 데이터 피드 생성 완료")
            
        except Exception as e:
            logger.error(f"{ticker} 데이터 로드 및 변환 실패: {str(e)}")
            
            # 기준 티커 실패 시 치명적 에러
            if ticker == touchstone:
                raise ValueError(f"기준 티커({touchstone}) 데이터 로드 및 변환 실패: {str(e)}")
    
    # 기준 티커가 로드되었는지 확인
    if touchstone not in bt_data_feeds:
        raise ValueError(f"기준 티커({touchstone}) 데이터가 로드되지 않았습니다. 백테스팅을 진행할 수 없습니다.")
    
    logger.info(f"총 {len(bt_data_feeds)}개 티커의 데이터 로드 및 변환 완료")
    return bt_data_feeds


def analyze_market_regime(df: pd.DataFrame, 
                        current_date: Union[str, datetime.date, datetime.datetime],
                        ma_type: str = 'daily',
                        ma_periods: List[int] = None,
                        wma_periods: List[int] = None,
                        log: Optional[logging.Logger] = None) -> Tuple[str, Dict[str, float]]:
    """
    특정 날짜의 시장 상태(레짐)를 분석합니다.
    
    Args:
        df: 기준 티커의 데이터프레임
        current_date: 분석할 날짜
        ma_type: 이동평균 타입 ('daily' 또는 'weekly')
        ma_periods: 이동평균 기간 목록 (기본값: config.MA_PERIODS)
        wma_periods: 주간 이동평균 기간 목록 (기본값: config.WMA_PERIODS)
        log: 로깅에 사용할 로거
        
    Returns:
        (시장 상태, 자산 배분 딕셔너리)
    """
    logger = log or logging.getLogger('backtest')
    ma_periods = ma_periods or config.get_config('MA_PERIODS')
    wma_periods = wma_periods or config.get_config('WMA_PERIODS')
    
    # 현재 날짜에 해당하는 데이터 가져오기
    current_date = pd.Timestamp(current_date)
    try:
        # 정확한 날짜가 없으면 이전 날짜 중 가장 가까운 날짜 사용
        if current_date not in df.index:
            available_dates = df.index[df.index <= current_date]
            if len(available_dates) == 0:
                raise ValueError(f"분석 가능한 날짜({current_date} 이전)가 없습니다.")
            current_date = available_dates[-1]
        
        row = df.loc[current_date]
    except Exception as e:
        logger.error(f"날짜({current_date}) 데이터 가져오기 실패: {str(e)}")
        raise ValueError(f"날짜({current_date}) 데이터를 찾을 수 없습니다.")
    
    # 현재 가격 가져오기
    current_price = row['Close']
    
    # 이동평균 타입에 따라 시장 상태 분석
    if ma_type == 'weekly':
        return analyze_weekly_market_regime(df, current_date, current_price, wma_periods, logger)
    else:
        return analyze_daily_market_regime(df, current_date, current_price, ma_periods, logger)


def analyze_daily_market_regime(df: pd.DataFrame, 
                               current_date: pd.Timestamp, 
                               current_price: float, 
                               ma_periods: List[int],
                               logger: logging.Logger) -> Tuple[str, Dict[str, float]]:
    """
    일간 이동평균을 기반으로 시장 상태를 분석합니다.
    
    Args:
        df: 데이터프레임
        current_date: 현재 날짜
        current_price: 현재 가격
        ma_periods: 이동평균 기간 목록
        logger: 로거
        
    Returns:
        (시장 상태, 자산 배분 딕셔너리)
    """
    # 이동평균 가져오기
    row = df.loc[current_date]
    ma_values = {}
    
    for period in ma_periods:
        ma_col = f'MA_{period}'
        if ma_col in row and not pd.isna(row[ma_col]):
            ma_values[period] = row[ma_col]
        else:
            # 이동평균 컬럼이 없으면 직접 계산
            logger.warning(f"{ma_col} 컬럼이 없습니다. 직접 계산합니다.")
            ma_values[period] = df['Close'].rolling(window=period).mean().loc[current_date]
    
    # 시장 상태 로깅
    logger.info(f"[{current_date}] 가격: {current_price:.2f}, 일간 MA 값: {ma_values}")
    
    # 이동평균 관계 분석
    ma_short = ma_values.get(ma_periods[0], 0)  # MA21
    ma_mid = ma_values.get(ma_periods[1], 0)    # MA50
    ma_mid2 = ma_values.get(ma_periods[2], 0)   # MA150
    ma_long = ma_values.get(ma_periods[3], 0)   # MA200
    
    # 시장 상태 결정
    # 1. MA21 > MA50 > MA200 인 경우 - 공격적 배분 (Aggressive)
    if current_price > ma_short and ma_short > ma_mid and ma_mid > ma_long:
        regime = "Aggressive"
        allocation = config.get_config('AGGRESSIVE_ALLOCATION')
        logger.info(f"시장 상태: {regime} (MA21 > MA50 > MA200)")
    
    # 2. MA50 > MA200 인 경우 - 중립적 배분 (Moderate)
    elif current_price > ma_mid and ma_mid > ma_long:
        regime = "Moderate"
        allocation = config.get_config('MODERATE_ALLOCATION')
        logger.info(f"시장 상태: {regime} (MA50 > MA200)")
    
    # 3. 현재가격 < MA150 이고 > MA200인 경우 - 중간 방어적 배분 (Mid Defensive)
    elif current_price < ma_mid2 and current_price > ma_long:
        regime = "MidDefensive"
        allocation = config.get_config('MID_DEFENSIVE_ALLOCATION')
        logger.info(f"시장 상태: {regime} (MA150 > 가격 > MA200)")
    
    # 4. MA50 < MA200 인 경우 - 방어적 배분 (Defensive)
    elif current_price < ma_long:
        regime = "Defensive"
        allocation = config.get_config('DEFENSIVE_ALLOCATION')
        logger.info(f"시장 상태: {regime} (가격 < MA200)")
    
    # 5. 그 외 경우 - 기본 배분 (Neutral)
    else:
        regime = "Neutral"
        allocation = config.get_config('ASSET_ALLOCATION')
        logger.info(f"시장 상태: {regime} (기타 조건)")
    
    logger.info(f"자산 배분: {allocation}")
    return regime, allocation


def analyze_weekly_market_regime(df: pd.DataFrame, 
                                current_date: pd.Timestamp, 
                                current_price: float, 
                                wma_periods: List[int],
                                logger: logging.Logger) -> Tuple[str, Dict[str, float]]:
    """
    주간 이동평균을 기반으로 시장 상태를 분석합니다.
    
    Args:
        df: 데이터프레임
        current_date: 현재 날짜
        current_price: 현재 가격
        wma_periods: 주간 이동평균 기간 목록
        logger: 로거
        
    Returns:
        (시장 상태, 자산 배분 딕셔너리)
    """
    # 주간 이동평균 가져오기
    row = df.loc[current_date]
    wma_values = {}
    
    for period in wma_periods:
        wma_col = f'WMA_{period}'
        if wma_col in row and not pd.isna(row[wma_col]):
            wma_values[period] = row[wma_col]
        else:
            logger.warning(f"{wma_col} 컬럼이 없습니다. 분석을 진행할 수 없습니다.")
            # 주간 이동평균이 없으면 기본 자산 배분 사용
            return "Neutral", config.get_config('ASSET_ALLOCATION')
    
    # 시장 상태 로깅
    logger.info(f"[{current_date}] 가격: {current_price:.2f}, 주간 MA 값: {wma_values}")
    
    # 주간 이동평균 관계 분석
    wma_short = wma_values.get(wma_periods[0], 0)  # WMA10
    wma_long = wma_values.get(wma_periods[1], 0)   # WMA30
    
    # 시장 상태 결정 (주간 이동평균 기준)
    # 1. CURRENT_PRICE > WMA10 AND WMA10 > WMA30: AGGRESSIVE
    if current_price > wma_short and wma_short > wma_long:
        regime = "Aggressive"
        allocation = config.get_config('AGGRESSIVE_ALLOCATION')
        logger.info(f"시장 상태: {regime} (가격 > WMA10 > WMA30)")
    
    # 2. CURRENT_PRICE < WMA10 AND CURRENT_PRICE > WMA30: MODERATE
    elif current_price < wma_short and current_price > wma_long:
        regime = "Moderate"
        allocation = config.get_config('MODERATE_ALLOCATION')
        logger.info(f"시장 상태: {regime} (WMA10 > 가격 > WMA30)")
    
    # 3. CURRENT_PRICE < WMA30: DEFENSIVE
    elif current_price < wma_long:
        regime = "Defensive"
        allocation = config.get_config('DEFENSIVE_ALLOCATION')
        logger.info(f"시장 상태: {regime} (가격 < WMA30)")
    
    # 4. 그 외 경우 - 기본 배분 (Neutral)
    else:
        regime = "Neutral"
        allocation = config.get_config('ASSET_ALLOCATION')
        logger.info(f"시장 상태: {regime} (기타 조건)")
    
    logger.info(f"자산 배분: {allocation}")
    return regime, allocation