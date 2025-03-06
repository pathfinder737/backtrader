#data_utils.py
import os
import logging
import datetime
import pandas as pd
import numpy as np
import backtrader as bt
import FinanceDataReader as fdr
import pandas_market_calendars as mcal
from pandas.tseries.offsets import BMonthEnd, BusinessDay
from typing import Dict, List, Optional, Tuple, Union
import config
import traceback

# 로거 초기화
logger = logging.getLogger('data_utils')

# NYSE 캘린더 초기화 (영업일 계산에 사용)
try:
    nyse = mcal.get_calendar('NYSE')
except Exception as e:
    logger.warning(f"Failed to initialize NYSE calendar: {e}. Using default business day.")
    nyse = None

def get_last_business_day(date_obj: Union[str, datetime.date, datetime.datetime]) -> datetime.date:
    """
    주어진 날짜의 달 마지막 영업일 계산.
    
    :param date_obj: 날짜 (문자열, date 또는 datetime)
    :return: 마지막 영업일 (date)
    """
    try:
        ts = pd.Timestamp(date_obj)
        last_day = ts + BMonthEnd(0)
        if nyse:
            schedule = nyse.schedule(start_date=last_day - pd.Timedelta(days=15), end_date=last_day)
            if not schedule.empty:
                return schedule.index[-1].date()
        return last_day.date()
    except Exception as e:
        logger.error(f"Error calculating last business day: {e}")
        return (pd.Timestamp(date_obj) + BMonthEnd(0)).date()

def get_first_business_day(date_obj: Union[str, datetime.date, datetime.datetime]) -> datetime.date:
    """
    주어진 날짜의 달 첫 번째 영업일 계산.
    
    :param date_obj: 날짜 (문자열, date 또는 datetime)
    :return: 첫 번째 영업일 (date)
    """
    try:
        ts = pd.Timestamp(date_obj).replace(day=1)
        if nyse:
            schedule = nyse.schedule(start_date=ts, end_date=ts + pd.Timedelta(days=15))
            if not schedule.empty:
                return schedule.index[0].date()
        return (ts + BusinessDay(0)).date()
    except Exception as e:
        logger.error(f"Error calculating first business day: {e}")
        return (pd.Timestamp(date_obj).replace(day=1) + BusinessDay(0)).date()

def is_last_business_day_of_month(date_obj: Union[str, datetime.date, datetime.datetime]) -> bool:
    """
    주어진 날짜가 달의 마지막 영업일인지 확인.
    
    :param date_obj: 날짜
    :return: True이면 마지막 영업일, 아니면 False
    """
    return pd.Timestamp(date_obj).date() == get_last_business_day(date_obj)

def is_first_business_day_of_month(date_obj: Union[str, datetime.date, datetime.datetime]) -> bool:
    """
    주어진 날짜가 달의 첫 번째 영업일인지 확인.
    
    :param date_obj: 날짜
    :return: True이면 첫 영업일, 아니면 False
    """
    return pd.Timestamp(date_obj).date() == get_first_business_day(date_obj)

# FinanceDataReaderData 클래스: Backtrader 데이터 피드 형식으로 변환
# 여기에는 'WMA_10'과 'WMA_30' 컬럼도 포함합니다.
class FinanceDataReaderData(bt.feeds.PandasData):
    params = (
        ('open',  'Open'),
        ('high',  'High'),
        ('low',   'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('wma_10', 'WMA_10'),
        ('wma_30', 'WMA_30'),
    )

def download_and_preprocess(ticker: str, start_date: str, end_date: str,
                            force_download: bool = False,
                            data_dir: Optional[str] = None,
                            log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    지정된 티커의 데이터를 다운로드 및 전처리하여 CSV로 저장 후 DataFrame 반환.
    **오프라인 모드에서도 항상 새롭게 다운로드합니다.**
    
    :param ticker: 티커 심볼
    :param start_date: 시작 날짜 ('YYYY-MM-DD')
    :param end_date: 종료 날짜 ('YYYY-MM-DD')
    :param force_download: 강제 다운로드 (오프라인 모드에서도 True)
    :param data_dir: 저장 디렉토리 (없으면 config의 DATA_DIR 사용)
    :param log: 로깅에 사용할 로거 (없으면 기본 logger 사용)
    :return: 전처리된 DataFrame
    """
    logger_local = log or logger
    directory = data_dir or config.config.DATA_DIR
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{ticker}.csv")
    
    # 캐시 사용 없이 항상 새롭게 다운로드
    force_download = True

    logger_local.info(f"Downloading data for {ticker} from {start_date} to {end_date}.")
    max_retries = 3
    retry_count = 0
    import time
    while retry_count < max_retries:
        try:
            if retry_count > 0:
                wait_time = 2 ** retry_count
                logger_local.info(f"Retry {retry_count}/{max_retries} for {ticker} after {wait_time}s.")
                time.sleep(wait_time)
            df = fdr.DataReader(ticker, start_date, end_date)
            if df.empty:
                raise ValueError(f"No data for {ticker}.")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            # 인덱스 이름을 "Date"로 설정하여 merge_asof에서 사용
            df.index.name = "Date"
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns {missing} for {ticker}.")
            if df.isnull().values.any():
                logger_local.info(f"Found missing values in {ticker} data.")
                df['Volume'].fillna(0, inplace=True)
                df.interpolate(method='linear', inplace=True)
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
            # 일간 이동평균 계산
            for period in config.config.get("MA_PERIODS", [21, 50, 150, 200]):
                df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
            weekly_df = calculate_weekly_data(df)
            for period in config.config.get("WMA_PERIODS", [10, 30]):
                df = map_weekly_ma_to_daily(df, weekly_df, period)
            df.to_csv(file_path)
            logger_local.info(f"Saved preprocessed data for {ticker} to {file_path} ({len(df)} rows).")
            return df
        except Exception as e:
            retry_count += 1
            logger_local.warning(f"Attempt {retry_count}/{max_retries} failed for {ticker}: {e}")
            if retry_count >= max_retries:
                logger_local.error(f"Max retries exceeded for {ticker}.")
                raise ConnectionError(f"Cannot retrieve data for {ticker}: {e}")
    raise RuntimeError("Unexpected error in download_and_preprocess.")

def calculate_weekly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    일간 데이터를 기반으로 주간 데이터를 계산합니다.
    
    :param df: 일간 DataFrame
    :return: 주간 DataFrame (각 주 마지막 거래일 기준)
    """
    weekly_df = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    for period in config.config.get("WMA_PERIODS", [10, 30]):
        weekly_df[f'WMA_{period}'] = weekly_df['Close'].rolling(window=period).mean()
    return weekly_df

def map_weekly_ma_to_daily(daily_df: pd.DataFrame, weekly_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    주간 이동평균 값을 일간 데이터에 매핑합니다.
    pandas의 merge_asof를 사용하며, 양쪽 DataFrame의 인덱스 이름을 "Date"로 강제 설정합니다.
    
    :param daily_df: 일간 DataFrame
    :param weekly_df: 주간 DataFrame
    :param period: 이동평균 기간 (예: 10 또는 30)
    :return: 일간 DataFrame에 주간 이동평균 컬럼 추가
    """
    # 인덱스 이름이 없으면 "Date"로 설정
    if daily_df.index.name is None:
        daily_df.index.name = "Date"
    if weekly_df.index.name is None:
        weekly_df.index.name = "Date"
    daily_df_sorted = daily_df.sort_index().reset_index()
    weekly_df_sorted = weekly_df.sort_index().reset_index()
    # weekly_df_sorted에서 해당 WMA 컬럼만 추출
    temp = weekly_df_sorted[['Date', f'WMA_{period}']]
    # merge_asof: daily_df_sorted와 temp를 'Date' 컬럼 기준으로 병합 (backward 방식)
    daily_df_merged = pd.merge_asof(daily_df_sorted, temp, on='Date', direction='backward')
    daily_df_merged.set_index('Date', inplace=True)
    return daily_df_merged

def prepare_backtrader_data(tickers: List[str], start_date: str, end_date: str,
                           data_mode: str = 'offline',
                           log: Optional[logging.Logger] = None) -> Dict[str, FinanceDataReaderData]:
    """
    여러 티커의 데이터를 강제 다운로드하여 Backtrader 데이터 피드로 변환합니다.
    
    :param tickers: 티커 목록 (리스트)
    :param start_date: 시작 날짜 ('YYYY-MM-DD')
    :param end_date: 종료 날짜 ('YYYY-MM-DD')
    :param data_mode: 'online'이면 강제 다운로드, offline에서도 강제 다운로드
    :param log: 로깅에 사용할 로거
    :return: 각 티커별 FinanceDataReaderData 객체 딕셔너리
    """
    logger_local = log or logger
    force_download = True  # 항상 새롭게 다운로드
    touchstone = config.config.get("TOUCHSTONE")
    bt_data_feeds = {}
    
    try:
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        logger_local.error(f"Invalid date format: {e}")
        raise ValueError("Dates must be in 'YYYY-MM-DD' format.")
    
    # touchstone 티커의 주간 데이터 파일 생성 (주간 이동평균 사용 시)
    if config.config.get("MA_TYPE") == 'weekly' and touchstone in tickers:
        try:
            logger_local.info(f"Creating weekly data file for touchstone ticker {touchstone}...")
            weekly_file = create_weekly_data_file(
                ticker=touchstone, 
                start_date=start_date, 
                end_date=end_date,
                data_dir=config.config.DATA_DIR,
                log=logger_local
            )
            logger_local.info(f"Weekly data file created: {weekly_file}")
        except Exception as e:
            logger_local.error(f"Failed to create weekly data file for {touchstone}: {e}")
            logger_local.error(traceback.format_exc())
            # 주간 파일 생성에 실패해도 백테스트는 계속 진행
            logger_local.warning("Continuing with backtesting despite weekly file creation failure")
    
    # 여기에서 원래의 티커 데이터 로딩 로직을 보존해야 함
    for ticker in tickers:
        try:
            logger_local.info(f"Loading data for {ticker}...")
            df = download_and_preprocess(ticker=ticker, start_date=start_date, end_date=end_date,
                                       force_download=force_download, log=logger_local)
            data_feed = FinanceDataReaderData(
                dataname=df,
                fromdate=start_dt,
                todate=end_dt,
                name=ticker
            )
            bt_data_feeds[ticker] = data_feed
            logger_local.info(f"Data feed created for {ticker}.")
        except Exception as e:
            logger_local.error(f"Failed to load data for {ticker}: {e}")
            if ticker == touchstone:
                raise ValueError(f"Touchstone ticker data retrieval failed for {touchstone}: {e}")
    
    if touchstone not in bt_data_feeds:
        raise ValueError(f"Touchstone ticker data for {touchstone} not loaded. Cannot proceed with backtesting.")
    
    logger_local.info(f"Loaded data for {len(bt_data_feeds)} tickers.")
    return bt_data_feeds

def analyze_market_regime(df: pd.DataFrame, current_date: Union[str, datetime.date, datetime.datetime],
                          ma_type: str = 'daily', ma_periods: List[int] = None, wma_periods: List[int] = None,
                          log: Optional[logging.Logger] = None) -> Tuple[str, Dict[str, float]]:
    """
    주어진 날짜의 시장 상태(레짐)를 분석하고 해당 상태에 따른 자산 배분을 반환합니다.
    
    :param df: 기준 티커의 DataFrame
    :param current_date: 분석할 날짜 (문자열 또는 datetime)
    :param ma_type: 'daily' 또는 'weekly'
    :param ma_periods: 일간 이동평균 기간 리스트 (없으면 config 사용)
    :param wma_periods: 주간 이동평균 기간 리스트 (없으면 config 사용)
    :param log: 로깅에 사용할 로거
    :return: (시장 상태, 자산 배분 딕셔너리)
    """
    logger_local = log or logger
    ma_periods = ma_periods or config.config.get("MA_PERIODS")
    wma_periods = wma_periods or config.config.get("WMA_PERIODS")
    current_date = pd.Timestamp(current_date)
    try:
        if current_date not in df.index:
            available = df.index[df.index <= current_date]
            if available.empty:
                raise ValueError(f"No data available before {current_date}")
            current_date = available[-1]
        row = df.loc[current_date]
    except Exception as e:
        logger_local.error(f"Error retrieving data for {current_date}: {e}")
        raise ValueError(f"Data not found for {current_date}")
    current_price = row['Close']
    if ma_type == 'weekly':
        return analyze_weekly_market_regime(df, current_date, current_price, wma_periods, logger_local)
    else:
        return analyze_daily_market_regime(df, current_date, current_price, ma_periods, logger_local)

def analyze_daily_market_regime(df: pd.DataFrame, current_date: pd.Timestamp, current_price: float,
                                ma_periods: List[int], logger: logging.Logger) -> Tuple[str, Dict[str, float]]:
    """
    일간 이동평균을 기반으로 시장 상태를 분석하고 적절한 자산 배분을 결정합니다.
    
    :param df: DataFrame
    :param current_date: 분석할 날짜 (pd.Timestamp)
    :param current_price: 해당 날짜의 종가
    :param ma_periods: 이동평균 기간 리스트
    :param logger: 로깅에 사용할 로거
    :return: (시장 상태, 자산 배분 딕셔너리)
    """
    row = df.loc[current_date]
    ma_values = {}
    for period in ma_periods:
        ma_col = f'MA_{period}'
        if ma_col in row and not pd.isna(row[ma_col]):
            ma_values[period] = row[ma_col]
        else:
            logger.warning(f"{ma_col} missing, computing on the fly.")
            ma_values[period] = df['Close'].rolling(window=period).mean().loc[current_date]
    logger.info(f"[{current_date}] Price: {current_price:.2f}, MA values: {ma_values}")
    ma_short = ma_values.get(ma_periods[0], 0)
    ma_mid = ma_values.get(ma_periods[1], 0)
    ma_mid2 = ma_values.get(ma_periods[2], 0)
    ma_long = ma_values.get(ma_periods[3], 0)
    if current_price > ma_short and ma_short > ma_mid and ma_mid > ma_long:
        regime = "Aggressive"
        allocation = config.config.get("AGGRESSIVE_ALLOCATION")
    elif current_price > ma_mid and ma_mid > ma_long:
        regime = "Moderate"
        allocation = config.config.get("MODERATE_ALLOCATION")
    elif current_price < ma_mid2 and current_price > ma_long:
        regime = "MidDefensive"
        allocation = config.config.get("MID_DEFENSIVE_ALLOCATION")
    elif current_price < ma_long:
        regime = "Defensive"
        allocation = config.config.get("DEFENSIVE_ALLOCATION")
    else:
        regime = "Neutral"
        allocation = config.config.get("ASSET_ALLOCATION")
    logger.info(f"Market regime: {regime}, Allocation: {allocation}")
    return regime, allocation

def analyze_weekly_market_regime(df: pd.DataFrame, current_date: pd.Timestamp, current_price: float,
                                 wma_periods: List[int], logger: logging.Logger) -> Tuple[str, Dict[str, float]]:
    """
    주간 이동평균을 기반으로 시장 상태를 분석하고 적절한 자산 배분을 결정합니다.
    
    :param df: DataFrame
    :param current_date: 분석할 날짜 (pd.Timestamp)
    :param current_price: 해당 날짜의 종가
    :param wma_periods: 주간 이동평균 기간 리스트
    :param logger: 로깅에 사용할 로거
    :return: (시장 상태, 자산 배분 딕셔너리)
    """
    row = df.loc[current_date]
    wma_values = {}
    # DataFrame의 열 이름은 CSV에서 대문자로 저장되므로 'WMA_10', 'WMA_30'
    for period in wma_periods:
        wma_col = f'WMA_{period}'
        if wma_col in row and not pd.isna(row[wma_col]):
            wma_values[period] = row[wma_col]
        else:
            logger.warning(f"Failed to get {wma_col}: Column not found in DataFrame. Using base allocation.")
            return "Neutral", config.config.get("ASSET_ALLOCATION")
    logger.info(f"[{current_date}] Price: {current_price:.2f}, Weekly MA: {wma_values}")
    wma_short = wma_values[wma_periods[0]]
    wma_long = wma_values[wma_periods[1]]
    if current_price > wma_short and wma_short > wma_long:
        regime = "Aggressive"
        allocation = config.config.get("AGGRESSIVE_ALLOCATION")
    elif current_price < wma_short and current_price > wma_long:
        regime = "Moderate"
        allocation = config.config.get("MODERATE_ALLOCATION")
    elif current_price < wma_long:
        regime = "Defensive"
        allocation = config.config.get("DEFENSIVE_ALLOCATION")
    else:
        regime = "Neutral"
        allocation = config.config.get("ASSET_ALLOCATION")
    logger.info(f"Weekly market regime: {regime}, Allocation: {allocation}")
    return regime, allocation

def create_weekly_data_file(ticker, start_date, end_date, data_dir=None, log=None):
    """
    특정 티커의 주간 데이터를 생성하여 별도 CSV 파일로 저장합니다.
    
    :param ticker: 티커 심볼 (예: 'SPY')
    :param start_date: 시작 날짜
    :param end_date: 종료 날짜
    :param data_dir: 저장 디렉토리
    :param log: 로거
    :return: 저장된 파일 경로
    """
    logger_local = log or logger
    directory = data_dir or config.config.DATA_DIR
    os.makedirs(directory, exist_ok=True)
    
    # 일간 데이터 다운로드
    logger_local.info(f"Downloading daily data for {ticker} weekly file creation...")
    df = fdr.DataReader(ticker, start_date, end_date)
    
    if df.empty:
        raise ValueError(f"No data available for {ticker}")
        
    # 주간 데이터 계산
    logger_local.info(f"Computing weekly data for {ticker}...")
    weekly_df = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # 주간 이동평균 계산
    for period in config.config.get("WMA_PERIODS", [10, 30]):
        weekly_df[f'WMA_{period}'] = weekly_df['Close'].rolling(window=period).mean()
    
    # 명시적으로 인덱스 이름을 'Date'로 설정
    weekly_df.index.name = 'Date'
    
    # 파일 저장 - index=True로 명시적으로 설정
    file_path = os.path.join(directory, f"{ticker}_weekly.csv")
    weekly_df.to_csv(file_path, index=True)
    logger_local.info(f"Saved weekly data for {ticker} to {file_path} ({len(weekly_df)} rows).")
    
    return file_path

