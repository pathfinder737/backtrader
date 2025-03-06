# data_utils.py
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

logger = logging.getLogger('data_utils')

try:
    nyse = mcal.get_calendar('NYSE')
except Exception as e:
    logger.warning(f"Failed to initialize NYSE calendar: {e}. Using default business day.")
    nyse = None

def get_last_business_day(date_obj: Union[str, datetime.date, datetime.datetime]) -> datetime.date:
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
    return pd.Timestamp(date_obj).date() == get_last_business_day(date_obj)

def is_first_business_day_of_month(date_obj: Union[str, datetime.date, datetime.datetime]) -> bool:
    return pd.Timestamp(date_obj).date() == get_first_business_day(date_obj)

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
    'touchstone'(예: SPY)인 경우에만 이동평균선(MA, WMA)을 계산하고,
    다른 자산(QQQ, IEF, IAU 등)은 이동평균 계산을 건너뛴다.

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

    # touchstone(기준티커) 가져오기: 예) "SPY"
    touchstone = config.config.get("TOUCHSTONE")

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

            # NaN 값 검사 및 처리
            orig_nan_count = df.isna().sum().sum()
            if orig_nan_count > 0:
                logger_local.info(f"Found {orig_nan_count} missing values in {ticker} data.")
                # 열별 NaN 개수 로깅
                for col in df.columns:
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        logger_local.info(f"{ticker}: Column '{col}' has {nan_count} NaN values")

                # 기본 NaN 처리
                df['Volume'].fillna(0, inplace=True)
                df.interpolate(method='linear', inplace=True)
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)

                # 처리 후 NaN 값 재확인
                post_nan_count = df.isna().sum().sum()
                if post_nan_count > 0:
                    logger_local.warning(f"Still have {post_nan_count} NaN values after interpolation in {ticker}")
                    for col in df.columns:
                        if df[col].isna().sum() > 0:
                            logger_local.warning(f"Column {col} still has NaN values")

            # --------------------------------------------------
            # (중요) 아래 부분: SPY(=touchstone)일 때만 MA, WMA 계산
            # --------------------------------------------------
            if ticker == touchstone:
                # 일간 이동평균(MA) 계산
                for period in config.config.get("MA_PERIODS", [21, 50, 150, 200]):
                    df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
                    ma_nan_count = df[f'MA_{period}'].isna().sum()
                    if ma_nan_count > 0:
                        logger_local.info(f"{ticker}: MA_{period} has {ma_nan_count} NaN values at the beginning")

                # 주간 데이터 계산
                weekly_df = calculate_weekly_data(df)
                logger_local.info(f"Created weekly data for {ticker} with {len(weekly_df)} rows")

                # 주간 이동평균 값을 일간 데이터에 매핑(WMA 계산)
                for period in config.config.get("WMA_PERIODS", [10, 30]):
                    df = map_weekly_ma_to_daily(df, weekly_df, period)
                    wma_col = f'WMA_{period}'
                    if wma_col in df.columns:
                        wma_nan_count = df[wma_col].isna().sum()
                        if wma_nan_count > 0:
                            logger_local.warning(f"{ticker}: {wma_col} has {wma_nan_count}/{len(df)} NaN values")
                logger_local.info(f"MA/WMA calculation done for {ticker}")
            else:
                # SPY가 아닌 티커(QQQ, IEF, IAU 등)는 MA 계산 스킵
                logger_local.info(f"Skipping MA calculations for non-touchstone ticker: {ticker}")

            # 최종 NaN 값 확인
            final_nan_count = df.isna().sum().sum()
            if final_nan_count > 0:
                logger_local.warning(f"Final DataFrame for {ticker} contains {final_nan_count} NaN values")
                for col in df.columns:
                    col_nan = df[col].isna().sum()
                    if col_nan > 0:
                        logger_local.warning(f"{ticker}: Column {col} has {col_nan} NaN values")

            # CSV 저장
            df.to_csv(file_path)
            logger_local.info(f"Saved preprocessed data for {ticker} to {file_path} ({len(df)} rows).")
            return df

        except Exception as e:
            retry_count += 1
            logger_local.warning(f"Attempt {retry_count}/{max_retries} failed for {ticker}: {e}")
            logger_local.warning(traceback.format_exc())
            if retry_count >= max_retries:
                logger_local.error(f"Max retries exceeded for {ticker}.")
                raise ConnectionError(f"Cannot retrieve data for {ticker}: {e}")

    raise RuntimeError("Unexpected error in download_and_preprocess.")

def calculate_weekly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    일간 데이터를 기반으로 주간 데이터를 계산합니다.
    NaN 값 처리와 로깅을 개선했습니다.
    
    :param df: 일간 DataFrame
    :return: 주간 DataFrame (각 주 마지막 거래일 기준)
    """
    # NaN 값 확인
    nan_values = df.isna().sum().sum()
    if nan_values > 0:
        logger.warning(f"Daily data contains {nan_values} NaN values before weekly conversion")
        
        # 주간 집계에 문제가 될 수 있는 NaN 값 처리
        temp_df = df.copy()
        temp_df.fillna(method='ffill', inplace=True)
        temp_df.fillna(method='bfill', inplace=True)
        
        # 처리 후에도 NaN이 남아있는지 확인
        remaining_nans = temp_df.isna().sum().sum()
        if remaining_nans > 0:
            logger.error(f"Still have {remaining_nans} NaN values after filling")
            # 최소한으로 작동하도록 0으로 채움
            temp_df.fillna(0, inplace=True)
        
        # 처리된 데이터프레임 사용
        df = temp_df
    
    # 주간 데이터 계산
    try:
        weekly_df = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # NaN 행 제거 (거래일이 없는 주)
        weekly_df.dropna(inplace=True)
        
        # 주간 이동평균 계산
        for period in config.config.get("WMA_PERIODS", [10, 30]):
            weekly_df[f'WMA_{period}'] = weekly_df['Close'].rolling(window=period).mean()
            
            # WMA의 NaN 값 개수 체크
            wma_nan_count = weekly_df[f'WMA_{period}'].isna().sum()
            if wma_nan_count > 0:
                logger.info(f"Weekly MA_{period} has {wma_nan_count}/{len(weekly_df)} NaN values at the beginning")
        
        # 최종 결과 확인
        if weekly_df.empty:
            logger.error("Weekly DataFrame is empty after processing")
        else:
            logger.info(f"Created weekly DataFrame with {len(weekly_df)} rows")
            
        return weekly_df
        
    except Exception as e:
        logger.error(f"Error calculating weekly data: {str(e)}")
        logger.error(traceback.format_exc())
        # 빈 프레임이라도 반환하여 상위 함수 진행
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

def map_weekly_ma_to_daily(daily_df: pd.DataFrame, weekly_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    주간 이동평균 값을 일간 데이터에 매핑합니다.
    기존 데이터프레임 컬럼 유지하며 WMA 컬럼만 추가합니다.
    """
    # 인덱스 이름 확인 및 설정
    if daily_df.index.name is None:
        daily_df.index.name = "Date"
    if weekly_df.index.name is None:
        weekly_df.index.name = "Date"
    
    # 원본 컬럼과 인덱스 보존을 위해 복제
    result_df = daily_df.copy()
    
    # WMA 열이 없거나 모두 NaN인지 확인
    wma_col = f'WMA_{period}'
    if wma_col not in weekly_df.columns or weekly_df[wma_col].isna().all():
        logger.error(f"{wma_col} missing or all NaN in weekly data")
        # 모든 값을 NaN으로 채운 열 추가 (백테스트에서 감지 후 처리)
        result_df[wma_col] = np.nan
        return result_df
    
    # 일간 데이터와 주간 데이터 준비
    daily_reset = daily_df.reset_index()
    weekly_reset = weekly_df.reset_index()
    
    # 필요한 열만 추출
    wma_data = weekly_reset[['Date', wma_col]].dropna(subset=[wma_col])
    
    # merge_asof로 병합 (일간 데이터에 가장 가까운 이전 주간 데이터 매핑)
    merged = pd.merge_asof(daily_reset, wma_data, on='Date', direction='backward')
    
    # WMA 열 추출해서 원본 데이터프레임에 추가
    result_df[wma_col] = merged[wma_col].values
    
    # NaN 값이 얼마나 있는지 확인
    nan_count = result_df[wma_col].isna().sum()
    if nan_count > 0:
        logger.warning(f"{wma_col} has {nan_count}/{len(result_df)} NaN values after mapping")
        
        # 첫 번째 유효값으로 초반 NaN 채우기 (백테스트 초기에 활용 가능)
        first_valid = result_df[wma_col].first_valid_index()
        if first_valid is not None:
            first_value = result_df.loc[first_valid, wma_col]
            result_df[wma_col] = result_df[wma_col].fillna(first_value)
            logger.info(f"Filled initial NaN values in {wma_col} with {first_value}")
    
    return result_df

def prepare_backtrader_data(tickers: List[str], start_date: str, end_date: str,
                           data_mode: str = 'offline',
                           log: Optional[logging.Logger] = None) -> Dict[str, FinanceDataReaderData]:

    """
    여러 티커의 데이터를 다운로드한 뒤,
    1) 비-터치스톤 자산 중 '가장 늦은 시작일'을 찾고
    2) 터치스톤(SPY)은 그보다 1년 앞선 구간부터,
       다른 자산은 해당 최신 시작일부터
    로 슬라이싱하여 모두 동일 구간으로 맞춘다.
    
    그리고 각 DataFrame을 Backtrader용 PandasData 형태로 변환해 반환.
    """
    logger_local = log or logger
    # 항상 새롭게 다운로드
    force_download = True
    touchstone = config.config.get("TOUCHSTONE")
    bt_data_feeds = {}

    # 날짜 검증
    try:
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        logger_local.error(f"Invalid date format: {e}")
        raise ValueError("Dates must be in 'YYYY-MM-DD' format.")

    # ---------------------------------------------------------
    # (1) 모든 티커의 DataFrame을 먼저 download_and_preprocess로 로드
    # ---------------------------------------------------------
    df_dict = {}
    for ticker in tickers:
        try:
            logger_local.info(f"Loading data for {ticker}...")
            df = download_and_preprocess(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                force_download=force_download,
                log=logger_local
            )

            if df is None or df.empty:
                logger_local.error(f"Empty or invalid DataFrame for {ticker}")
                if ticker == touchstone:
                    raise ValueError(f"Touchstone {touchstone} has no valid data!")
            df_dict[ticker] = df

        except Exception as e:
            logger_local.error(f"Failed to load data for {ticker}: {e}")
            logger_local.error(traceback.format_exc())
            # Touchstone 로드 실패 시 바로 중단
            if ticker == touchstone:
                raise ValueError(f"Touchstone ticker data retrieval failed for {touchstone}: {e}")

    if touchstone not in df_dict:
        raise ValueError(f"Touchstone {touchstone} not loaded. Cannot proceed.")

    # ---------------------------------------------------------
    # (2) 비-터치스톤 자산들의 '가장 늦은 시작일' 찾기
    # ---------------------------------------------------------
    non_touch_tickers = [t for t in tickers if t != touchstone]
    start_dates = []
    for nt in non_touch_tickers:
        df_nt = df_dict.get(nt, pd.DataFrame())
        if not df_nt.empty:
            start_dates.append(df_nt.index.min())

    # start_dates가 비어있다면, 비-터치스톤이 모두 empty라서 문제이므로 예외 처리
    if not start_dates:
        logger_local.warning("No valid non-touchstone tickers with data. Proceeding without alignment.")
        # 이렇게 되면 사실상 touchstone만 있는 상태이므로, 바로 feed 생성으로 넘어감
        # 필요 시 여기서 return 해도 됨
        max_start_date = None
    else:
        max_start_date = max(start_dates)
        logger_local.info(f"Max start date among non-touchstone: {max_start_date}")

    # ---------------------------------------------------------
    # (3) 터치스톤(SPY)은 '가장 늦은 시작일 - 1년'부터 슬라이싱
    #     나머지 자산은 '가장 늦은 시작일'부터 슬라이싱
    # ---------------------------------------------------------
    if max_start_date is not None:
        one_year_before = max_start_date - pd.DateOffset(years=1)

        # Touchstone df
        df_touch = df_dict[touchstone]
        # 슬라이싱 수행
        df_dict[touchstone] = df_touch.loc[df_touch.index >= one_year_before]

        for nt in non_touch_tickers:
            df_nt = df_dict[nt]
            df_dict[nt] = df_nt.loc[df_nt.index >= max_start_date]

    # ---------------------------------------------------------
    # (4) 슬라이싱 후, Backtrader 데이터 피드로 변환
    # ---------------------------------------------------------
    from data_utils import FinanceDataReaderData  # 이미 이 파일 안에 있다면 import不要

    for ticker in tickers:
        df_sliced = df_dict[ticker]
        if df_sliced.empty:
            logger_local.warning(f"Data for {ticker} is empty after slicing. Skipping.")
            continue

        nan_count = df_sliced.isna().sum().sum()
        if nan_count > 0:
            logger_local.warning(f"DataFrame for {ticker} contains {nan_count} NaN values after slicing")

        # Backtrader feed 생성
        data_feed = FinanceDataReaderData(
            dataname=df_sliced,
            fromdate=start_dt,
            todate=end_dt,
            name=ticker
        )
        bt_data_feeds[ticker] = data_feed
        logger_local.info(f"Created data feed for {ticker} with {len(df_sliced)} rows after slicing.")

    # 마지막으로 touchstone 불러오기 확인
    if touchstone not in bt_data_feeds:
        raise ValueError(f"Touchstone {touchstone} data feed not created. Cannot proceed.")

    logger_local.info(f"Loaded and aligned data for {len(bt_data_feeds)} tickers.")
    return bt_data_feeds

def analyze_market_regime(df: pd.DataFrame, current_date: Union[str, datetime.date, datetime.datetime],
                          ma_type: str = 'daily', ma_periods: List[int] = None, wma_periods: List[int] = None,
                          log: Optional[logging.Logger] = None) -> Tuple[str, Dict[str, float]]:
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
    row = df.loc[current_date]
    wma_values = {}
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
    logger_local = log or logger
    directory = data_dir or config.config.DATA_DIR
    os.makedirs(directory, exist_ok=True)
    
    logger_local.info(f"Downloading daily data for {ticker} weekly file creation...")
    df = fdr.DataReader(ticker, start_date, end_date)
    
    if df.empty:
        raise ValueError(f"No data available for {ticker}")
        
    logger_local.info(f"Computing weekly data for {ticker}...")
    weekly_df = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    for period in config.config.get("WMA_PERIODS", [10, 30]):
        weekly_df[f'WMA_{period}'] = weekly_df['Close'].rolling(window=period).mean()
    
    weekly_df.index.name = 'Date'
    file_path = os.path.join(directory, f"{ticker}_weekly.csv")
    weekly_df.to_csv(file_path, index=True)
    logger_local.info(f"Saved weekly data for {ticker} to {file_path} ({len(weekly_df)} rows).")
    
    return file_path
