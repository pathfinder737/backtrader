# ========================= data_utils.py =========================
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
import traceback  # traceback 모듈 사용
import config

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

# ------------------ data_utils.py ------------------
# 본 파일 상단 부분( import, 기존 함수 등)은 생략
# 아래 새로 추가된 함수들을 전체 코드로 첨부

def is_first_business_day_of_quarter(date_obj) -> bool:
    """
    분기의 첫 영업일(1, 4, 7, 10월)인지 확인하고,
    해당 월의 첫 영업일인지 추가 확인
    """
    ts = pd.Timestamp(date_obj)
    if ts.month not in [1, 4, 7, 10]:
        return False
    return is_first_business_day_of_month(ts)

def is_last_business_day_of_quarter(date_obj) -> bool:
    """
    분기의 마지막 영업일(3, 6, 9, 12월)인지 확인하고,
    해당 월의 마지막 영업일인지 추가 확인
    """
    ts = pd.Timestamp(date_obj)
    if ts.month not in [3, 6, 9, 12]:
        return False
    return is_last_business_day_of_month(ts)

def is_first_business_day_of_half(date_obj) -> bool:
    """
    반기의 첫 영업일(1, 7월)인지 확인하고,
    해당 월의 첫 영업일인지 추가 확인
    """
    ts = pd.Timestamp(date_obj)
    if ts.month not in [1, 7]:
        return False
    return is_first_business_day_of_month(ts)

def is_last_business_day_of_half(date_obj) -> bool:
    """
    반기의 마지막 영업일(6, 12월)인지 확인하고,
    해당 월의 마지막 영업일인지 추가 확인
    """
    ts = pd.Timestamp(date_obj)
    if ts.month not in [6, 12]:
        return False
    return is_last_business_day_of_month(ts)

def is_first_business_day_of_year(date_obj) -> bool:
    """
    연초 첫 영업일(1월)인지 확인하고,
    해당 월의 첫 영업일인지 추가 확인
    """
    ts = pd.Timestamp(date_obj)
    if ts.month != 1:
        return False
    return is_first_business_day_of_month(ts)

def is_last_business_day_of_year(date_obj) -> bool:
    """
    연말 마지막 영업일(12월)인지 확인하고,
    해당 월의 마지막 영업일인지 추가 확인
    """
    ts = pd.Timestamp(date_obj)
    if ts.month != 12:
        return False
    return is_last_business_day_of_month(ts)

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

def download_and_preprocess(ticker: str,
                            start_date: str,
                            end_date: str,
                            force_download: bool = False,
                            data_dir: Optional[str] = None,
                            log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    1) 지정된 티커 데이터를 다운로드
    2) NaN 처리
    3) 만약 터치스톤이면 MA/WMA 계산
    4) 원본(raw) CSV는 "ticker_raw.csv"로 저장
    5) (슬라이싱 등은 여기서 하지 않고) 그대로 DataFrame 반환

    :param ticker: 티커 심볼
    :param start_date: 시작 날짜
    :param end_date: 종료 날짜
    :param force_download: 강제 다운로드
    :param data_dir: 저장 디렉토리
    :param log: 로거
    :return: 전처리된(하지만 슬라이싱 전) DataFrame
    """
    logger_local = log or logger
    directory = data_dir or config.config.DATA_DIR
    os.makedirs(directory, exist_ok=True)

    # touchstone(기준티커) 가져오기
    touchstone = config.config.get("TOUCHSTONE")

    # 항상 새롭게 다운로드
    force_download = True

    raw_file_path = os.path.join(directory, f"{ticker}_raw.csv")

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

            df.index.name = "Date"

            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns {missing} for {ticker}.")

            # 기본 NaN 처리
            orig_nan_count = df.isna().sum().sum()
            if orig_nan_count > 0:
                logger_local.info(f"Found {orig_nan_count} missing values in {ticker} data.")
                for col in df.columns:
                    nc = df[col].isna().sum()
                    if nc > 0:
                        logger_local.info(f"{ticker}: Column '{col}' has {nc} NaN")

                df['Volume'].fillna(0, inplace=True)
                df.interpolate(method='linear', inplace=True)
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)

            # 터치스톤만 MA/WMA 계산
            if ticker == touchstone:
                for period in config.config.get("MA_PERIODS", [21, 50, 150, 200]):
                    df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()

                weekly_df = calculate_weekly_data(df)
                for period in config.config.get("WMA_PERIODS", [10, 30]):
                    df = map_weekly_ma_to_daily(df, weekly_df, period)

            # ---- 여기서는 슬라이싱 하지 않는다. 원본(raw) CSV 저장 후 그대로 반환 ----
            df.to_csv(raw_file_path)
            logger_local.info(f"Saved RAW data for {ticker} to {raw_file_path} ({len(df)} rows).")
            return df

        except Exception as e:
            retry_count += 1
            logger_local.warning(f"Attempt {retry_count}/{max_retries} failed for {ticker}: {e}")
            logger_local.warning(traceback.format_exc())
            if retry_count >= max_retries:
                logger_local.error(f"Max retries exceeded for {ticker}.")
                raise ConnectionError(f"Cannot retrieve data for {ticker}: {e}")

    # 예외적으로 반복문이 빠져나갔을 때
    raise RuntimeError("Unexpected error in download_and_preprocess.")

def calculate_weekly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    일간 -> 주간 변환
    """
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Daily data has {nan_count} NaNs before weekly conversion")
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

    try:
        weekly_df = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        weekly_df.dropna(inplace=True)

        for period in config.config.get("WMA_PERIODS", [10, 30]):
            weekly_df[f'WMA_{period}'] = weekly_df['Close'].rolling(window=period).mean()

        return weekly_df
    except Exception as e:
        logger.error(f"Error calculating weekly data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['Open','High','Low','Close','Volume'])

def map_weekly_ma_to_daily(daily_df: pd.DataFrame,
                           weekly_df: pd.DataFrame,
                           period: int) -> pd.DataFrame:
    """
    주간 MA를 일간에 매핑
    """
    wma_col = f'WMA_{period}'
    if wma_col not in weekly_df.columns:
        daily_df[wma_col] = np.nan
        return daily_df

    daily_cp = daily_df.copy()
    # merge_asof
    daily_reset = daily_cp.reset_index()
    weekly_reset = weekly_df.reset_index()
    wma_data = weekly_reset[['Date', wma_col]].dropna(subset=[wma_col])
    merged = pd.merge_asof(daily_reset, wma_data, on='Date', direction='backward')
    daily_cp[wma_col] = merged[wma_col].values
    return daily_cp

def prepare_backtrader_data(tickers: List[str],
                            start_date: str,
                            end_date: str,
                            data_mode: str = 'offline',
                            log: Optional[logging.Logger] = None
                           ) -> Dict[str, FinanceDataReaderData]:
    """
    1) 모든 티커를 download_and_preprocess()로 로드 (raw CSV 생성)
    2) 비-터치스톤 자산 중 '가장 늦은 시작일' 찾음
    3) 터치스톤은 그보다 1년 앞의 데이터부터, 비-터치스톤은 최신 시작일부터 슬라이싱
    4) 최종 슬라이싱된 df를 "ticker.csv"로 저장
    5) Backtrader용 feed 생성하여 반환
    """
    logger_local = log or logger
    touchstone = config.config.get("TOUCHSTONE")
    bt_data_feeds = {}

    try:
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        logger_local.error(f"Invalid date format: {e}")
        raise ValueError("Dates must be in 'YYYY-MM-DD' format.")

    df_dict = {}
    # (1) 모든 티커 raw 데이터 로드
    for t in tickers:
        logger_local.info(f"Loading data for {t}...")
        df = download_and_preprocess(
            ticker=t,
            start_date=start_date,
            end_date=end_date,
            force_download=True,
            log=logger_local
        )
        if df is None or df.empty:
            logger_local.warning(f"{t}: Data is empty. Possibly no valid data.")
            if t == touchstone:
                raise ValueError(f"Touchstone {touchstone} has no valid data!")
        df_dict[t] = df

    if touchstone not in df_dict:
        raise ValueError(f"Touchstone {touchstone} not loaded. Cannot proceed.")

    # (2) 비-터치스톤의 최신 시작일
    non_touch = [x for x in tickers if x != touchstone]
    start_dates = []
    for nt in non_touch:
        df_nt = df_dict[nt]
        if not df_nt.empty:
            start_dates.append(df_nt.index.min())
    if not start_dates:
        logger_local.warning("No non-touchstone data found. Will not align.")
        max_start_date = None
    else:
        max_start_date = max(start_dates)
        logger_local.info(f"Max start date among non-touchstone: {max_start_date}")

    # (3) 터치스톤: max_start_date - 1년 / 비터치스톤: max_start_date
    if max_start_date is not None:
        one_year_before = max_start_date - pd.DateOffset(years=1)
        df_dict[touchstone] = df_dict[touchstone].loc[df_dict[touchstone].index >= one_year_before]
        for nt in non_touch:
            df_dict[nt] = df_dict[nt].loc[df_dict[nt].index >= max_start_date]

    # (4) 슬라이싱 후 최종 CSV 저장 + Backtrader feed 생성
    for t in tickers:
        final_df = df_dict[t]
        if final_df.empty:
            logger_local.warning(f"{t}: empty after slicing. Skipping feed.")
            continue

        final_csv_path = os.path.join(config.config.DATA_DIR, f"{t}.csv")
        final_df.to_csv(final_csv_path)
        logger_local.info(f"Saved ALIGNED data for {t} to {final_csv_path} ({len(final_df)} rows).")

        data_feed = FinanceDataReaderData(
            dataname=final_df,
            fromdate=start_dt,
            todate=end_dt,
            name=t
        )
        bt_data_feeds[t] = data_feed
        logger_local.info(f"Created data feed for {t} with {len(final_df)} rows after slicing.")

    if touchstone not in bt_data_feeds:
        raise ValueError(f"Touchstone {touchstone} data feed not created. Cannot proceed.")

    logger_local.info(f"Loaded and aligned data for {len(bt_data_feeds)} tickers.")
    return bt_data_feeds
