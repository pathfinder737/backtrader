# date_data_utils.py
# 날짜 유틸리티 및 데이터 처리 관련 함수들을 정의합니다.

import os
import logging
import datetime
import pandas as pd
import numpy as np
import backtrader as bt
import FinanceDataReader as fdr
import pandas_market_calendars as mcal
from pandas.tseries.offsets import BMonthEnd, BusinessDay
import config

# ------------------------------
# NYSE 캘린더 및 날짜 유틸리티 함수
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
    try:
        ts = pd.Timestamp(date)
        last_day_of_month = ts + BMonthEnd(0)
        # 15일 범위로 늘려 긴 휴일 기간도 고려
        schedule = nyse.schedule(start_date=last_day_of_month - pd.Timedelta(days=15), 
                                end_date=last_day_of_month)
        if not schedule.empty:
            return schedule.index[-1].date()
        else:
            return (ts + BMonthEnd(0)).date()
    except Exception as e:
        logging.error(f"마지막 영업일 계산 중 오류: {str(e)}")
        # 에러 발생 시 기본 BMonthEnd 사용 (NYSE 캘린더에 의존하지 않음)
        return (ts + BMonthEnd(0)).date()

def get_first_business_day(date):
    """
    주어진 날짜가 속한 월의 첫 영업일을 반환합니다.
    
    Args:
        date: 날짜 객체 또는 문자열
        
    Returns:
        첫 영업일(date 객체)
    """
    try:
        ts = pd.Timestamp(date).replace(day=1)
        # 15일 범위로 늘려 긴 휴일 기간도 고려
        schedule = nyse.schedule(start_date=ts, 
                                end_date=ts + pd.Timedelta(days=15))
        if not schedule.empty:
            return schedule.index[0].date()
        else:
            return (ts + BusinessDay(0)).date()
    except Exception as e:
        logging.error(f"첫 영업일 계산 중 오류: {str(e)}")
        # 에러 발생 시 기본 BusinessDay 사용 (NYSE 캘린더에 의존하지 않음)
        return (ts + BusinessDay(0)).date()

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