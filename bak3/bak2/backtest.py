# backtest.py
# 백테스트 실행 및 결과 시각화 기능을 담당하는 모듈입니다.

import os
import logging
import datetime
import traceback
import backtrader as bt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

import config
from data_utils import prepare_backtrader_data
from strategy import MonthlyRebalanceStrategy


def run_backtest(config_file: Optional[str] = None, 
               logger: Optional[logging.Logger] = None):
    """
    백테스트를 실행하고 결과를 시각화합니다.
    
    Args:
        config_file: 설정 파일 경로 (None이면 기본 설정 사용)
        logger: 로깅에 사용할 로거
        
    Returns:
        백테스트 결과
    """
    # 로거 설정
    log = logger or logging.getLogger('backtest')
    
    # 설정 로드
    if config_file:
        config.load_config(config_file)
    
    # 설정값 유효성 검사
    validation_results = config.validate_config()
    if not validation_results.get('is_valid', False):
        log.error("설정 유효성 검사 실패")
        raise ValueError("설정이 유효하지 않습니다. 로그를 확인하세요.")
    
    # Cerebro 객체 생성
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MonthlyRebalanceStrategy)

    # 설정값 가져오기
    start_date = config.get_config('START_DATE')
    end_date = config.get_config('END_DATE')
    tickers = config.get_config('TICKERS')
    touchstone = config.get_config('TOUCHSTONE')
    data_mode = config.get_config('DATA_MODE')

    # TOUCHSTONE이 TICKERS에 있는지 확인
    if touchstone not in tickers:
        log.error(f"기준 티커({touchstone})가 티커 목록에 없습니다.")
        raise ValueError(f"기준 티커({touchstone})가 티커 목록에 포함되어 있어야 합니다.")
    
    # 자산 배분에 포함된 모든 티커가 TICKERS에 있는지 확인
    allocation_tickers = set(config.get_config('ASSET_ALLOCATION').keys())
    missing_tickers = allocation_tickers - set(tickers)
    if missing_tickers:
        log.error(f"자산 배분에 포함되었지만 TICKERS에 없는 티커: {missing_tickers}")
        raise ValueError(f"자산 배분에 포함된 모든 티커는 TICKERS 목록에 있어야 합니다. 누락된 티커: {missing_tickers}")
    
    # 데이터 로드 및 추가
    try:
        log.info("데이터 로드 시작")
        data_feeds = prepare_backtrader_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            data_mode=data_mode,
            log=log
        )
        
        # 모든 데이터 피드를 Cerebro에 추가
        for ticker, data_feed in data_feeds.items():
            cerebro.adddata(data_feed, name=ticker)
            log.info(f"{ticker} 데이터 추가 완료")
            
        log.info("모든 데이터 로드 및 추가 완료")
    except Exception as e:
        log.error(f"데이터 로드 및 추가 중 오류 발생: {str(e)}")
        log.error(traceback.format_exc())
        raise
    
    # 초기 설정
    cerebro.broker.setcash(config.get_config('INITIAL_CASH'))
    cerebro.broker.setcommission(commission=config.get_config('COMMISSION'))

    # 분석 지표 추가
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=config.get_config('RISK_FREE_RATE'))
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

    # 백테스팅 실행
    initial_value = cerebro.broker.getvalue()
    log.info('Initial portfolio value: {:,.2f}'.format(initial_value))
    
    try:
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        log.info('Final portfolio value: {:,.2f}'.format(final_value))
        total_return = ((final_value / config.get_config('INITIAL_CASH')) - 1) * 100
        log.info(f"Total return: {total_return:.2f}%")
        
        # 결과 시각화
        visualize_results(results, initial_value, final_value, log)
        
        return results
        
    except Exception as e:
        log.error(f"백테스트 실행 중 오류 발생: {str(e)}")
        log.error(traceback.format_exc())
        raise


def visualize_results(results, initial_value, final_value, logger=None):
    """
    백테스트 결과를 시각화합니다.
    
    Args:
        results: 백테스트 결과
        initial_value: 초기 포트폴리오 가치
        final_value: 최종 포트폴리오 가치
        logger: 로깅에 사용할 로거
    """
    log = logger or logging.getLogger('backtest')
    
    if not results or len(results) == 0:
        log.warning("시각화를 위한 백테스트 결과가 없습니다.")
        return
        
    strategy = results[0]  # 첫 번째 전략 인스턴스
    
    if not hasattr(strategy, 'portfolio_dates') or not strategy.portfolio_dates:
        log.warning("포트폴리오 가치 데이터가 없어 그래프를 생성할 수 없습니다.")
        return
    
    # 타임스탬프를 파일명에 추가
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 결과 저장 디렉토리 확인
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # 1. 포트폴리오 가치 그래프
    create_portfolio_value_chart(
        strategy=strategy, 
        initial_value=initial_value, 
        final_value=final_value, 
        timestamp=timestamp,
        logger=log
    )
    
    # 2. 연간 수익률 그래프
    create_annual_returns_chart(
        strategy=strategy, 
        timestamp=timestamp,
        logger=log
    )
    
    # 3. 월간 수익률 히트맵
    create_monthly_returns_heatmap(
        strategy=strategy, 
        timestamp=timestamp,
        logger=log
    )
    
    # 4. 최대 낙폭 그래프
    create_drawdown_chart(
        strategy=strategy, 
        timestamp=timestamp,
        logger=log
    )


def create_portfolio_value_chart(strategy, initial_value, final_value, timestamp, logger=None):
    """
    포트폴리오 가치 변화 그래프를 생성합니다.
    
    Args:
        strategy: 전략 인스턴스
        initial_value: 초기 포트폴리오 가치
        final_value: 최종 포트폴리오 가치
        timestamp: 파일명에 추가할 타임스탬프
        logger: 로깅에 사용할 로거
    """
    log = logger or logging.getLogger('backtest')
    
    try:
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
        
        # 결과 저장
        file_path = os.path.join(config.RESULTS_DIR, f'portfolio_value_{timestamp}.png')
        plt.savefig(file_path)
        log.info(f"포트폴리오 가치 그래프를 '{file_path}'에 저장했습니다.")
        
        plt.close()
    except Exception as e:
        log.error(f"포트폴리오 가치 그래프 생성 중 오류 발생: {str(e)}")


def create_annual_returns_chart(strategy, timestamp, logger=None):
    """
    연간 수익률 막대 그래프를 생성합니다.
    
    Args:
        strategy: 전략 인스턴스
        timestamp: 파일명에 추가할 타임스탬프
        logger: 로깅에 사용할 로거
    """
    log = logger or logging.getLogger('backtest')
    
    try:
        annual_metrics = strategy.compute_annual_metrics()
        if not annual_metrics:
            log.warning("연간 수익률 데이터가 없어 그래프를 생성할 수 없습니다.")
            return
        
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
        
        # 결과 저장
        file_path = os.path.join(config.RESULTS_DIR, f'annual_returns_{timestamp}.png')
        plt.savefig(file_path)
        log.info(f"연간 수익률 그래프를 '{file_path}'에 저장했습니다.")
        
        plt.close()
    except Exception as e:
        log.error(f"연간 수익률 그래프 생성 중 오류 발생: {str(e)}")


def create_monthly_returns_heatmap(strategy, timestamp, logger=None):
    """
    월간 수익률 히트맵을 생성합니다.
    
    Args:
        strategy: 전략 인스턴스
        timestamp: 파일명에 추가할 타임스탬프
        logger: 로깅에 사용할 로거
    """
    log = logger or logging.getLogger('backtest')
    
    try:
        # 포트폴리오 데이터를 DataFrame으로 변환
        df = pd.DataFrame({
            'Date': strategy.portfolio_dates,
            'Value': strategy.portfolio_values
        })
        
        # 날짜 형식 변환 및 월간 수익률 계산
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 월별 수익률 계산
        monthly_returns = df['Value'].resample('M').last().pct_change() * 100
        monthly_returns = monthly_returns.to_frame('Return')
        monthly_returns['Year'] = monthly_returns.index.year
        monthly_returns['Month'] = monthly_returns.index.month
        
        # 피벗 테이블 생성
        pivot_table = monthly_returns.pivot_table(
            values='Return', 
            index='Year', 
            columns='Month', 
            aggfunc='mean'
        )
        
        # 열 이름을 월 이름으로 변경
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[m-1] for m in pivot_table.columns]
        
        # 히트맵 그리기
        plt.figure(figsize=(12, 8))
        
        # seaborn이 있으면 사용, 없으면 matplotlib로 대체
        try:
            import seaborn as sns
            sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
        except ImportError:
            # matplotlib로 히트맵 구현
            im = plt.imshow(pivot_table.values, cmap="RdYlGn")
            
            # 주석 추가
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    value = pivot_table.values[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if abs(value) > 10 else 'black'
                        plt.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color)
            
            # 축 설정
            plt.colorbar(im)
            plt.yticks(range(len(pivot_table.index)), pivot_table.index)
            plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
        
        plt.title('Monthly Returns Heatmap (%)')
        plt.tight_layout()
        
        # 결과 저장
        file_path = os.path.join(config.RESULTS_DIR, f'monthly_returns_heatmap_{timestamp}.png')
        plt.savefig(file_path)
        log.info(f"월간 수익률 히트맵을 '{file_path}'에 저장했습니다.")
        
        plt.close()
    except Exception as e:
        log.error(f"월간 수익률 히트맵 생성 중 오류 발생: {str(e)}")


def create_drawdown_chart(strategy, timestamp, logger=None):
    """
    최대 낙폭(Drawdown) 그래프를 생성합니다.
    
    Args:
        strategy: 전략 인스턴스
        timestamp: 파일명에 추가할 타임스탬프
        logger: 로깅에 사용할 로거
    """
    log = logger or logging.getLogger('backtest')
    
    try:
        # 포트폴리오 가치 배열
        values = np.array(strategy.portfolio_values)
        dates = strategy.portfolio_dates
        
        # 최대 누적 가치 및 낙폭 계산
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100  # 백분율로 변환
        
        # 그래프 그리기
        plt.figure(figsize=(12, 8))
        plt.plot(dates, drawdown)
        plt.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # x축 눈금 설정 (10개 내외로 제한)
        n = len(dates)
        step = max(1, n // 10)
        plt.xticks(dates[::step], [d.strftime('%Y-%m') for d in dates[::step]], rotation=45)
        
        # 최대 낙폭 표시
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        max_dd_date = dates[max_dd_idx].strftime('%Y-%m-%d')
        
        plt.axhline(y=max_dd, color='r', linestyle='--', alpha=0.7)
        plt.text(dates[len(dates)//2], max_dd*1.1, f'Max Drawdown: {max_dd:.2f}% ({max_dd_date})', 
                ha='center', color='r')
        
        plt.tight_layout()
        
        # 결과 저장
        file_path = os.path.join(config.RESULTS_DIR, f'drawdown_{timestamp}.png')
        plt.savefig(file_path)
        log.info(f"최대 낙폭 그래프를 '{file_path}'에 저장했습니다.")
        
        plt.close()
    except Exception as e:
        log.error(f"최대 낙폭 그래프 생성 중 오류 발생: {str(e)}")


def save_results_to_csv(strategy, timestamp, logger=None):
    """
    백테스트 결과를 CSV 파일로 저장합니다.
    
    Args:
        strategy: 전략 인스턴스
        timestamp: 파일명에 추가할 타임스탬프
        logger: 로깅에 사용할 로거
    """
    log = logger or logging.getLogger('backtest')
    
    try:
        # 포트폴리오 가치 데이터 저장
        portfolio_df = pd.DataFrame({
            'Date': strategy.portfolio_dates,
            'Value': strategy.portfolio_values
        })
        
        portfolio_file = os.path.join(config.RESULTS_DIR, f'portfolio_values_{timestamp}.csv')
        portfolio_df.to_csv(portfolio_file, index=False)
        log.info(f"포트폴리오 가치 데이터를 '{portfolio_file}'에 저장했습니다.")
        
        # 연간 성과 지표 저장
        annual_metrics = strategy.compute_annual_metrics()
        if annual_metrics:
            annual_df = pd.DataFrame.from_dict(annual_metrics, orient='index')
            annual_file = os.path.join(config.RESULTS_DIR, f'annual_metrics_{timestamp}.csv')
            annual_df.to_csv(annual_file)
            log.info(f"연간 성과 지표를 '{annual_file}'에 저장했습니다.")
        
        # 일별 수익률 계산 및 저장
        if hasattr(strategy, 'daily_returns') and strategy.daily_returns:
            daily_df = pd.DataFrame({
                'Date': strategy.portfolio_dates[1:],  # 첫 날은 수익률 계산에서 제외
                'Return': strategy.daily_returns
            })
            daily_file = os.path.join(config.RESULTS_DIR, f'daily_returns_{timestamp}.csv')
            daily_df.to_csv(daily_file, index=False)
            log.info(f"일별 수익률 데이터를 '{daily_file}'에 저장했습니다.")
    
    except Exception as e:
        log.error(f"결과 CSV 저장 중 오류 발생: {str(e)}")