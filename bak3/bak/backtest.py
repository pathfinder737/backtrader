# backtest.py
# 백테스트 실행 및 결과 시각화 기능을 담당하는 모듈입니다.

import logging
import datetime
import traceback
import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import config
from date_data_utils import download_and_preprocess, FinanceDataReaderData
from strategy import MonthlyRebalanceStrategy

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
            if config.DATA_MODE == 'online':
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
            try:
                start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError as e:
                logging.error(f"날짜 형식 오류: {e}")
                raise ValueError(f"날짜 형식이 올바르지 않습니다. 'YYYY-MM-DD' 형식을 사용하세요.")
            
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