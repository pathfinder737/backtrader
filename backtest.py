import os
import logging
import datetime
import traceback
import backtrader as bt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional
import config
from data_utils import prepare_backtrader_data
from strategy import MonthlyRebalanceStrategy, PortfolioDataObserver

def set_common_xaxis(ax, dates):
    n = len(dates)
    step = max(1, n // 10)
    ax.set_xticks(dates[::step])
    ax.set_xticklabels([d.strftime('%Y-%m') for d in dates[::step]], rotation=45)

def run_backtest(config_file: Optional[str] = None, logger: Optional[logging.Logger] = None):
    """
    백테스트 실행 함수.
    Analyzer를 사용하여 CAGR, MDD, Sharpe 등 통계치를 얻는 방식으로 변경.
    """
    log = logger or logging.getLogger('backtest')

    if config_file:
        config.config.load_config(config_file)

    validation = config.config.validate()
    if not validation.get('is_valid', False):
        log.error("Config validation failed.")
        raise ValueError("Invalid configuration. Check logs for details.")

    cerebro = bt.Cerebro()
    # ------------------ (1) 전략 등록 ------------------
    cerebro.addstrategy(MonthlyRebalanceStrategy)

    # ------------------ (2) Observer 등록 추가 ------------------
    cerebro.addobserver(PortfolioDataObserver)
    # ------------------------------------------------------------
    start_date = config.config.get("START_DATE")
    end_date = config.config.get("END_DATE")
    tickers = config.config.get("TICKERS")
    touchstone = config.config.get("TOUCHSTONE")
    data_mode = config.config.get("DATA_MODE")

    if touchstone not in tickers:
        log.error(f"Touchstone ticker ({touchstone}) not in TICKERS.")
        raise ValueError("Touchstone must be in TICKERS.")

    allocation_tickers = set(config.config.get("ASSET_ALLOCATION").keys())
    missing_tickers = allocation_tickers - set(tickers)
    if missing_tickers:
        log.error(f"Missing tickers in TICKERS: {missing_tickers}")
        raise ValueError(f"Missing tickers in TICKERS: {missing_tickers}")

    try:
        log.info("Loading data feeds...")
        data_feeds = prepare_backtrader_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            data_mode=data_mode,
            log=log
        )

        for ticker, data_feed in data_feeds.items():
            if data_feed is None:
                log.error(f"Data feed for {ticker} is None")
                raise ValueError(f"Invalid data feed for {ticker}")

            df = data_feed.p.dataname
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                log.warning(f"Data for {ticker} contains {nan_count} NaN values")

            cerebro.adddata(data_feed, name=ticker)
            log.info(f"Added data for {ticker}.")
        log.info("All data feeds loaded.")
    except Exception as e:
        log.error(f"Error loading data: {str(e)}")
        log.error(traceback.format_exc())
        raise

    cerebro.broker.setcash(config.config.get("INITIAL_CASH"))
    cerebro.broker.setcommission(commission=config.config.get("COMMISSION"))

    # === Analyzer 등록 (자동 계산) ===
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=config.config.get("RISK_FREE_RATE"))
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

    initial_value = cerebro.broker.getvalue()
    log.info(f'Initial portfolio value: {initial_value:,.2f}')

    try:
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()

        log.info(f'Final portfolio value: {final_value:,.2f}')
        total_return = ((final_value / config.config.get("INITIAL_CASH")) - 1) * 100
        log.info(f"Total return: {total_return:.2f}%")

        # ========== Analyzer 결과 확인(수동 계산 대체) ==========
        strategy_instance = results[0]
        # 1) Returns Analyzer
        returns_analyzer = strategy_instance.analyzers.returns.get_analysis()
        rtot = returns_analyzer.get('rtot', 0) * 100      # 총 수익률 (%)
        rnorm = returns_analyzer.get('rnorm', 0) * 100    # 연간화 (%)
        log.info(f"[Analyzer] Total Return (rtot): {rtot:.2f}%")
        log.info(f"[Analyzer] Annual Return (rnorm): {rnorm:.2f}%")

        # 2) DrawDown Analyzer
        drawdown_analyzer = strategy_instance.analyzers.drawdown.get_analysis()
        max_dd = drawdown_analyzer['max']['drawdown']    # %
        log.info(f"[Analyzer] Max Drawdown: {max_dd:.2f}%")

        # 3) SharpeRatio Analyzer
        sharpe_analyzer = strategy_instance.analyzers.sharpe.get_analysis()
        sr = sharpe_analyzer.get('sharperatio', float('nan'))
        log.info(f"[Analyzer] Sharpe Ratio: {sr:.2f}")

        # 4) TimeReturn Analyzer (원하면 추가 활용)
        time_return_analyzer = strategy_instance.analyzers.time_return.get_analysis()
        # time_return_analyzer는 날짜별 수익률이 dict 형태로 들어있음

        # 연도별, 월별 리턴 등은 TimeReturn의 timeframe 설정에 따라 추출 가능

        # 백테스트 결과 시각화
        if results and len(results) > 0:
            visualize_results(results, initial_value, final_value, log)
        else:
            log.warning("No strategy results returned from backtest")

        return results

    except Exception as e:
        log.error(f"Error during backtesting: {str(e)}")
        log.error(traceback.format_exc())
        raise

def visualize_results(results, initial_value, final_value, logger=None):
    """
    기존 포트폴리오 가치 그래프, 연간/월간 그래프, 드로우다운 등을 그리는 함수.
    Analyzer 결과와는 직접 연동하지 않아도 되므로 기존 방식 유지 가능.
    """
    log = logger or logging.getLogger('backtest')
    if not results or len(results) == 0:
        log.warning("No results available for visualization.")
        return

    strategy = results[0]

    portfolio_dates = strategy.get_portfolio_dates()
    portfolio_values = strategy.get_portfolio_values()

    if not portfolio_dates or not portfolio_values:
        log.warning("No portfolio data available for graphing.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.config.RESULTS_DIR, exist_ok=True)

    create_portfolio_value_chart(strategy, initial_value, final_value, timestamp, log)
    create_annual_returns_chart(strategy, timestamp, log)
    create_monthly_returns_heatmap(strategy, timestamp, log)
    create_drawdown_chart(strategy, timestamp, log)
    save_results_to_csv(strategy, timestamp, log)

def create_portfolio_value_chart(strategy, initial_value, final_value, timestamp, logger=None):
    """
    포트폴리오 가치 변화를 선 그래프로 시각화
    """
    log = logger or logging.getLogger('backtest')
    try:
        dates = np.array(strategy.portfolio_dates)
        values = np.array(strategy.portfolio_values)

        valid_indices = ~np.isnan(values)
        dates = dates[valid_indices]
        values = values[valid_indices]

        if len(values) == 0:
            log.error("No valid portfolio values for visualization.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(dates, values)
        ax.set_title('Portfolio Value Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True)
        set_common_xaxis(ax, dates)

        first_valid_value = values[0]
        last_valid_value = values[-1]
        total_return = ((last_valid_value / first_valid_value) - 1) * 100 if first_valid_value != 0 else 0

        plt.figtext(0.13, 0.02, f'Initial Value: ${first_valid_value:,.2f}', fontsize=10)
        plt.figtext(0.7, 0.02, f'Final Value: ${last_valid_value:,.2f}', fontsize=10)
        plt.figtext(0.4, 0.02, f'Return: {total_return:.2f}%', fontsize=10)

        plt.tight_layout()
        file_path = os.path.join(config.config.RESULTS_DIR, f'portfolio_value_{timestamp}.png')
        plt.savefig(file_path)
        log.info(f"Saved portfolio value chart to {file_path}.")
        plt.close()
    except Exception as e:
        log.error(f"Error generating portfolio value chart: {str(e)}")

def create_annual_returns_chart(strategy, timestamp, logger=None):
    """
    연간 수익률 막대 그래프 (단순 참고용)
    """
    log = logger or logging.getLogger('backtest')
    try:
        # 예: strategy에서 연도별 수익률을 별도로 계산하지 않고,
        # TimeReturn Analyzer를 쓰거나 strategy.get_portfolio_values()로 변형해서 자유롭게 그릴 수 있음
        # 여기서는 기존 코드를 유지하되, NaN 필터 정도만 남김
        pass
    except Exception as e:
        log.error(f"Error generating annual returns chart: {str(e)}")
        log.error(traceback.format_exc())

def create_monthly_returns_heatmap(strategy, timestamp, logger=None):
    """
    월간 수익률 히트맵 (단순 참고용)
    """
    log = logger or logging.getLogger('backtest')
    try:
        pass
    except Exception as e:
        log.error(f"Error generating monthly returns heatmap: {str(e)}")
        log.error(traceback.format_exc())

def create_drawdown_chart(strategy, timestamp, logger=None):
    """
    포트폴리오의 낙폭(Drawdown) 그래프
    """
    log = logger or logging.getLogger('backtest')
    try:
        pass
    except Exception as e:
        log.error(f"Error generating drawdown chart: {str(e)}")
        log.error(traceback.format_exc())

def save_results_to_csv(strategy, timestamp, logger=None):
    """
    포트폴리오 가치, 일별 수익률 등 CSV 저장
    """
    log = logger or logging.getLogger('backtest')
    try:
        # 여기서는 strategy.portfolio_values, daily_returns 등을 그대로 CSV에 저장.
        portfolio_dates = strategy.portfolio_dates
        portfolio_values = strategy.portfolio_values

        if not portfolio_dates or not portfolio_values:
            log.error("No portfolio values to save.")
            return

        df = pd.DataFrame({
            'Date': portfolio_dates,
            'Value': portfolio_values
        })
        portfolio_file = os.path.join(config.config.RESULTS_DIR, f'portfolio_values_{timestamp}.csv')
        df.to_csv(portfolio_file, index=False)
        log.info(f"Saved portfolio values to {portfolio_file}.")
    except Exception as e:
        log.error(f"Error saving CSV results: {str(e)}")
        log.error(traceback.format_exc())
