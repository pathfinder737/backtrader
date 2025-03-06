#backtest.py
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
from strategy import MonthlyRebalanceStrategy

# 공통 x축 설정 함수: 날짜 리스트에서 약 10개 눈금을 균등하게 표시합니다.
def set_common_xaxis(ax, dates):
    n = len(dates)
    step = max(1, n // 10)
    ax.set_xticks(dates[::step])
    ax.set_xticklabels([d.strftime('%Y-%m') for d in dates[::step]], rotation=45)

def run_backtest(config_file: Optional[str] = None, logger: Optional[logging.Logger] = None):
    """
    백테스트 실행 함수.
    설정 파일 로드, 데이터 피드 추가, Cerebro 구성, 분석기 등록 및 결과 시각화를 수행합니다.
    
    :param config_file: 사용자 정의 설정 파일 경로 (없으면 기본 설정 사용)
    :param logger: 로깅에 사용할 로거 (없으면 기본 로거 사용)
    :return: 백테스트 결과 (전략 인스턴스 리스트)
    """
    log = logger or logging.getLogger('backtest')
    
    if config_file:
        config.config.load_config(config_file)
    
    validation = config.config.validate()
    if not validation.get('is_valid', False):
        log.error("Config validation failed.")
        raise ValueError("Invalid configuration. Check logs for details.")
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MonthlyRebalanceStrategy)
    
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
            cerebro.adddata(data_feed, name=ticker)
            log.info(f"Added data for {ticker}.")
        log.info("All data feeds loaded.")
    except Exception as e:
        log.error(f"Error loading data: {str(e)}")
        log.error(traceback.format_exc())
        raise
    
    cerebro.broker.setcash(config.config.get("INITIAL_CASH"))
    cerebro.broker.setcommission(commission=config.config.get("COMMISSION"))
    
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
        visualize_results(results, initial_value, final_value, log)
        return results
    except Exception as e:
        log.error(f"Error during backtesting: {str(e)}")
        log.error(traceback.format_exc())
        raise

def visualize_results(results, initial_value, final_value, logger=None):
    """
    백테스트 결과를 여러 차트로 시각화하고 저장합니다.
    
    :param results: 전략 인스턴스 리스트 (백테스트 결과)
    :param initial_value: 초기 포트폴리오 가치
    :param final_value: 최종 포트폴리오 가치
    :param logger: 로깅에 사용할 로거
    """
    log = logger or logging.getLogger('backtest')
    if not results or len(results) == 0:
        log.warning("No results available for visualization.")
        return
    strategy = results[0]
    if not hasattr(strategy, 'portfolio_dates') or not strategy.portfolio_dates:
        log.warning("No portfolio dates available for graphing.")
        return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.config.RESULTS_DIR, exist_ok=True)
    
    create_portfolio_value_chart(strategy, initial_value, final_value, timestamp, log)
    create_annual_returns_chart(strategy, timestamp, log)
    create_monthly_returns_heatmap(strategy, timestamp, log)
    create_drawdown_chart(strategy, timestamp, log)

def create_portfolio_value_chart(strategy, initial_value, final_value, timestamp, logger=None):
    """
    포트폴리오 가치 변화를 선 그래프로 시각화하고 저장합니다.
    
    :param strategy: 전략 인스턴스 (포트폴리오 가치, 날짜 정보 포함)
    :param initial_value: 초기 포트폴리오 가치
    :param final_value: 최종 포트폴리오 가치
    :param timestamp: 파일명에 사용할 타임스탬프 문자열
    :param logger: 로깅에 사용할 로거
    """
    log = logger or logging.getLogger('backtest')
    try:
        dates = strategy.portfolio_dates
        values = strategy.portfolio_values
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(dates, values)
        ax.set_title('Portfolio Value Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True)
        set_common_xaxis(ax, dates)
        plt.figtext(0.13, 0.02, f'Initial Value: ${initial_value:,.2f}', fontsize=10)
        plt.figtext(0.7, 0.02, f'Final Value: ${final_value:,.2f}', fontsize=10)
        plt.figtext(0.4, 0.02, f'Return: {((final_value/initial_value)-1)*100:.2f}%', fontsize=10)
        plt.tight_layout()
        file_path = os.path.join(config.config.RESULTS_DIR, f'portfolio_value_{timestamp}.png')
        plt.savefig(file_path)
        log.info(f"Saved portfolio value chart to {file_path}.")
        plt.close()
    except Exception as e:
        log.error(f"Error generating portfolio value chart: {str(e)}")

def create_annual_returns_chart(strategy, timestamp, logger=None):
    """
    연간 수익률을 막대 그래프로 시각화하고 저장합니다.
    
    :param strategy: 전략 인스턴스 (연간 성과 계산 가능)
    :param timestamp: 파일명에 사용할 타임스탬프 문자열
    :param logger: 로깅에 사용할 로거
    """
    log = logger or logging.getLogger('backtest')
    try:
        annual_metrics = strategy.compute_annual_metrics()
        if not annual_metrics:
            log.warning("No annual returns data available.")
            return
        years = sorted(annual_metrics.keys())
        returns = [annual_metrics[year]['Return'] for year in years]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(years, returns)
        for i, bar in enumerate(bars):
            bar.set_color('blue' if returns[i] >= 0 else 'red')
            ax.text(i, returns[i] + (5 if returns[i] >= 0 else -5),
                    f'{returns[i]:.1f}%', ha='center',
                    va='bottom' if returns[i] >= 0 else 'top', fontsize=9)
        ax.set_title('Annual Returns')
        ax.set_xlabel('Year')
        ax.set_ylabel('Return (%)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        file_path = os.path.join(config.config.RESULTS_DIR, f'annual_returns_{timestamp}.png')
        plt.savefig(file_path)
        log.info(f"Saved annual returns chart to {file_path}.")
        plt.close()
    except Exception as e:
        log.error(f"Error generating annual returns chart: {str(e)}")

def create_monthly_returns_heatmap(strategy, timestamp, logger=None):
    """
    월간 수익률 히트맵을 생성하여 저장합니다.
    
    :param strategy: 전략 인스턴스 (포트폴리오 날짜, 가치 정보 포함)
    :param timestamp: 파일명에 사용할 타임스탬프 문자열
    :param logger: 로깅에 사용할 로거
    """
    log = logger or logging.getLogger('backtest')
    try:
        df = pd.DataFrame({
            'Date': strategy.portfolio_dates,
            'Value': strategy.portfolio_values
        })
        # 반드시 인덱스 이름을 "Date"로 설정 (merge_asof에서 필요)
        df.index.name = "Date"
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        monthly_returns = df['Value'].resample('ME').last().pct_change() * 100
        monthly_returns = monthly_returns.to_frame('Return')
        monthly_returns['Year'] = monthly_returns.index.year
        monthly_returns['Month'] = monthly_returns.index.month
        pivot_table = monthly_returns.pivot_table(values='Return', index='Year', columns='Month', aggfunc='mean')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[m - 1] for m in pivot_table.columns]
        fig, ax = plt.subplots(figsize=(12, 8))
        try:
            import seaborn as sns
            sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax)
        except ImportError:
            im = ax.imshow(pivot_table.values, cmap="RdYlGn")
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    value = pivot_table.values[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if abs(value) > 10 else 'black'
                        ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color)
            plt.colorbar(im, ax=ax)
            ax.set_yticks(range(len(pivot_table.index)))
            ax.set_yticklabels(pivot_table.index)
            ax.set_xticks(range(len(pivot_table.columns)))
            ax.set_xticklabels(pivot_table.columns)
        ax.set_title('Monthly Returns Heatmap (%)')
        plt.tight_layout()
        file_path = os.path.join(config.config.RESULTS_DIR, f'monthly_returns_heatmap_{timestamp}.png')
        plt.savefig(file_path)
        log.info(f"Saved monthly returns heatmap to {file_path}.")
        plt.close()
    except Exception as e:
        log.error(f"Error generating monthly returns heatmap: {str(e)}")

def create_drawdown_chart(strategy, timestamp, logger=None):
    """
    포트폴리오의 최대 낙폭(Drawdown)을 선 그래프로 시각화하고 저장합니다.
    
    :param strategy: 전략 인스턴스 (포트폴리오 가치 및 날짜 정보 포함)
    :param timestamp: 파일명에 사용할 타임스탬프 문자열
    :param logger: 로깅에 사용할 로거
    """
    log = logger or logging.getLogger('backtest')
    try:
        values = np.array(strategy.portfolio_values)
        dates = strategy.portfolio_dates
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(dates, drawdown)
        ax.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        ax.set_title('Portfolio Drawdown')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        set_common_xaxis(ax, dates)
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        max_dd_date = dates[max_dd_idx].strftime('%Y-%m-%d')
        ax.axhline(y=max_dd, color='r', linestyle='--', alpha=0.7)
        ax.text(dates[len(dates)//2], max_dd * 1.1, f'Max Drawdown: {max_dd:.2f}% ({max_dd_date})', 
                ha='center', color='r')
        plt.tight_layout()
        file_path = os.path.join(config.config.RESULTS_DIR, f'drawdown_{timestamp}.png')
        plt.savefig(file_path)
        log.info(f"Saved drawdown chart to {file_path}.")
        plt.close()
    except Exception as e:
        log.error(f"Error generating drawdown chart: {str(e)}")

def save_results_to_csv(strategy, timestamp, logger=None):
    """
    백테스트 결과(포트폴리오 가치, 연간 성과, 일별 수익률)를 CSV 파일로 저장합니다.
    
    :param strategy: 전략 인스턴스 (성과 데이터 포함)
    :param timestamp: 파일명에 사용할 타임스탬프 문자열
    :param logger: 로깅에 사용할 로거
    """
    log = logger or logging.getLogger('backtest')
    try:
        portfolio_df = pd.DataFrame({
            'Date': strategy.portfolio_dates,
            'Value': strategy.portfolio_values
        })
        portfolio_file = os.path.join(config.config.RESULTS_DIR, f'portfolio_values_{timestamp}.csv')
        portfolio_df.to_csv(portfolio_file, index=False)
        log.info(f"Saved portfolio values to {portfolio_file}.")
        
        annual_metrics = strategy.compute_annual_metrics()
        if annual_metrics:
            annual_df = pd.DataFrame.from_dict(annual_metrics, orient='index')
            annual_file = os.path.join(config.config.RESULTS_DIR, f'annual_metrics_{timestamp}.csv')
            annual_df.to_csv(annual_file)
            log.info(f"Saved annual metrics to {annual_file}.")
        
        if hasattr(strategy, 'daily_returns') and strategy.daily_returns:
            daily_df = pd.DataFrame({
                'Date': strategy.portfolio_dates[1:],
                'Return': strategy.daily_returns
            })
            daily_file = os.path.join(config.config.RESULTS_DIR, f'daily_returns_{timestamp}.csv')
            daily_df.to_csv(daily_file, index=False)
            log.info(f"Saved daily returns to {daily_file}.")
    except Exception as e:
        log.error(f"Error saving CSV results: {str(e)}")
