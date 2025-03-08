import os
import logging
import datetime
import traceback
import backtrader as bt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
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

        # ========== Analyzer 결과 확인 =====================
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
            # (1) 연도별 스탯 CSV
            save_annual_stats_to_csv(strategy_instance, timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), logger=log)

            # (2) 리밸런싱 이력 CSV
            save_rebalance_history_to_csv(strategy_instance, timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), logger=log)
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


def save_rebalance_history_to_csv(strategy, timestamp, logger=None):
    """
    Strategy에 저장된 리밸런싱 이력(rebalance_history)을 CSV로 저장
    """
    log = logger or logging.getLogger('backtest')
    try:
        records = strategy.rebalance_history
        if not records:
            log.info("No rebalance history to save.")
            return

        # 각 이벤트별로 trades(매수/매도)를 펼쳐서 저장하거나, 
        # 혹은 JSON 형태 그대로 저장할 수도 있음
        rows = []
        for event in records:
            date = event['date']
            spy_close = event['spy_close']
            ma21 = event['MA_21']
            ma50 = event['MA_50']
            ma150 = event['MA_150']
            ma200 = event['MA_200']
            target_alloc = event['target_allocation']  # dict
            trades = event['trades']  # list of dict

            for trade in trades:
                rows.append({
                    'date': date,
                    'spy_close': spy_close,
                    'MA_21': ma21,
                    'MA_50': ma50,
                    'MA_150': ma150,
                    'MA_200': ma200,
                    'asset': trade['asset'],
                    'action': trade['action'],
                    'shares': trade['shares'],
                    'target_allocation': str(target_alloc)  # 문자열로 저장
                })

        df_rebal = pd.DataFrame(rows)
        out_path = os.path.join(config.config.RESULTS_DIR, f"rebalance_history_{timestamp}.csv")
        df_rebal.to_csv(out_path, index=False)
        log.info(f"Saved rebalance history to {out_path}.")
    except Exception as e:
        log.error(f"Error saving rebalance history CSV: {e}")
        log.error(traceback.format_exc())



def save_annual_stats_to_csv(strategy, timestamp, logger=None):
    """
    연도별로 CAGR, MDD, Sharpe, Sortino, 표준편차 등을 계산해 CSV로 저장.
    """
    log = logger or logging.getLogger('backtest')

    try:
        # (1) daily return 데이터프레임 생성
        time_return_analyzer = strategy.analyzers.time_return.get_analysis()  # {datetime: daily_return}
        if not time_return_analyzer:
            log.warning("No time_return analyzer data found.")
            return

        # dict -> DataFrame
        df_ret = pd.DataFrame(
            [(dt, rtn) for dt, rtn in time_return_analyzer.items()],
            columns=['date', 'daily_return']
        )
        df_ret.sort_values('date', inplace=True)
        df_ret.reset_index(drop=True, inplace=True)
        df_ret['date'] = pd.to_datetime(df_ret['date'])
        df_ret['year'] = df_ret['date'].dt.year

        # (2) 연도별 지표 계산
        # Sharpe, Sortino, 표준편차는 일간수익률 기반으로 연간화(annualized)
        # MDD는 해당 연도 구간 내 최대낙폭
        # CAGR = (연말 누적수익률) ^(1/연수) - 1
        results = []

        # 전체 수익 곡선을 사용해서 년도별 MDD를 구할 수도 있으나,
        # 여기서는 "연도별로 따로 초기화해서" 최대 낙폭 측정하는 예시
        grouped = df_ret.groupby('year')

        for year, grp in grouped:
            # (2-1) 해당 연도 데이터
            dr = grp['daily_return'].values  # numpy array
            if len(dr) == 0:
                continue

            # (2-2) 누적수익률 계산
            # 1일차를 1.0으로 두고, 다음날 (1 + daily_return) 누적
            # => 연말 누적수익률(cum_prod - 1)
            cum_ret = (1 + dr).cumprod()
            final_cum_ret = cum_ret[-1] - 1.0  # 해당 연도의 마지막 값(0.x 형태)

            # (2-3) MDD 계산
            # 연도 내 최대고점 대비 하락폭
            running_max = np.maximum.accumulate(cum_ret)
            drawdown = (cum_ret - running_max) / running_max
            mdd = drawdown.min() * 100  # % 표현

            # (2-4) CAGR
            # 단순히 1년치면 final_cum_ret이 곧 1년 수익률이지만,
            # 기간이 정확히 1년이 아닐 수 있으므로 '연 단위' 보정 가능(이 예시는 단순 처리)
            # 일 수 기준으로 연환산하는 방식 (252거래일 가정)
            num_days = len(dr)
            cagr = (1 + final_cum_ret)**(252/num_days) - 1  # 대략 거래일 기준

            # (2-5) Sharpe Ratio, Sortino Ratio
            # 예: (mean(dr) - risk_free/252) / stdev(dr) * sqrt(252)
            # Sortino는 '음수 수익률'만 stdev 계산
            daily_rf = strategy.p.riskfreerate / 252.0 if hasattr(strategy.p, 'riskfreerate') else 0
            excess_return = dr - daily_rf

            mean_excess_return = np.mean(excess_return)
            std_excess_return = np.std(excess_return, ddof=1)  # 샘플표준편차
            sharpe_annual = 0.0
            if std_excess_return > 0:
                sharpe_annual = (mean_excess_return * 252) / (std_excess_return * math.sqrt(252))

            # Sortino: 음수 수익률 표준편차만 사용
            negative_returns = excess_return[excess_return < 0]
            downside_std = np.std(negative_returns, ddof=1)
            sortino_annual = 0.0
            if len(negative_returns) > 0 and downside_std > 0:
                sortino_annual = (mean_excess_return * 252) / (downside_std * math.sqrt(252))

            # (2-6) 표준편차(연율화)
            std_annual = std_excess_return * math.sqrt(252)

            # (2-7) 연말 자산가치 / 이익
            # 백테스트 중 실제 브로커의 '년초 자산가치'를 추적하려면
            # year가 바뀔 때의 포트 가치나, year 말 포트 가치를 기록해야 함.
            # 여기서는 daily_return만으로 단순히 "가정 초 기초자산=1"이라 보고 곱셈.
            # "실제 현금"은 strategy.broker.getvalue()를 날짜별로 기록해둔 뒤, 
            # groupby('year') 해서 마지막 값을 가져오는 방식이 필요함.
            # 예: 아래는 단순히 초깃값 1로 가정하는 예시
            end_value = cum_ret[-1]  # 1기준
            profit = end_value - 1.0      # 1기준 -> profit

            results.append({
                'year': year,
                'final_asset_ratio': f"{end_value:.4f}",  # (초기=1 대비)
                'profit_ratio': f"{profit:.4f}",
                'CAGR': f"{cagr*100:.2f}",
                'MDD': f"{mdd:.2f}",
                'Sharpe': f"{sharpe_annual:.2f}",
                'Sortino': f"{sortino_annual:.2f}",
                'Std_Dev': f"{std_annual:.2f}",
            })

        df_yearly = pd.DataFrame(results)
        if len(df_yearly) == 0:
            log.warning("No annual stats to save.")
            return

        annual_stats_path = os.path.join(
            config.config.RESULTS_DIR,
            f"annual_stats_{timestamp}.csv"
        )
        df_yearly.to_csv(annual_stats_path, index=False)
        log.info(f"Saved annual stats to {annual_stats_path}.")
    except Exception as e:
        log.error(f"Error saving annual stats: {e}")
        log.error(traceback.format_exc())