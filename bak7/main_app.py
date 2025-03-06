# main_app.py
import os
import sys
import logging
import traceback
import argparse
import numpy as np
from datetime import datetime
import config
from backtest import run_backtest, save_results_to_csv

def setup_environment(config_file=None):
    try:
        config.config.load_config(config_file)
        os.makedirs(config.config.DATA_DIR, exist_ok=True)
        os.makedirs(config.config.LOG_DIR, exist_ok=True)
        os.makedirs(config.config.RESULTS_DIR, exist_ok=True)
        logger = config.config.setup_logging()
        config.config.log_config(logger)
        return logger
    except PermissionError:
        print(f"Permission denied for directories: {config.config.LOG_DIR} or {config.config.DATA_DIR}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during environment setup: {e}")
        traceback.print_exc()
        sys.exit(1)

def validate_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Asset Allocation Backtesting System')
    parser.add_argument('--config', type=str, help='Path to config file (YAML)')
    parser.add_argument('--start', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--tickers', type=str, help='Ticker list (comma-separated)')
    parser.add_argument('--cash', type=float, help='Initial cash')
    parser.add_argument('--online', action='store_true', help='Use online mode (force data download)')
    parser.add_argument('--no-regime', action='store_true', help='Disable market regime detection')
    parser.add_argument('--fractional', action='store_true', help='Allow fractional shares')
    return parser.parse_args()

def apply_arguments(args):
    if args.start:
        if not validate_date(args.start):
            raise ValueError("Start date must be in YYYY-MM-DD format.")
        config.config.set("START_DATE", args.start)
    if args.end:
        if not validate_date(args.end):
            raise ValueError("End date must be in YYYY-MM-DD format.")
        config.config.set("END_DATE", args.end)
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
        config.config.set("TICKERS", tickers)
    if args.cash:
        config.config.set("INITIAL_CASH", args.cash)
    if args.online:
        config.config.set("DATA_MODE", "online")
    if args.no_regime:
        config.config.set("USE_MARKET_REGIME", False)
    if args.fractional:
        config.config.set("FRACTIONAL_SHARES", True)

def main():
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    args = parse_arguments()
    try:
        logger = setup_environment(args.config)
        apply_arguments(args)
        logger.info("Starting backtest...")
        results = run_backtest(config_file=args.config, logger=logger)
        logger.info("Backtest completed.")
        if results and len(results) > 0:
            strategy = results[0]
            # CSV 저장 (기존 기능)
            save_results_to_csv(strategy, timestamp, logger)
            
            # 최종 메트릭 및 연도별 메트릭 계산 후 출력
            final_metrics = strategy.compute_final_metrics()
            annual_metrics = strategy.compute_annual_metrics()
            
            print("\n===== 최종 결과 =====")
            print(f"총자산: {final_metrics.get('TotalAsset', np.nan):,.2f}")
            print(f"총이익: {final_metrics.get('TotalProfit', np.nan):,.2f}")
            print(f"CAGR: {final_metrics.get('CAGR', np.nan):.2f}%")
            print(f"MDD: {final_metrics.get('MDD', np.nan):.2f}%")
            print(f"Sharpe: {final_metrics.get('Sharpe', np.nan):.2f}")
            print(f"Sortino: {final_metrics.get('Sortino', np.nan):.2f}")
            print(f"표준편차: {final_metrics.get('StdDev', np.nan):.2f}")
            
            print("\n===== 연도별 결과 =====")
            for year in sorted(annual_metrics.keys()):
                metrics = annual_metrics[year]
                print(f"\n[{year}]")
                print(f"총자산: {metrics.get('TotalAsset', np.nan):,.2f}")
                print(f"총이익: {metrics.get('TotalProfit', np.nan):,.2f}")
                print(f"CAGR: {metrics.get('CAGR', np.nan):.2f}%")
                print(f"MDD: {metrics.get('MDD', np.nan):.2f}%")
                print(f"Sharpe: {metrics.get('Sharpe', np.nan):.2f}")
                print(f"Sortino: {metrics.get('Sortino', np.nan):.2f}")
                print(f"표준편차: {metrics.get('StdDev', np.nan):.2f}")
            
            # 추가: 포트폴리오 전체 수익률도 출력 (기존 코드 참조)
            portfolio_values = strategy.get_portfolio_values()
            if portfolio_values and len(portfolio_values) > 1:
                valid_values = [v for v in portfolio_values if not np.isnan(v) and v > 0]
                if len(valid_values) >= 2:
                    total_return = ((valid_values[-1] / valid_values[0]) - 1) * 100
                    print(f"\n전체 수익률: {total_return:.2f}%")
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"Total execution time: {elapsed:.2f} seconds")
        print("\n================================================")
        print("       Backtest execution completed.          ")
        print(f"       Execution time: {elapsed:.2f} seconds")
        print("       Results saved in:", config.config.RESULTS_DIR)
        print("================================================\n")
    except Exception as e:
        logging.error(f"Backtest execution failed: {e}")
        logging.error(traceback.format_exc())
        print("\n================================================")
        print("           Backtest execution failed!           ")
        print(f"Error: {e}")
        print("Check log file in:", os.path.join(config.config.LOG_DIR, "backtest_*.log"))
        print("================================================\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
