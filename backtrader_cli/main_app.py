import os
import sys
import logging
import traceback
import argparse
import numpy as np
from datetime import datetime
import config
from backtest import run_backtest

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

        # 백테스트 실행
        results = run_backtest(config_file=args.config, logger=logger)
        logger.info("Backtest completed.")

        # Analyzer 결과를 콘솔에 추가 출력 (원하면 중복일 수도 있음)
        if results and len(results) > 0:
            strategy = results[0]
            analysis = strategy.analyzers

            returns = analysis.returns.get_analysis()
            total_return_pct = returns.get('rtot', 0) * 100
            annual_return_pct = returns.get('rnorm', 0) * 100

            drawdown = analysis.drawdown.get_analysis()
            max_drawdown_pct = drawdown['max']['drawdown']

            sharpe = analysis.sharpe.get_analysis()
            sharpe_ratio = sharpe.get('sharperatio', float('nan'))

            final_port_value = strategy.broker.getvalue()
            pnl = final_port_value - config.config.get("INITIAL_CASH")

            print("\n===== Analyzer Results =====")
            print(f"Final Portfolio Value: {final_port_value:,.2f}")
            print(f"Total Return (rtot): {total_return_pct:.2f}%")
            print(f"Annual Return (rnorm): {annual_return_pct:.2f}%")
            print(f"Max Drawdown: {max_drawdown_pct:.2f}%")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Net Profit: {pnl:,.2f}")
        else:
            print("No results to analyze.")

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
