#main_app.py
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
    """
    환경 설정 초기화.
    설정 파일 로드, 필요한 디렉토리 생성, 로깅 설정 및 설정 정보 로그 기록을 수행합니다.
    
    :param config_file: 사용자 정의 설정 파일 경로 (YAML)
    :return: 초기화된 로거 객체
    """
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
    """
    날짜 문자열이 'YYYY-MM-DD' 형식인지 검증합니다.
    
    :param date_str: 날짜 문자열
    :return: True이면 형식 맞음, 아니면 False
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def parse_arguments():
    """
    명령행 인자를 파싱합니다.
    
    :return: 파싱된 인자 객체
    """
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
    """
    파싱된 명령행 인자를 설정에 적용합니다.
    
    :param args: 파싱된 인자 객체
    """
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
    """
    백테스트 메인 실행 함수.
    환경 설정, 명령행 인자 적용, 백테스트 실행, 결과 저장 및 실행 시간 출력 등을 수행합니다.
    """
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
            save_results_to_csv(strategy, timestamp, logger)
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"Total execution time: {elapsed:.2f} seconds")
        print("\n================================================")
        print("       Backtest execution completed.          ")
        print(f"       Execution time: {elapsed:.2f} seconds")
        print("       Results saved in:", config.config.RESULTS_DIR)
        print("================================================\n")
        
        if results and len(results) > 0:
            strategy = results[0]
            # 포트폴리오 값 유효성 확인
            portfolio_values = strategy.get_portfolio_values()
            if portfolio_values and len(portfolio_values) > 1:
                # NaN 값 필터링
                valid_values = [v for v in portfolio_values if not np.isnan(v) and v > 0]
                if len(valid_values) >= 2:
                    total_return = ((valid_values[-1] / valid_values[0]) - 1) * 100
                    print(f"Total Return: {total_return:.2f}%")
                    annual_metrics = strategy.compute_annual_metrics()
                    if annual_metrics:
                        valid_years = [y for y in annual_metrics.keys() 
                                    if not np.isnan(annual_metrics[y]['Return'])]
                        if valid_years:
                            last_year = max(valid_years)
                            print(f"{last_year} Year Return: {annual_metrics[last_year]['Return']:.2f}%")
                else:
                    logger.warning("No valid portfolio values for return calculation")
            else:
                logger.warning("Insufficient portfolio data for total return calculation")

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
