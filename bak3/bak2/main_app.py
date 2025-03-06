# main_app.py
# 백테스팅 애플리케이션의 메인 실행 파일입니다.
# 기본 설정으로 실행
# python main.py

# # 사용자 정의 설정으로 실행
# python main.py --config my_config.yaml

# 명령행으로 설정 변경
# python main.py --start 2010-01-01 --end 2022-12-31 --tickers SPY,QQQ,AGG,GLD --cash 200000 

# 온라인 모드 및 소수점 주식 거래 허용
# python main.py --online --fractional

import os
import sys
import logging
import traceback
import argparse
from datetime import datetime

import config
from backtest import run_backtest, save_results_to_csv


def setup_environment(config_file=None):
    """
    로깅 및 데이터 디렉터리 설정을 초기화합니다.
    
    Args:
        config_file: 설정 파일 경로 (None이면 기본 설정 사용)
        
    Returns:
        로거 객체
    """
    try:
        # 설정 로드
        config.load_config(config_file)
        
        # 필수 디렉토리 생성
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        # 로거 설정
        logger = config.setup_logging()
        
        # 설정 정보 로깅
        config.log_config(logger)
        
        return logger
        
    except PermissionError:
        print(f"폴더 생성 권한이 없습니다: {config.LOG_DIR} 또는 {config.DATA_DIR}")
        sys.exit(1)
    except Exception as e:
        print(f"환경 설정 중 오류 발생: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


def parse_arguments():
    """
    명령행 인자를 파싱합니다.
    
    Returns:
        파싱된 인자 객체
    """
    parser = argparse.ArgumentParser(description='자산 배분 백테스팅 시스템')
    
    parser.add_argument('--config', type=str, help='설정 파일 경로 (YAML)')
    parser.add_argument('--start', type=str, help='백테스트 시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='백테스트 종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--tickers', type=str, help='티커 목록 (쉼표로 구분)')
    parser.add_argument('--cash', type=float, help='초기 자본금')
    parser.add_argument('--online', action='store_true', help='온라인 모드 사용 (강제 데이터 다운로드)')
    parser.add_argument('--no-regime', action='store_true', help='시장 상태 감지 비활성화')
    parser.add_argument('--fractional', action='store_true', help='소수점 주식 거래 허용')
    
    return parser.parse_args()


def apply_arguments(args):
    """
    명령행 인자를 설정에 적용합니다.
    
    Args:
        args: 파싱된 인자 객체
    """
    if args.start:
        config.set_config('START_DATE', args.start)
    
    if args.end:
        config.set_config('END_DATE', args.end)
    
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
        config.set_config('TICKERS', tickers)
    
    if args.cash:
        config.set_config('INITIAL_CASH', args.cash)
    
    if args.online:
        config.set_config('DATA_MODE', 'online')
    
    if args.no_regime:
        config.set_config('USE_MARKET_REGIME', False)
    
    if args.fractional:
        config.set_config('FRACTIONAL_SHARES', True)


def main():
    """
    메인 실행 함수입니다.
    """
    # 실행 시간 기록
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    # 명령행 인자 파싱
    args = parse_arguments()
    
    try:
        # 환경 설정 및 로거 초기화
        logger = setup_environment(args.config)
        
        # 명령행 인자 적용
        apply_arguments(args)
        
        # 백테스트 실행
        logger.info("백테스트 시작")
        results = run_backtest(config_file=args.config, logger=logger)
        logger.info("백테스트 완료")
        
        # 결과 CSV 저장
        if results and len(results) > 0:
            strategy = results[0]
            save_results_to_csv(strategy=strategy, timestamp=timestamp, logger=logger)
        
        # 실행 시간 계산
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        
        # 성공 메시지 출력
        logger.info(f"전체 실행 시간: {elapsed_time.total_seconds():.2f}초")
        
        print("\n================================================")
        print("           백테스트 실행이 완료되었습니다.        ")
        print("           결과 그래프가 저장되었습니다.         ")
        print(f"           실행 시간: {elapsed_time.total_seconds():.2f}초")
        print("================================================\n")
        
        if results and len(results) > 0:
            strategy = results[0]
            # 주요 지표 출력
            total_return = ((strategy.portfolio_values[-1] / strategy.portfolio_values[0]) - 1) * 100
            print(f"총 수익률: {total_return:.2f}%")
            
            # 연간 지표 계산
            annual_metrics = strategy.compute_annual_metrics()
            if annual_metrics:
                # 마지막 연도의 수익률 출력
                last_year = max(annual_metrics.keys())
                print(f"{last_year}년 수익률: {annual_metrics[last_year]['Return']:.2f}%")
            
            print("\n결과 파일은 다음 위치에 저장되었습니다:")
            print(f"{config.RESULTS_DIR}\n")
        
    except Exception as e:
        logging.error(f"백테스트 실행 실패: {str(e)}")
        logging.error(traceback.format_exc())
        
        # 오류 메시지 출력
        print("\n================================================")
        print("             백테스트 실행 실패!                ")
        print(f"오류: {str(e)}")
        print("로그 파일을 확인하세요: ", os.path.join(config.LOG_DIR, "backtest_*.log"))
        print("================================================\n")
        
        sys.exit(1)


if __name__ == "__main__":
    main()