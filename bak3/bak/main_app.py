# main_app.py
# 백테스팅 애플리케이션의 메인 실행 파일입니다.

import os
import logging
import traceback
import config
from backtest import run_backtest

# ------------------------------
# 기본 설정 로드 및 폴더/로그 설정
# ------------------------------
def setup_environment():
    """
    로깅 및 데이터 디렉터리 설정을 초기화합니다.
    """
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.DATA_DIR, exist_ok=True)
    except PermissionError:
        print(f"폴더 생성 권한이 없습니다: {config.LOG_DIR} 또는 {config.DATA_DIR}")
        raise
    except Exception as e:
        print(f"폴더 생성 중 오류 발생: {str(e)}")
        raise

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE, mode='a', encoding='utf-8'),  # 'w'에서 'a'로 변경하여 로그 누적
            logging.StreamHandler()
        ]
    )

    logging.info(f"데이터 모드 설정: {config.DATA_MODE.upper()}")

# main_app.py
# 백테스팅 애플리케이션의 메인 실행 파일입니다.

import os
import logging
import traceback
import config
from backtest import run_backtest

# ------------------------------
# 기본 설정 로드 및 폴더/로그 설정
# ------------------------------
def setup_environment():
    """
    로깅 및 데이터 디렉터리 설정을 초기화합니다.
    """
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.DATA_DIR, exist_ok=True)
    except PermissionError:
        print(f"폴더 생성 권한이 없습니다: {config.LOG_DIR} 또는 {config.DATA_DIR}")
        raise
    except Exception as e:
        print(f"폴더 생성 중 오류 발생: {str(e)}")
        raise

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE, mode='a', encoding='utf-8'),  # 로그 누적
            logging.StreamHandler()
        ]
    )

    logging.info(f"데이터 모드 설정: {config.DATA_MODE.upper()}")
    
    # config 설정 로깅
    logging.info("=============== 설정 정보 ===============")
    logging.info(f"기간: {config.START_DATE} ~ {config.END_DATE}")
    logging.info(f"티커: {config.TICKERS}")
    logging.info(f"기준 티커: {config.TOUCHSTONE}")
    logging.info(f"기본 자산 배분: {config.ASSET_ALLOCATION}")
    logging.info(f"공격적 자산 배분: {config.AGGRESSIVE_ALLOCATION}")
    logging.info(f"중립적 자산 배분: {config.MODERATE_ALLOCATION}")
    logging.info(f"방어적 자산 배분: {config.DEFENSIVE_ALLOCATION}")
    logging.info(f"이동평균 기간: {config.MA_PERIODS}")
    logging.info(f"초기 자본: {config.INITIAL_CASH:,.2f}")
    logging.info(f"수수료: {config.COMMISSION * 100:.2f}%")
    logging.info(f"시장 상태 활용: {config.USE_MARKET_REGIME}")
    logging.info(f"소수점 주식 거래: {getattr(config, 'FRACTIONAL_SHARES', False)}")
    logging.info("=========================================")

# ------------------------------
# 메인 실행 코드
# ------------------------------
if __name__ == "__main__":
    try:
        # 환경 설정
        setup_environment()
        
        # 백테스트 실행
        logging.info("백테스트 시작")
        results = run_backtest()
        logging.info("백테스트 완료")
        
        # 성공 메시지 출력
        print("\n================================================")
        print("           백테스트 실행이 완료되었습니다.        ")
        print("           결과 그래프가 저장되었습니다.         ")
        print("================================================\n")
        
    except Exception as e:
        logging.error(f"백테스트 실행 실패: {str(e)}")
        logging.error(traceback.format_exc())
        
        # 오류 메시지 출력
        print("\n================================================")
        print("             백테스트 실행 실패!                ")
        print(f"오류: {str(e)}")
        print("로그 파일을 확인하세요: ", config.LOG_FILE)
        print("================================================\n")