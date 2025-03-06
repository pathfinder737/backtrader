# config.py
# 백테스팅 기본 설정 및 설정 관리 기능을 제공합니다.

import os
import logging
import datetime
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 전역 설정 변수
_CONFIG = None

# 기본 설정
DEFAULT_CONFIG = {
    # 데이터 처리 모드: 'online'이면 실시간 다운로드, 'offline'이면 무조건 다운로드 후 전처리하여 CSV 파일을 사용
    "DATA_MODE": os.environ.get("DATA_MODE", "offline"),
    
    # 데이터 기간 설정
    "START_DATE": os.environ.get("START_DATE", "2000-01-01"),
    "END_DATE": os.environ.get("END_DATE", "2024-12-31"),
    
    # 사용할 티커 목록 (TOUCHSTONE 포함되어야 함)
    "TICKERS": ["SPY", "QQQ", "IAU", "IEF"],  
    
    # 기본 자산 배분 설정 (SPY는 TOUCHSTONE으로만 사용되므로 포함하지 않음)
    "ASSET_ALLOCATION": {"QQQ": 0.5, "IAU": 0.3, "IEF": 0.2},
    
    # 시장 상태별 자산 배분 설정
    # MA21 > MA50 > MA200: 공격적 배분
    "AGGRESSIVE_ALLOCATION": {"QQQ": 0.8, "IAU": 0.2, "IEF": 0},
    # MA50 > MA200: 중립적 배분
    "MODERATE_ALLOCATION": {"QQQ": 0.7, "IAU": 0.2, "IEF": 0.1},
    # MA50 < MA200: 방어적 배분
    "DEFENSIVE_ALLOCATION": {"QQQ": 0, "IAU": 0.5, "IEF": 0.5},
    # 새로운 자산 배분 상태: 중간 방어적
    # MA150 > 현재가격 > MA200: 중간 방어적 배분
    "MID_DEFENSIVE_ALLOCATION": {"QQQ": 0.3, "IAU": 0.5, "IEF": 0.2},
    
    # 이동평균 계산 기준 티커
    "TOUCHSTONE": "SPY",
    
    # 이동평균 기간 설정 (추가된 150일 이평)
    "MA_PERIODS": [21, 50, 150, 200],  # 여러 이동평균 기간 정의
    
    # 주간 이동평균 기간 설정
    "WMA_PERIODS": [10, 30],  # 주간 이동평균 기간 정의
    
    # 이동평균 타입 설정 ('daily' 또는 'weekly')
    "MA_TYPE": "weekly",  # 기본값은 주간 이동평균
    
    # 초기 자본 및 수수료
    "INITIAL_CASH": 100000.0,
    "COMMISSION": 0.001,
    
    # 현금 버퍼 비율 (총 포트폴리오 가치의 %)
    "CASH_BUFFER_PERCENT": 1.0,
    
    # 주문 처리 설정
    "PROCESS_SELL_FIRST": True,  # 매도 주문 먼저 처리
    
    # 리스크 프리 레이트 (분석 지표 계산에 사용)
    "RISK_FREE_RATE": 0.01,
    
    # 소수점 주식 거래 허용 여부 (False: 정수 단위만 거래, True: 소수점 허용)
    "FRACTIONAL_SHARES": False,
    
    # 시장 상태에 따른 자산 배분 활용 여부
    "USE_MARKET_REGIME": True,
    
    # 로깅 레벨 설정
    "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
}


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    설정을 로드합니다. 파일이 제공되면 파일에서 로드하고, 
    그렇지 않으면 기본값을 사용합니다.
    
    Args:
        config_file: 설정 파일 경로 (YAML 형식)
        
    Returns:
        설정 딕셔너리
    """
    global _CONFIG
    
    # 기본 설정으로 초기화
    config = DEFAULT_CONFIG.copy()
    
    # 파일에서 설정 로드 (있는 경우)
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config.update(file_config)
        except Exception as e:
            print(f"설정 파일 로드 실패: {str(e)}")
    
    # 필수 디렉토리 생성
    for directory in [DATA_DIR, LOG_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    _CONFIG = config
    return config


def get_config(key: Optional[str] = None, default: Any = None) -> Any:
    """
    설정값을 가져옵니다.
    
    Args:
        key: 설정 키 (None이면 전체 설정 반환)
        default: 키가 없을 경우 반환할 기본값
        
    Returns:
        설정값 또는 전체 설정
    """
    global _CONFIG
    
    # 설정이 로드되지 않았으면 로드
    if _CONFIG is None:
        load_config()
    
    # 키가 없으면 전체 설정 반환
    if key is None:
        return _CONFIG
    
    # 키에 해당하는 설정값 반환
    return _CONFIG.get(key, default)


def set_config(key: str, value: Any) -> None:
    """
    설정값을 변경합니다.
    
    Args:
        key: 설정 키
        value: 새 설정값
    """
    global _CONFIG
    
    # 설정이 로드되지 않았으면 로드
    if _CONFIG is None:
        load_config()
    
    _CONFIG[key] = value


def save_config(filepath: str) -> bool:
    """
    현재 설정을 파일로 저장합니다.
    
    Args:
        filepath: 저장할 파일 경로
        
    Returns:
        성공 여부
    """
    global _CONFIG
    
    # 설정이 로드되지 않았으면 로드
    if _CONFIG is None:
        load_config()
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(_CONFIG, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"설정 저장 실패: {str(e)}")
        return False


def validate_config() -> Dict[str, bool]:
    """
    현재 설정의 유효성을 검사합니다.
    
    Returns:
        각 설정 항목의 유효성 결과 딕셔너리
    """
    global _CONFIG
    
    # 설정이 로드되지 않았으면 로드
    if _CONFIG is None:
        load_config()
    
    validation_results = {}
    
    # 날짜 형식 검증
    for date_key in ['START_DATE', 'END_DATE']:
        try:
            datetime.datetime.strptime(_CONFIG[date_key], '%Y-%m-%d')
            validation_results[date_key] = True
        except (ValueError, KeyError):
            validation_results[date_key] = False
    
    # TOUCHSTONE이 TICKERS에 포함되어 있는지 검증
    validation_results['TOUCHSTONE'] = _CONFIG.get('TOUCHSTONE') in _CONFIG.get('TICKERS', [])
    
    # 자산 배분 비율 합이 1인지 검증
    for alloc_key in ['ASSET_ALLOCATION', 'AGGRESSIVE_ALLOCATION', 'MODERATE_ALLOCATION', 
                      'DEFENSIVE_ALLOCATION', 'MID_DEFENSIVE_ALLOCATION']:
        try:
            alloc = _CONFIG.get(alloc_key, {})
            # 자산 배분 티커가 모두 TICKERS에 있는지 확인
            all_in_tickers = all(ticker in _CONFIG.get('TICKERS', []) for ticker in alloc.keys())
            # 합이 1에 가까운지 확인 (0.99 ~ 1.01)
            total = sum(alloc.values())
            sum_valid = 0.99 <= total <= 1.01
            
            validation_results[alloc_key] = all_in_tickers and sum_valid
        except Exception:
            validation_results[alloc_key] = False
    
    # 이동평균 기간이 유효한지 검증
    try:
        ma_periods = _CONFIG.get('MA_PERIODS', [])
        validation_results['MA_PERIODS'] = (
            isinstance(ma_periods, list) and 
            len(ma_periods) > 0 and 
            all(isinstance(p, int) and p > 0 for p in ma_periods)
        )
    except Exception:
        validation_results['MA_PERIODS'] = False
    
    # 주간 이동평균 기간이 유효한지 검증
    try:
        wma_periods = _CONFIG.get('WMA_PERIODS', [])
        validation_results['WMA_PERIODS'] = (
            isinstance(wma_periods, list) and 
            len(wma_periods) > 0 and 
            all(isinstance(p, int) and p > 0 for p in wma_periods)
        )
    except Exception:
        validation_results['WMA_PERIODS'] = False
    
    # 이동평균 타입이 유효한지 검증
    validation_results['MA_TYPE'] = _CONFIG.get('MA_TYPE') in ['daily', 'weekly']
    
    # 전체 유효성 판단
    validation_results['is_valid'] = all(validation_results.values())
    
    return validation_results


def setup_logging() -> logging.Logger:
    """
    로깅을 설정합니다.
    
    Returns:
        설정된 로거
    """
    global _CONFIG
    
    # 설정이 로드되지 않았으면 로드
    if _CONFIG is None:
        load_config()
    
    # 로그 레벨 매핑
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = level_map.get(_CONFIG.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    
    # 타임스탬프로 로그 파일명 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"backtest_{timestamp}.log")
    
    # 로거 설정
    logger = logging.getLogger("backtest")
    logger.setLevel(log_level)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    return logger


def log_config(logger: logging.Logger) -> None:
    """
    현재 설정 정보를 로그에 기록합니다.
    
    Args:
        logger: 로거 객체
    """
    global _CONFIG
    
    # 설정이 로드되지 않았으면 로드
    if _CONFIG is None:
        load_config()
    
    logger.info("=============== 설정 정보 ===============")
    logger.info(f"기간: {_CONFIG['START_DATE']} ~ {_CONFIG['END_DATE']}")
    logger.info(f"티커: {_CONFIG['TICKERS']}")
    logger.info(f"기준 티커: {_CONFIG['TOUCHSTONE']}")
    logger.info(f"이동평균 타입: {_CONFIG['MA_TYPE']} (daily/weekly)")
    logger.info(f"기본 자산 배분: {_CONFIG['ASSET_ALLOCATION']}")
    logger.info(f"공격적 자산 배분: {_CONFIG['AGGRESSIVE_ALLOCATION']}")
    logger.info(f"중립적 자산 배분: {_CONFIG['MODERATE_ALLOCATION']}")
    logger.info(f"방어적 자산 배분: {_CONFIG['DEFENSIVE_ALLOCATION']}")
    logger.info(f"중간 방어적 자산 배분: {_CONFIG['MID_DEFENSIVE_ALLOCATION']}")
    logger.info(f"이동평균 기간: {_CONFIG['MA_PERIODS']}")
    logger.info(f"주간 이동평균 기간: {_CONFIG['WMA_PERIODS']}")
    logger.info(f"초기 자본: {_CONFIG['INITIAL_CASH']:,.2f}")
    logger.info(f"수수료: {_CONFIG['COMMISSION'] * 100:.2f}%")
    logger.info(f"시장 상태 활용: {_CONFIG['USE_MARKET_REGIME']}")
    logger.info(f"소수점 주식 거래: {_CONFIG.get('FRACTIONAL_SHARES', False)}")
    logger.info("=========================================")