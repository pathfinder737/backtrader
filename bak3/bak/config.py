# config.py
# 백테스팅 기본 설정 파일

# 데이터 처리 모드: 'online'이면 실시간 다운로드, 'offline'이면 무조건 다운로드 후 전처리하여 CSV 파일을 사용
DATA_MODE = 'offline'  # 기본값: offline

# 데이터 기간 설정
START_DATE = '2000-01-01'
END_DATE = '2023-12-31'

# 사용할 티커 목록 (TOUCHSTONE 포함되어야 함)
TICKERS = ['SPY', 'QQQ', 'IAU', 'IEF']  

# 기본 자산 배분 설정 (SPY는 TOUCHSTONE으로만 사용되므로 포함하지 않음)
ASSET_ALLOCATION = {'QQQ': 0.5, 'IAU': 0.3, 'IEF': 0.2}

# 시장 상태별 자산 배분 설정
# MA21 > MA50 > MA200: 공격적 배분
AGGRESSIVE_ALLOCATION = {'QQQ': 0.8, 'IAU': 0.2, 'IEF': 0}
# MA50 > MA200: 중립적 배분
MODERATE_ALLOCATION = {'QQQ': 0.7, 'IAU': 0.2, 'IEF': 0.1}
# MA50 < MA200: 방어적 배분
DEFENSIVE_ALLOCATION = {'QQQ': 0, 'IAU': 0.5, 'IEF': 0.5}
# 이동평균 계산 기준 티커
TOUCHSTONE = 'SPY'

# 이동평균 기간 설정
MA_PERIODS = [21, 50, 200]  # 여러 이동평균 기간 정의

# 초기 자본 및 수수료
INITIAL_CASH = 100000.0
COMMISSION = 0.001

# 현금 버퍼 비율 (총 포트폴리오 가치의 %)
CASH_BUFFER_PERCENT = 1.0

# 주문 처리 설정
PROCESS_SELL_FIRST = True  # 매도 주문 먼저 처리

# 로그 및 데이터 저장 경로
LOG_DIR = 'logs'
LOG_FILE = 'logs/backtest.log'
DATA_DIR = 'data'

# 리스크 프리 레이트 (분석 지표 계산에 사용)
RISK_FREE_RATE = 0.01

# 소수점 주식 거래 허용 여부 (False: 정수 단위만 거래, True: 소수점 허용)
FRACTIONAL_SHARES = False

# 시장 상태에 따른 자산 배분 활용 여부
USE_MARKET_REGIME = True