# config.py
import os
import logging
import datetime
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_file: str = None):
        self.BASE_DIR = Path(__file__).resolve().parent
        
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")
        

        self.DEFAULT_CONFIG = {
            "DATA_MODE": os.environ.get("DATA_MODE", "offline"),
            "START_DATE": os.environ.get("START_DATE", "2000-01-01"),
            "END_DATE": os.environ.get("END_DATE", "2025-3-31"),
            "TICKERS": ["SPY", "QQQ", "GLD"],
            "ASSET_ALLOCATION": {"QQQ": 0.5, "GLD": 0.5},
            "AGGRESSIVE_ALLOCATION": {"QQQ": 0.7, "GLD": 0.3},
            "MODERATE_ALLOCATION": {"QQQ": 0.6, "GLD": 0.4},
            "MID_DEFENSIVE_ALLOCATION": {"QQQ": 0.5, "GLD": 0.5},
            "DEFENSIVE_ALLOCATION": {"QQQ": 0.5, "GLD": 0.5},
            "TOUCHSTONE": "SPY",
            "MA_PERIODS": [21, 50, 150, 200],
            "WMA_PERIODS": [10, 30],
            "MA_TYPE": "weekly",  # 'weekly, daily'로  모드전환환
            "INITIAL_CASH": 100000.0,
            "COMMISSION": 0.001,
            "CASH_BUFFER_PERCENT": 1.0,
            "PROCESS_SELL_FIRST": True,
            "RISK_FREE_RATE": 0.01,
            "FRACTIONAL_SHARES": False,
            "USE_MARKET_REGIME": True,
            "REBALANCING_TERM": "MONTH",  # <-- 추가
            "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO")
        }

        # # 김단테 All Weather
        #   { "VTI": 0.12, "VEA": 0.12, "VWO": 0.12, "DBC": 0.07,"IAU": 0.07, "EDV": 0.18,"LTPZ": 0.18,"VCLT": 0.7,"EMLC": 0.7 }
        #   { "VOO": 0.12, "EFA": 0.12, "EEM": 0.12, "DBC": 0.07,"GLD": 0.07, "EDV": 0.18,"LTPZ": 0.18,"LQD": 0.07,"EMLC": 0.07 }


        # self.DEFAULT_CONFIG = {
        #     "DATA_MODE": os.environ.get("DATA_MODE", "offline"),
        #     "START_DATE": os.environ.get("START_DATE", "2000-01-01"),
        #     "END_DATE": os.environ.get("END_DATE", "2025-3-31"),
        #     "TICKERS": ["SPY", "VOO", "EFA", "EEM","DBC","GLD","EDV","LTPZ","LQD","EMLC"],
        #     "ASSET_ALLOCATION": { "VOO": 0.12, "EFA": 0.12, "EEM": 0.12, "DBC": 0.07,"GLD": 0.07, "EDV": 0.18,"LTPZ": 0.18,"LQD": 0.07,"EMLC": 0.07 },
        #     "AGGRESSIVE_ALLOCATION": { "VOO": 0.12, "EFA": 0.12, "EEM": 0.12, "DBC": 0.07,"GLD": 0.07, "EDV": 0.18,"LTPZ": 0.18,"LQD": 0.07,"EMLC": 0.07 },
        #     "MODERATE_ALLOCATION": { "VOO": 0.12, "EFA": 0.12, "EEM": 0.12, "DBC": 0.07,"GLD": 0.07, "EDV": 0.18,"LTPZ": 0.18,"LQD": 0.07,"EMLC": 0.07 },
        #     "MID_DEFENSIVE_ALLOCATION": { "VOO": 0.12, "EFA": 0.12, "EEM": 0.12, "DBC": 0.07,"GLD": 0.07, "EDV": 0.18,"LTPZ": 0.18,"LQD": 0.07,"EMLC": 0.07 },
        #     "DEFENSIVE_ALLOCATION": { "VOO": 0.12, "EFA": 0.12, "EEM": 0.12, "DBC": 0.07,"GLD": 0.07, "EDV": 0.18,"LTPZ": 0.18,"LQD": 0.07,"EMLC": 0.07 },
        #     "TOUCHSTONE": "SPY",
        #     "MA_PERIODS": [21, 50, 150, 200],
        #     "WMA_PERIODS": [10, 30],
        #     "MA_TYPE": "daily",  # 'weekly, daily'로  모드전환환
        #     "INITIAL_CASH": 100000.0,
        #     "COMMISSION": 0.001,
        #     "CASH_BUFFER_PERCENT": 1.0,
        #     "PROCESS_SELL_FIRST": True,
        #     "RISK_FREE_RATE": 0.01,
        #     "FRACTIONAL_SHARES": False,
        #     "USE_MARKET_REGIME": False,
        #     "REBALANCING_TERM": "YEAR",  # <-- 추가 MONTH, QUATER, HALF, YEAR
        #     "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO")
        # }

        # # All Season
        #   
        #   { "VOO": 0.3, "IEF": 0.15, "TLT": 0.4, "GLD" : 0.075, "DBC" : 0.075 }

        # self.DEFAULT_CONFIG = {
        #     "DATA_MODE": os.environ.get("DATA_MODE", "offline"),
        #     "START_DATE": os.environ.get("START_DATE", "2000-01-01"),
        #     "END_DATE": os.environ.get("END_DATE", "2025-3-31"),
        #     "TICKERS": ["SPY", "VOO", "IEF", "TLT","GLD","DBC"],
        #     "ASSET_ALLOCATION": { "VOO": 0.3, "IEF": 0.15, "TLT": 0.4, "GLD" : 0.075, "DBC" : 0.075 },
        #     "AGGRESSIVE_ALLOCATION": { "VOO": 0.3, "IEF": 0.15, "TLT": 0.4, "GLD" : 0.075, "DBC" : 0.075 },
        #     "MODERATE_ALLOCATION": { "VOO": 0.3, "IEF": 0.15, "TLT": 0.4, "GLD" : 0.075, "DBC" : 0.075 },
        #     "MID_DEFENSIVE_ALLOCATION": { "VOO": 0.3, "IEF": 0.15, "TLT": 0.4, "GLD" : 0.075, "DBC" : 0.075 },
        #     "DEFENSIVE_ALLOCATION": { "VOO": 0.3, "IEF": 0.15, "TLT": 0.4, "GLD" : 0.075, "DBC" : 0.075 },
        #     "TOUCHSTONE": "SPY",
        #     "MA_PERIODS": [21, 50, 150, 200],
        #     "WMA_PERIODS": [10, 30],
        #     "MA_TYPE": "daily",  # 'weekly, daily'로  모드전환환
        #     "INITIAL_CASH": 100000.0,
        #     "COMMISSION": 0.001,
        #     "CASH_BUFFER_PERCENT": 1.0,
        #     "PROCESS_SELL_FIRST": True,
        #     "RISK_FREE_RATE": 0.01,
        #     "FRACTIONAL_SHARES": False,
        #     "USE_MARKET_REGIME": False,
        #     "REBALANCING_TERM": "YEAR",  # <-- 추가 MONTH, QUATER, HALF, YEAR
        #     "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO")
        # }

       
        # ===== Analyzer Results =====
        # { "QQQ": 0.5, "GLD": 0.5 }
        # Final Portfolio Value: 949,069.25
        # Total Return (rtot): 225.03%
        # Annual Return (rnorm): 11.16%
        # Max Drawdown: 31.72%
        # Sharpe Ratio: 0.72
        # Net Profit: 849,069.25
        # 2025-03-08 14:41:38,624 - 

        # self.DEFAULT_CONFIG = {
        #     "DATA_MODE": os.environ.get("DATA_MODE", "offline"),
        #     "START_DATE": os.environ.get("START_DATE", "2000-01-01"),
        #     "END_DATE": os.environ.get("END_DATE", "2025-3-31"),
        #     "TICKERS": ["SPY", "QQQ","GLD","LQD"],
        #     "ASSET_ALLOCATION": { "QQQ": 0.5, "GLD": 0.5, "LQD": 0 },
        #     "AGGRESSIVE_ALLOCATION": { "QQQ": 0.5, "GLD": 0.5, "LQD": 0 },
        #     "MODERATE_ALLOCATION": { "QQQ": 0.5, "GLD": 0.5, "LQD": 0 },
        #     "MID_DEFENSIVE_ALLOCATION": { "QQQ": 0.5, "GLD": 0.5, "LQD": 0 },
        #     "DEFENSIVE_ALLOCATION":{ "QQQ": 0.5, "GLD": 0.5, "LQD": 0 },
        #     "TOUCHSTONE": "SPY",
        #     "MA_PERIODS": [21, 50, 150, 200],
        #     "WMA_PERIODS": [10, 30],
        #     "MA_TYPE": "daily",  # 'weekly, daily'로  모드전환환
        #     "INITIAL_CASH": 100000.0,
        #     "COMMISSION": 0.0009,
        #     "CASH_BUFFER_PERCENT": 1.0,
        #     "PROCESS_SELL_FIRST": True,
        #     "RISK_FREE_RATE": 0.01,
        #     "FRACTIONAL_SHARES": False,
        #     "USE_MARKET_REGIME": False,
        #     "REBALANCING_TERM": "YEAR",  # <-- 추가 MONTH, QUATER, HALF, YEAR
        #     "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO")
        # }


        self.settings = self.DEFAULT_CONFIG.copy()
        if config_file:
            self.load_config(config_file)
        self._ensure_directories()

    def _ensure_directories(self):
        for directory in [self.DATA_DIR, self.LOG_DIR, self.RESULTS_DIR]:
            os.makedirs(directory, exist_ok=True)

    def load_config(self, config_file: str) -> None:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self.settings.update(file_config)
        except Exception as e:
            print(f"Failed to load config file {config_file}: {e}")

    def get(self, key: str, default=None):
        return self.settings.get(key, default)

    def set(self, key: str, value) -> None:
        self.settings[key] = value

    def save_config(self, filepath: str) -> bool:
        try:
            import yaml
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self.settings, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False

    def validate(self) -> dict:
        """
        설정값 유효성 검증
        """
        validation = {}

        # 날짜 형식 검사
        for date_key in ['START_DATE', 'END_DATE']:
            try:
                datetime.datetime.strptime(self.settings[date_key], '%Y-%m-%d')
                validation[date_key] = True
            except Exception:
                validation[date_key] = False

        # Touchstone 티커가 TICKERS 목록 안에 있는지 확인
        validation['TOUCHSTONE'] = self.settings.get('TOUCHSTONE') in self.settings.get('TICKERS', [])

        # 자산 할당 비율 합계 1.0 근사치인지 확인
        for alloc_key in ['ASSET_ALLOCATION', 'AGGRESSIVE_ALLOCATION', 'MODERATE_ALLOCATION',
                          'DEFENSIVE_ALLOCATION', 'MID_DEFENSIVE_ALLOCATION']:
            alloc = self.settings.get(alloc_key, {})
            all_in_tickers = all(ticker in self.settings.get('TICKERS', []) for ticker in alloc.keys())
            total = sum(alloc.values())
            validation[alloc_key] = all_in_tickers and (0.99 <= total <= 1.01)

        # 이동평균 관련 설정 검사
        ma_periods = self.settings.get('MA_PERIODS', [])
        validation['MA_PERIODS'] = isinstance(ma_periods, list) and all(isinstance(p, int) and p > 0 for p in ma_periods)
        wma_periods = self.settings.get('WMA_PERIODS', [])
        validation['WMA_PERIODS'] = isinstance(wma_periods, list) and all(isinstance(p, int) and p > 0 for p in wma_periods)
        validation['MA_TYPE'] = self.settings.get('MA_TYPE') in ['daily', 'weekly']

        # [추가] REBALANCING_TERM 유효성 검사
        valid_terms = ["MONTH", "QUARTER", "HALF", "YEAR"]
        rebal_term = self.settings.get('REBALANCING_TERM')
        validation['REBALANCING_TERM'] = (rebal_term in valid_terms)

        # 최종
        validation['is_valid'] = all(validation.values())
        return validation

    def setup_logging(self) -> logging.Logger:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        log_level = level_map.get(self.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.LOG_DIR, f"backtest_{timestamp}.log")
        logger = logging.getLogger("backtest")
        logger.setLevel(log_level)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)
        return logger

    def log_config(self, logger: logging.Logger) -> None:
        logger.info("=============== Configurations ===============")
        for key, value in self.settings.items():
            logger.info(f"{key}: {value}")
        logger.info("===============================================")

config = Config()
