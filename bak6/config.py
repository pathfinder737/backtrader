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
            "END_DATE": os.environ.get("END_DATE", "2024-12-31"),
            "TICKERS": ["SPY", "QQQ", "IAU", "IEF"],
            "ASSET_ALLOCATION": {"QQQ": 0.5, "IAU": 0.3, "IEF": 0.2},
            "AGGRESSIVE_ALLOCATION": {"QQQ": 0.9, "IAU": 0.1, "IEF": 0},
            "MODERATE_ALLOCATION": {"QQQ": 0.8, "IAU": 0.1, "IEF": 0.1},
            "DEFENSIVE_ALLOCATION": {"QQQ": 0, "IAU": 0.5, "IEF": 0.5},
            "MID_DEFENSIVE_ALLOCATION": {"QQQ": 0.4, "IAU": 0.4, "IEF": 0.2},
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
            "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO")
        }
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
        validation = {}
        for date_key in ['START_DATE', 'END_DATE']:
            try:
                datetime.datetime.strptime(self.settings[date_key], '%Y-%m-%d')
                validation[date_key] = True
            except Exception:
                validation[date_key] = False
        validation['TOUCHSTONE'] = self.settings.get('TOUCHSTONE') in self.settings.get('TICKERS', [])

        for alloc_key in ['ASSET_ALLOCATION', 'AGGRESSIVE_ALLOCATION', 'MODERATE_ALLOCATION',
                          'DEFENSIVE_ALLOCATION', 'MID_DEFENSIVE_ALLOCATION']:
            alloc = self.settings.get(alloc_key, {})
            all_in_tickers = all(ticker in self.settings.get('TICKERS', []) for ticker in alloc.keys())
            total = sum(alloc.values())
            validation[alloc_key] = all_in_tickers and (0.99 <= total <= 1.01)

        ma_periods = self.settings.get('MA_PERIODS', [])
        validation['MA_PERIODS'] = isinstance(ma_periods, list) and all(isinstance(p, int) and p > 0 for p in ma_periods)
        wma_periods = self.settings.get('WMA_PERIODS', [])
        validation['WMA_PERIODS'] = isinstance(wma_periods, list) and all(isinstance(p, int) and p > 0 for p in wma_periods)
        validation['MA_TYPE'] = self.settings.get('MA_TYPE') in ['daily', 'weekly']
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
