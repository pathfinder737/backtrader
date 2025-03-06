# bt_frequent_composite.py
import backtrader as bt
import FinanceDataReader as fdr
import datetime

class FrequentCompositeStrategy(bt.Strategy):
    params = dict(
        rsi_period=51,        # RSI 기간
        macd_fast=26,         # MACD 단기 EMA 기간
        macd_slow=76,         # MACD 장기 EMA 기간
        macd_signal=17         # MACD 시그널 기간
    )

    def __init__(self):
        # RSI 지표와 50 기준선 교차
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.rsi_cross = bt.indicators.CrossOver(self.rsi, 50)
        
        # MACD 지표 및 MACD와 시그널 라인의 교차
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        # 포지션이 없으면 매수 조건: RSI 또는 MACD의 상승 교차 발생 시
        if not self.position:
            if self.rsi_cross > 0 or self.macd_cross > 0:
                self.buy()
        else:
            # 포지션 보유 중이면 매도 조건: RSI 또는 MACD의 하락 교차 발생 시
            if self.rsi_cross < 0 or self.macd_cross < 0:
                self.close()

if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # 데이터 기간 설정: 2010.01.01 ~ 2020.12.31
    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime(2020, 12, 31)
    data_df = fdr.DataReader('QQQ', start=start_date, end=end_date)

    # Pandas DataFrame을 Backtrader 데이터 피드로 변환
    data = bt.feeds.PandasData(
        dataname=data_df,
        datetime=None,  # DataFrame 인덱스가 datetime이면 None 사용
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1  # OpenInterest 데이터 없음
    )
    cerebro.adddata(data)

    # 전략 추가
    cerebro.addstrategy(FrequentCompositeStrategy)

    # 분석기 추가
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Months, _name='timereturn_monthly')

    # 초기 자금 및 수수료 설정
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # 백테스트 실행
    results = cerebro.run()

    # 결과 출력
    print("----- 분석 결과 -----")

    returns = results[0].analyzers.returns.get_analysis()
    print("\n1. Returns Analyzer:")
    print(f"  - 총 수익률 (Total Return): {returns.get('rtot', 0):.2f}%")
    print(f"  - 연간 수익률 (Annualized Return): {returns.get('rnorm', 0):.2f}%")

    sharpe = results[0].analyzers.sharpe.get_analysis()
    print("\n2. Sharpe Ratio Analyzer:")
    if sharpe and sharpe.get('sharperatio') is not None:
        print(f"  - Sharpe Ratio: {sharpe['sharperatio']:.2f}")
    else:
        print("  - Sharpe Ratio not available")

    drawdown = results[0].analyzers.drawdown.get_analysis()
    print("\n3. DrawDown Analyzer:")
    print(f"  - 최대 낙폭 (Max Drawdown): {drawdown['max']['drawdown']:.2f}")
    print(f"  - 최대 낙폭률 (Max Drawdown Percentage): {drawdown['max'].get('drawdownpct', 0):.2f}%")
    print(f"  - 최대 낙폭 기간 (Max Drawdown Duration): {drawdown['max']['len']}")

    sqn = results[0].analyzers.sqn.get_analysis()
    print("\n4. SQN Analyzer:")
    print(f"  - 시스템 품질 지수 (SQN): {sqn['sqn']:.2f}")

    trade_analysis = results[0].analyzers.trade.get_analysis()
    print("\n5. Trade Analyzer:")
    print(f"  - 총 거래 횟수 (Total Trades): {trade_analysis['total']['total']}")
    print(f"  - 승리 횟수 (Winning Trades): {trade_analysis['won']['total']}")
    print(f"  - 패배 횟수 (Losing Trades): {trade_analysis['lost']['total']}")
    print(f"  - 총 손익 (Net Profit): {trade_analysis['pnl']['net']['total']:.2f}")
    print(f"  - 총 손실 (Net Loss): {trade_analysis['pnl']['net']['average']:.2f}")
    print(f"  - 최대 이익 거래 (Biggest Winner): {trade_analysis['won']['pnl']['max']:.2f}")
    print(f"  - 최대 손실 거래 (Biggest Loser): {trade_analysis['lost']['pnl']['max']:.2f}")
    print(f"  - 평균 승리 거래 이익 (Average Winner): {trade_analysis['won']['pnl']['average']:.2f}")
    print(f"  - 평균 손실 거래 손실 (Average Loser): {trade_analysis['lost']['pnl']['average']:.2f}")
    profit_factor = (trade_analysis['won']['pnl']['total'] /
                     abs(trade_analysis['lost']['pnl']['total'])
                     if trade_analysis['lost']['pnl']['total'] != 0 else 0)
    recovery_factor = (trade_analysis['pnl']['net']['total'] /
                       abs(drawdown['max']['drawdown'])
                       if drawdown['max']['drawdown'] != 0 else 0)
    print(f"  - 이익 계수 (Profit Factor): {profit_factor:.2f}")
    print(f"  - 복구 계수 (Recovery Factor): {recovery_factor:.2f}")

    # timereturn_monthly = results[0].analyzers.timereturn_monthly.get_analysis()
    # print("\n8. TimeReturn Analyzer (Monthly):")
    # for period, return_value in timereturn_monthly.items():
    #     print(f"  - {period.strftime('%Y-%m')}: {return_value:.2f}%")

    port_value = cerebro.broker.getvalue()
    pnl = port_value - 100000.0
    print(f"\n----- 최종 결과 -----")
    print(f"최종 포트폴리오 가치: {port_value:.2f}")
    print(f"총 손익 (Net Profit): {pnl:.2f}")
    print(f"총 수익률 (Return on Investment): {pnl / 100000.0 * 100:.2f}%")

    # CAGR 계산
    initial_value = 100000.0
    final_value = port_value
    years = (end_date - start_date).days / 365.25
    cagr = (final_value / initial_value) ** (1 / years) - 1
    print(f"CAGR: {cagr:.2%}")

    # 필요시 시각화
    # cerebro.plot()
