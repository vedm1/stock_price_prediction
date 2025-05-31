
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Backtest:
    """
    Comprehensive backtesting framework for trading strategies
    """

    def __init__(self, data, initial_capital=100000, commission=0.001):
        """
        Initialize backtesting engine

        Parameters:
        data: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        initial_capital: Starting capital
        commission: Transaction cost as percentage (0.001 = 0.1%)
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.positions = []
        self.trades = []
        self.equity_curve = []

    def run_strategy(self, strategy_func):
        """
        Run a trading strategy

        Parameters:
        strategy_func: Function that takes data and returns signals
        """
        # Generate signals
        self.data = strategy_func(self.data)

        # Initialize tracking variables
        cash = self.initial_capital
        position = 0
        equity = []

        for i in range(len(self.data)):
            row = self.data.iloc[i]

            # Record current equity
            current_equity = cash + position * row['Close']
            equity.append(current_equity)

            # Check for buy signal
            if row.get('Signal') == 1 and position == 0:
                # Buy
                shares = int(cash * 0.95 / row['Close'])  # Use 95% of cash
                if shares > 0:
                    cost = shares * row['Close'] * (1 + self.commission)
                    if cost <= cash:
                        cash -= cost
                        position = shares
                        self.trades.append({
                            'Date': row['Date'],
                            'Type': 'BUY',
                            'Price': row['Close'],
                            'Shares': shares,
                            'Cost': cost,
                            'Cash': cash
                        })

            # Check for sell signal
            elif row.get('Signal') == -1 and position > 0:
                # Sell
                proceeds = position * row['Close'] * (1 - self.commission)
                cash += proceeds
                self.trades.append({
                    'Date': row['Date'],
                    'Type': 'SELL',
                    'Price': row['Close'],
                    'Shares': position,
                    'Proceeds': proceeds,
                    'Cash': cash
                })
                position = 0

        # Close any remaining position
        if position > 0:
            final_proceeds = position * self.data.iloc[-1]['Close'] * (1 - self.commission)
            cash += final_proceeds
            equity[-1] = cash

        self.equity_curve = equity
        self.final_capital = equity[-1]

        return self

    def calculate_metrics(self):
        """Calculate performance metrics"""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Basic metrics
        total_return = (self.final_capital - self.initial_capital) / self.initial_capital

        # Risk metrics
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

        # Drawdown
        cumulative = equity / equity[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        if self.trades:
            winning_trades = sum(1 for i in range(0, len(self.trades), 2) 
                               if i+1 < len(self.trades) and 
                               self.trades[i+1]['Proceeds'] > self.trades[i]['Cost'])
            total_trades = len(self.trades) // 2
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
        else:
            win_rate = 0

        metrics = {
            'Total Return': f"{total_return*100:.2f}%",
            'Final Capital': f"${self.final_capital:,.2f}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown*100:.2f}%",
            'Win Rate': f"{win_rate*100:.2f}%",
            'Number of Trades': len(self.trades),
            'Average Return per Trade': f"{returns.mean()*100:.2f}%" if len(returns) > 0 else "0.00%"
        }

        return metrics

    def plot_results(self):
        """Plot backtesting results"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Price and signals
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], 'b-', label='Close Price')

        # Mark buy/sell points
        buys = self.data[self.data['Signal'] == 1]
        sells = self.data[self.data['Signal'] == -1]

        ax1.scatter(buys.index, buys['Close'], color='green', marker='^', 
                   s=100, label='Buy', zorder=5)
        ax1.scatter(sells.index, sells['Close'], color='red', marker='v', 
                   s=100, label='Sell', zorder=5)

        ax1.set_ylabel('Price')
        ax1.set_title('Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Equity curve
        ax2 = axes[1]
        ax2.plot(self.equity_curve, 'g-', linewidth=2)
        ax2.axhline(y=self.initial_capital, color='r', linestyle='--', 
                   label='Initial Capital')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_title('Equity Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Drawdown
        ax3 = axes[2]
        equity = np.array(self.equity_curve)
        cumulative = equity / equity[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100

        ax3.fill_between(range(len(drawdown)), drawdown, 0, 
                        color='red', alpha=0.3)
        ax3.plot(drawdown, 'r-', linewidth=1)
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Time')
        ax3.set_title('Drawdown')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

# Example Trading Strategies

def moving_average_strategy(data, short_window=20, long_window=50):
    """Simple Moving Average Crossover Strategy"""
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()

    data['Signal'] = 0
    data.loc[data['SMA_short'] > data['SMA_long'], 'Signal'] = 1
    data.loc[data['SMA_short'] < data['SMA_long'], 'Signal'] = -1

    # Generate actual trading signals (on crossover)
    data['Position'] = data['Signal'].diff()
    data.loc[data['Position'] > 0, 'Signal'] = 1
    data.loc[data['Position'] < 0, 'Signal'] = -1
    data.loc[data['Position'] == 0, 'Signal'] = 0

    return data

def rsi_strategy(data, period=14, oversold=30, overbought=70):
    """RSI Strategy"""
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Generate signals
    data['Signal'] = 0
    data.loc[data['RSI'] < oversold, 'Signal'] = 1  # Buy when oversold
    data.loc[data['RSI'] > overbought, 'Signal'] = -1  # Sell when overbought

    return data

def bollinger_bands_strategy(data, period=20, std_dev=2):
    """Bollinger Bands Strategy"""
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['STD'] = data['Close'].rolling(window=period).std()
    data['Upper'] = data['SMA'] + (data['STD'] * std_dev)
    data['Lower'] = data['SMA'] - (data['STD'] * std_dev)

    data['Signal'] = 0
    data.loc[data['Close'] < data['Lower'], 'Signal'] = 1  # Buy at lower band
    data.loc[data['Close'] > data['Upper'], 'Signal'] = -1  # Sell at upper band

    return data

# Advanced Backtesting Class with more features

class AdvancedBacktest(Backtest):
    """Extended backtesting with more features"""

    def __init__(self, data, initial_capital=100000, commission=0.001):
        super().__init__(data, initial_capital, commission)
        self.benchmark_returns = None

    def set_benchmark(self, benchmark_data):
        """Set benchmark for comparison"""
        self.benchmark_returns = benchmark_data['Close'].pct_change().dropna()

    def calculate_advanced_metrics(self):
        """Calculate additional performance metrics"""
        metrics = self.calculate_metrics()

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Additional metrics
        if len(returns) > 0:
            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino_ratio = np.sqrt(252) * returns.mean() / downside_std if downside_std > 0 else 0

            # Calmar Ratio
            max_dd = float(metrics['Max Drawdown'].strip('%')) / 100
            annual_return = (self.final_capital / self.initial_capital) ** (252/len(returns)) - 1
            calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0

            metrics.update({
                'Sortino Ratio': f"{sortino_ratio:.2f}",
                'Calmar Ratio': f"{calmar_ratio:.2f}",
                'Annual Return': f"{annual_return*100:.2f}%",
                'Volatility': f"{np.std(returns)*np.sqrt(252)*100:.2f}%"
            })

        return metrics

    def monte_carlo_simulation(self, n_simulations=1000):
        """Run Monte Carlo simulation on the strategy"""
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]

        final_values = []
        for _ in range(n_simulations):
            # Randomly sample returns with replacement
            simulated_returns = np.random.choice(returns, size=len(returns), replace=True)

            # Calculate final value
            final_value = self.initial_capital * np.prod(1 + simulated_returns)
            final_values.append(final_value)

        # Calculate statistics
        final_values = np.array(final_values)

        return {
            'Mean Final Value': np.mean(final_values),
            'Median Final Value': np.median(final_values),
            '5th Percentile': np.percentile(final_values, 5),
            '95th Percentile': np.percentile(final_values, 95),
            'Probability of Profit': (final_values > self.initial_capital).mean()
        }

# Vectorized Backtesting (Faster)

class VectorizedBacktest:
    """Fast vectorized backtesting"""

    def __init__(self, data, initial_capital=100000, commission=0.001):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission

    def run_strategy(self, strategy_func):
        """Run strategy using vectorized operations"""
        # Generate signals
        self.data = strategy_func(self.data)

        # Calculate returns
        self.data['Returns'] = self.data['Close'].pct_change()

        # Calculate strategy returns
        self.data['Strategy_Returns'] = self.data['Signal'].shift(1) * self.data['Returns']

        # Account for transaction costs
        self.data['Trade'] = self.data['Signal'].diff().abs()
        self.data['Strategy_Returns'] -= self.data['Trade'] * self.commission

        # Calculate cumulative returns
        self.data['Cumulative_Returns'] = (1 + self.data['Strategy_Returns']).cumprod()
        self.data['Portfolio_Value'] = self.initial_capital * self.data['Cumulative_Returns']

        return self

    def get_metrics(self):
        """Calculate performance metrics"""
        total_return = self.data['Cumulative_Returns'].iloc[-1] - 1

        # Sharpe ratio
        sharpe = np.sqrt(252) * self.data['Strategy_Returns'].mean() / self.data['Strategy_Returns'].std()

        # Max drawdown
        cumulative = self.data['Cumulative_Returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'Total Return': f"{total_return*100:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown*100:.2f}%",
            'Final Value': f"${self.data['Portfolio_Value'].iloc[-1]:,.2f}"
        }

# Example usage
if __name__ == "__main__":
    print("=== Backtesting Framework Examples ===\n")

    print("1. Simple Moving Average Strategy:")
    print("   bt = Backtest(data, initial_capital=100000)")
    print("   bt.run_strategy(moving_average_strategy)")
    print("   metrics = bt.calculate_metrics()")
    print("   bt.plot_results()")

    print("\n2. Vectorized Backtesting (Faster):")
    print("   vbt = VectorizedBacktest(data)")
    print("   vbt.run_strategy(rsi_strategy)")
    print("   metrics = vbt.get_metrics()")

    print("\n3. Advanced Features:")
    print("   abt = AdvancedBacktest(data)")
    print("   abt.run_strategy(bollinger_bands_strategy)")
    print("   mc_results = abt.monte_carlo_simulation()")
