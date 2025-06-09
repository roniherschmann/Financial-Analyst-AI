# Portfolio Risk Management System

## Overview

This is a comprehensive risk management system designed for hedge fund portfolios, specifically built to handle a $4M portfolio. The system provides advanced risk analytics, portfolio optimization, and factor-based analysis using modern quantitative finance techniques.

## Key Features

### 1. **Risk Analytics**
- **Value at Risk (VaR)** calculations using multiple methodologies:
  - Parametric VaR (analytical approach)
  - Historical simulation
  - Monte Carlo simulation
  - Factor-based simulation
- **Conditional Value at Risk (CVaR)** for tail risk assessment
- **Risk decomposition** into systematic and idiosyncratic components

### 2. **Portfolio Optimization**
- Multiple optimization strategies:
  - Minimum variance portfolios
  - Maximum Sharpe ratio optimization
  - Risk parity allocation
  - Target return optimization
- **Efficient frontier** generation and visualization
- **Black-Litterman** model for incorporating market views

### 3. **Factor Analysis**
- **Barra-style factor model** using ETF proxies for factors:
  - Market, Sectors (Technology, Financials, Energy, Healthcare)
  - Style factors (Momentum, Quality, Low Volatility)
- Factor exposure analysis and risk attribution
- R-squared analysis to assess model fit

### 4. **Stress Testing**
- Scenario-based stress tests
- Factor shock analysis
- Historical crisis simulation

## System Architecture

### Core Components

1. **`main.py`** - Central orchestration and risk management system
2. **`data_fetch.py`** - Market data retrieval from Yahoo Finance and FRED
3. **`barra_factor.py`** - Factor model implementation and analysis
4. **`montecarlo.py`** - Monte Carlo simulation engine
5. **`efficient_frontier.py`** - Portfolio optimization algorithms

### Visualization Modules

6. **`clean_visualizations.py`** - Monte Carlo path visualizations
7. **`professional_efficient_frontier.py`** - Professional-grade efficient frontier plots

## Installation & Setup

### Prerequisites
```bash
# Required Python packages
pip install numpy pandas matplotlib seaborn scipy
pip install yfinance fredapi
pip install scikit-learn cvxpy
pip install python-dotenv
```

### Environment Configuration
Create a `.env` file with the following variables:
```env
# Portfolio Settings
PORTFOLIO_VALUE=4000000
MC_SIMULATIONS=10000
MC_TIME_HORIZON=21

# Risk Limits
MAX_VAR_95=0.02
MAX_CONCENTRATION=0.25
MIN_SHARPE=0.5

# Data Settings
FRED_API_KEY=your_fred_api_key_here
DEFAULT_START_YEARS=3
DEFAULT_END_DATE=today
ENVIRONMENT=production
```

## Usage Examples

### Basic Portfolio Analysis
```python
from main import RiskManagementSystem

# Initialize system
risk_system = RiskManagementSystem(portfolio_value=4_000_000)
risk_system.initialize()

# Define portfolio
tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
weights = [0.25, 0.20, 0.20, 0.20, 0.15]

# Run analysis
results = risk_system.analyze_portfolio(tickers, weights)
risk_system.display_results(results)
risk_system.generate_report(results, 'portfolio_report.png')
```

### Portfolio Optimization
```python
# Find optimal allocation
optimal_results = risk_system.optimize_portfolio(
    tickers, 
    method='sharpe'  # or 'min_variance', 'risk_parity'
)
```

### Stress Testing
```python
# Run stress tests
stress_results = risk_system.run_stress_tests(tickers, weights)
```

## Key Metrics Explained

### Risk Metrics
- **VaR (95%)**: Maximum expected loss over the time horizon with 95% confidence
- **CVaR**: Average loss beyond the VaR threshold
- **Annual Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return metric

### Factor Analysis
- **Factor Betas**: Sensitivity to each factor (market, sectors, styles)
- **R-squared**: Proportion of variance explained by factors
- **Systematic Risk**: Risk from market factors
- **Idiosyncratic Risk**: Asset-specific risk

## Output Examples

The system generates several types of outputs:

1. **Risk Report Dashboard** - Comprehensive 6-panel visualization showing:
   - Portfolio allocation pie chart
   - Returns distribution with normal overlay
   - VaR comparison across methods
   - Factor exposures
   - VaR analysis histogram
   - Performance metrics table

2. **Monte Carlo Paths** - Visualization of 1,000+ simulated portfolio paths
   - Expected path and confidence bands
   - 5th/95th percentile boundaries

3. **Efficient Frontier** - Professional visualization showing:
   - Risk-return tradeoff
   - Current portfolio position
   - Optimal portfolios (min variance, max Sharpe)
   - Capital allocation line

4. **Factor Analysis** - Multiple visualizations:
   - Factor exposure heatmap
   - Risk decomposition donut chart
   - Factor performance over time

## Risk Limits & Compliance

The system monitors compliance with configurable risk limits:
- Maximum VaR (95%): 2% of portfolio value
- Maximum concentration: 25% in any single asset
- Minimum Sharpe ratio: 0.5

Violations are flagged in the risk report.

## Data Sources

- **Market Data**: Yahoo Finance for stock prices
- **Factor Data**: ETF proxies for factor returns
- **Macro Data**: FRED API for interest rates and economic indicators

## Performance Considerations

- Handles portfolios with 3+ years of historical data
- Monte Carlo simulations are computationally intensive (10,000 paths default)
- Factor model requires at least 60 days of overlapping data

## Error Handling

The system includes robust error handling for:
- Missing or incomplete market data
- API connection failures
- Numerical instabilities in optimization
- Data alignment issues

## Limitations

1. Assumes liquid, tradeable assets
2. Historical data dependency for risk estimates
3. Factor model uses ETF proxies (not pure factors)
4. Does not account for transaction costs or market impact

## Future Enhancements

Potential improvements could include:
- Real-time risk monitoring
- Alternative risk measures (Expected Shortfall, Drawdown)
- Machine learning-based factor models
- Integration with execution systems
- Multi-currency support

## Support

For issues or questions:
1. Check the generated log files for detailed error messages
2. Ensure all dependencies are correctly installed
3. Verify data availability for the specified date range
4. Confirm API keys are properly configured

---

**Note**: This system is designed for professional portfolio management. Always validate results and ensure compliance with regulatory requirements before making investment decisions.