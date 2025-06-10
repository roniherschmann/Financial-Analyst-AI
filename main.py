# main.py
"""
Production risk management system for $4M hedge fund portfolio.
Clean, deployable code with proper error handling.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

from data_fetch import DataFetcher
from barra_factor import BarraFactorModel
from montecarlo import MonteCarloEngine
from efficient_frontier import PortfolioOptimizer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskManagementSystem:
    """
    Integrated risk management system for hedge fund portfolios.
    """
    def __init__(self, portfolio_value: Optional[float] = None):
        # Load from environment or use provided value
        self.portfolio_value = portfolio_value or float(os.getenv('PORTFOLIO_VALUE', 4_000_000))
        
        self.data_fetcher = None
        self.barra_model = BarraFactorModel()
        
        # Load Monte Carlo settings from environment
        mc_sims = int(os.getenv('MC_SIMULATIONS', 10000))
        mc_horizon = int(os.getenv('MC_TIME_HORIZON', 21))
        self.mc_engine = MonteCarloEngine(n_simulations=mc_sims, time_horizon=mc_horizon)
        
        # Load risk limits from environment
        self.risk_limits = {
            'max_var_95': float(os.getenv('MAX_VAR_95', 0.02)),
            'max_concentration': float(os.getenv('MAX_CONCENTRATION', 0.25)),
            'min_sharpe': float(os.getenv('MIN_SHARPE', 0.5)),
        }
        
        logger.info(f"Risk Management System initialized for ${self.portfolio_value:,.0f} portfolio")
    
    def initialize(self, fred_api_key: Optional[str] = None):
        """Initialize data connections."""
        # Use provided key or load from environment
        fred_key = fred_api_key or os.getenv('FRED_API_KEY')
        
        if not fred_key:
            logger.warning("No FRED API key provided. Macro factors will use synthetic data.")
        
        self.data_fetcher = DataFetcher(fred_api_key=fred_key)
        logger.info("Data connections initialized")
    
    def analyze_portfolio(self,
                         tickers: List[str],
                         weights: Optional[np.ndarray] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict:
        """
        Complete portfolio risk analysis pipeline.
        """
        # Handle date defaults using environment settings
        if end_date is None:
            # First check for specific END_DATE in env
            end_date = os.getenv('END_DATE')
            if not end_date:
                end_date_setting = os.getenv('DEFAULT_END_DATE', 'today')
                if end_date_setting == 'today':
                    end_date = datetime.now().strftime('%Y-%m-%d')
                else:
                    end_date = end_date_setting
                
        if start_date is None:
            # First check for specific START_DATE in env
            start_date = os.getenv('START_DATE')
            if not start_date:
                years_back = int(os.getenv('DEFAULT_START_YEARS', 3))
                start_date = (datetime.now() - timedelta(days=years_back*365)).strftime('%Y-%m-%d')
        
        # Validate tickers
        if not tickers or len(tickers) == 0:
            raise ValueError("At least one ticker must be provided")
        
        # Default equal weights
        if weights is None:
            weights = np.ones(len(tickers)) / len(tickers)
        else:
            weights = np.array(weights)
            if len(tickers) != len(weights):
                raise ValueError(f"Number of tickers ({len(tickers)}) must match number of weights ({len(weights)})")
            
            # Normalize weights if they don't sum to 1
            weight_sum = np.sum(weights)
            if not np.isclose(weight_sum, 1.0):
                logger.warning(f"Weights sum to {weight_sum:.4f}, normalizing to 1.0")
                weights = weights / weight_sum
        
        logger.info(f"Analyzing portfolio: {dict(zip(tickers, weights))}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
        
        # 1. Fetch and prepare data
        logger.info("Fetching market data...")
        prices = self.data_fetcher.fetch_prices(tickers, start_date, end_date)
        returns = self.data_fetcher.calculate_returns(prices)
        
        # Validate data
        if len(returns) < int(os.getenv('TRADING_DAYS', 252)):
            logger.warning("Less than 1 year of data available")
        
        # 2. Factor model analysis
        logger.info("Building factor model...")
        try:
            # Estimate complete factor model
            factor_results = self.barra_model.estimate_factor_model(returns, prices)
            
            # Portfolio risk decomposition
            risk_decomp = self.barra_model.calculate_portfolio_risk(weights)
        except Exception as e:
            logger.error(f"Factor model failed: {e}")
            risk_decomp = None
        
        # 3. VaR/CVaR calculations
        logger.info("Running risk simulations...")
        risk_metrics = {}
        
        # Parametric VaR
        risk_metrics['parametric'] = self.mc_engine.parametric_var(returns, weights, self.portfolio_value)
        
        # Historical simulation
        risk_metrics['historical'] = self.mc_engine.historical_simulation(returns, weights, self.portfolio_value)
        
        # Full Monte Carlo
        risk_metrics['monte_carlo'] = self.mc_engine.monte_carlo_simulation(returns, weights, self.portfolio_value)
        
        # Factor-based (if model available)
        if risk_decomp:
            risk_metrics['factor_based'] = self.mc_engine.factor_based_simulation(
                returns, weights, self.portfolio_value, self.barra_model
            )
        
        # 4. Check risk limits
        limit_breaches = self.check_risk_limits(risk_metrics, weights)
        
        # 5. Compile results
        results = {
            'portfolio_value': self.portfolio_value,
            'weights': dict(zip(tickers, weights)),
            'returns_data': {
                'prices': prices,
                'returns': returns,
                'start_date': start_date,
                'end_date': end_date
            },
            'risk_metrics': risk_metrics,
            'risk_decomposition': risk_decomp,
            'factor_results': factor_results if 'factor_results' in locals() else None,
            'risk_limits_check': limit_breaches,
            'tickers': tickers
        }
        
        return results
    
    def optimize_portfolio(self,
                          tickers: List[str],
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          method: str = 'sharpe',
                          constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio allocation.
        """
        # Fetch data
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        
        prices = self.data_fetcher.fetch_prices(tickers, start_date, end_date)
        returns = self.data_fetcher.calculate_returns(prices)
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(returns)
        
        # Run optimization
        if method == 'min_variance':
            result = optimizer.optimize_portfolio(objective='min_variance')
            weights = result['weights']
            metrics = result
        elif method == 'sharpe':
            result = optimizer.optimize_portfolio(objective='sharpe')
            weights = result['weights']
            metrics = result
        elif method == 'risk_parity':
            result = optimizer.optimize_portfolio(objective='risk_parity')
            weights = result['weights']
            metrics = result
        elif method == 'target_return':
            target_ret = constraints.get('target_return', 0.10)
            result = optimizer.optimize_portfolio(objective='target_return', target_return=target_ret)
            weights = result['weights']
            metrics = result
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Run risk analysis on optimal portfolio
        analysis = self.analyze_portfolio(tickers, weights, start_date, end_date)
        analysis['optimization'] = {
            'method': method,
            'weights': dict(zip(tickers, weights)),
            'metrics': metrics
        }
        
        return analysis
    
    def check_risk_limits(self, risk_metrics: Dict, weights: np.ndarray) -> Dict:
        """
        Check if portfolio breaches risk limits.
        """
        violations = []
        
        # Check VaR limit
        var_95 = risk_metrics['parametric']['VaR_0.95'] / self.portfolio_value
        if var_95 > self.risk_limits['max_var_95']:
            violations.append({
                'limit': 'VaR 95%',
                'current': var_95,
                'limit_value': self.risk_limits['max_var_95'],
                'breach': 'EXCEEDED'
            })
        
        # Check concentration
        max_weight = np.max(weights)
        if max_weight > self.risk_limits['max_concentration']:
            violations.append({
                'limit': 'Concentration',
                'current': max_weight,
                'limit_value': self.risk_limits['max_concentration'],
                'breach': 'EXCEEDED'
            })
        
        # Check Sharpe ratio
        rf_rate = float(os.getenv('RISK_FREE_RATE', 0.02))
        expected_return = risk_metrics['parametric'].get('expected_return', 0)
        annual_vol = risk_metrics['parametric'].get('annual_vol', 1)
        sharpe = (expected_return - rf_rate) / annual_vol if annual_vol > 0 else 0
        if sharpe < self.risk_limits['min_sharpe']:
            violations.append({
                'limit': 'Sharpe Ratio',
                'current': sharpe,
                'limit_value': self.risk_limits['min_sharpe'],
                'breach': 'BELOW MINIMUM'
            })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
    
    def run_stress_tests(self,
                        tickers: List[str],
                        weights: np.ndarray,
                        scenarios: Optional[List[Dict]] = None) -> Dict:
        """
        Run stress test scenarios.
        """
        if scenarios is None:
            # Default stress scenarios
            scenarios = [
                {
                    'name': 'Market Crash',
                    'factors': {'market': -0.20, 'volatility': 0.50}
                },
                {
                    'name': 'Tech Bubble Burst',
                    'factors': {'momentum': -0.30, 'size': -0.15}
                },
                {
                    'name': 'Interest Rate Shock',
                    'factors': {'value': 0.10, 'quality': 0.05}
                }
            ]
        
        # Fetch recent data for baseline
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        prices = self.data_fetcher.fetch_prices(tickers, start_date, end_date)
        returns = self.data_fetcher.calculate_returns(prices)
        
        stress_results = {}
        
        for scenario in scenarios:
            logger.info(f"Running stress test: {scenario['name']}")
            
            # Apply factor shocks
            shocks = scenario.get('factors', {})
            
            try:
                if hasattr(self.barra_model, 'factor_exposures'):
                    scenario_result = self.mc_engine.factor_based_simulation(
                        returns, weights, self.portfolio_value, self.barra_model, shocks
                    )
                else:
                    # Fallback to simple shock
                    shocked_returns = returns * (1 + scenario.get('shock', -0.10))
                    scenario_result = self.mc_engine.historical_simulation(
                        shocked_returns, weights, self.portfolio_value
                    )
                
                stress_results[scenario['name']] = scenario_result
                
            except Exception as e:
                logger.error(f"Stress test failed for {scenario['name']}: {e}")
                stress_results[scenario['name']] = {'error': str(e)}
        
        return stress_results
    
    def generate_report(self, results: Dict, output_path: Optional[str] = None):
        """
        Generate comprehensive risk report with visualizations.
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Portfolio Allocation
        ax1 = plt.subplot(2, 3, 1)
        self._plot_allocation(ax1, results)
        
        # 2. Returns Distribution
        ax2 = plt.subplot(2, 3, 2)
        self._plot_returns_distribution(ax2, results)
        
        # 3. Risk Metrics Comparison
        ax3 = plt.subplot(2, 3, 3)
        self._plot_risk_comparison(ax3, results)
        
        # 4. Factor Exposures
        ax4 = plt.subplot(2, 3, 4)
        self._plot_factor_exposures(ax4, results)
        
        # 5. VaR Visualization
        ax5 = plt.subplot(2, 3, 5)
        self._plot_var_analysis(ax5, results)
        
        # 6. Performance Metrics
        ax6 = plt.subplot(2, 3, 6)
        self._plot_performance_table(ax6, results)
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Report saved to {output_path}")
        
        plt.close(fig)  # Close the figure to free memory
        
        # Generate text report
        self._save_text_report(results)
        
        return fig
    
    def _plot_allocation(self, ax, results):
        """Plot portfolio allocation pie chart."""
        weights = results['weights']
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        wedges, texts, autotexts = ax.pie(weights.values(), labels=weights.keys(), 
                                           autopct='%1.1f%%', colors=colors,
                                           startangle=90)
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax.set_title('Portfolio Allocation', fontsize=14, pad=20)
    
    def _plot_returns_distribution(self, ax, results):
        """Plot returns distribution with normal overlay."""
        returns = results['returns_data']['returns']
        weights = np.array(list(results['weights'].values()))
        portfolio_returns = returns @ weights
        
        ax.hist(portfolio_returns, bins=50, density=True, alpha=0.7, label='Actual')
        
        # Fit normal distribution
        mu, sigma = portfolio_returns.mean(), portfolio_returns.std()
        x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label='Normal')
        
        ax.set_xlabel('Daily Returns')
        ax.set_ylabel('Density')
        ax.set_title('Returns Distribution')
        ax.legend()
    
    def _plot_risk_comparison(self, ax, results):
        """Compare VaR across different methods."""
        methods = ['parametric', 'historical', 'monte_carlo']
        var_95 = []
        var_99 = []
        
        for method in methods:
            if method in results['risk_metrics']:
                var_95.append(results['risk_metrics'][method]['VaR_0.95'])
                var_99.append(results['risk_metrics'][method]['VaR_0.99'])
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, var_95, width, label='95% VaR')
        ax.bar(x + width/2, var_99, width, label='99% VaR')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('VaR ($)')
        ax.set_title('VaR Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
    
    def _plot_factor_exposures(self, ax, results):
        """Plot factor exposures if available."""
        if results['risk_decomposition'] and 'portfolio_betas' in results['risk_decomposition']:
            betas = results['risk_decomposition']['portfolio_betas']
            # Sort by absolute value and take top 6
            sorted_betas = sorted(betas.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
            factors = [item[0] for item in sorted_betas]
            values = [item[1] for item in sorted_betas]
            
            colors = ['green' if v > 0 else 'red' for v in values]
            ax.barh(factors, values, color=colors)
            ax.set_xlabel('Portfolio Beta')
            ax.set_title('Top Factor Exposures')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Factor data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Factor Exposures')
    
    def _plot_var_analysis(self, ax, results):
        """Visualize VaR and CVaR."""
        returns = results['returns_data']['returns']
        weights = np.array(list(results['weights'].values()))
        portfolio_returns = returns @ weights
        
        # Sort returns
        sorted_returns = np.sort(portfolio_returns)
        
        # Calculate VaR thresholds
        var_95_idx = int(len(sorted_returns) * 0.05)
        var_99_idx = int(len(sorted_returns) * 0.01)
        
        ax.hist(sorted_returns, bins=50, alpha=0.7)
        ax.axvline(sorted_returns[var_95_idx], color='orange', 
                  linestyle='--', label='95% VaR')
        ax.axvline(sorted_returns[var_99_idx], color='red', 
                  linestyle='--', label='99% VaR')
        
        ax.set_xlabel('Returns')
        ax.set_ylabel('Frequency')
        ax.set_title('VaR Analysis')
        ax.legend()
    
    def _plot_performance_table(self, ax, results):
        """Display key performance metrics in a table."""
        metrics = results['risk_metrics']['parametric']
        
        # Add risk decomposition info if available
        if results['risk_decomposition']:
            risk_decomp = results['risk_decomposition']
            systematic_pct = risk_decomp['risk_decomposition']['systematic'] * 100
            idiosyncratic_pct = risk_decomp['risk_decomposition']['idiosyncratic'] * 100
        else:
            systematic_pct = 0
            idiosyncratic_pct = 0
        
        # Create metrics table
        table_data = [
            ['Annual Return', f"{metrics['expected_return']:.2%}"],
            ['Annual Volatility', f"{metrics['annual_vol']:.2%}"],
            ['Sharpe Ratio', f"{metrics['expected_return']/metrics['annual_vol']:.2f}"],
            ['95% Daily VaR', f"${metrics['VaR_0.95']:,.0f}"],
            ['99% Daily VaR', f"${metrics['VaR_0.99']:,.0f}"],
            ['Systematic Risk', f"{systematic_pct:.1f}%"],
            ['Idiosyncratic Risk', f"{idiosyncratic_pct:.1f}%"]
        ]
        
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data, 
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the header row
        for i in range(2):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Performance Metrics', fontsize=14, pad=20)
    
    def _save_text_report(self, results):
        """Save detailed text report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_outputs/risk_report_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("PORTFOLIO RISK ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Portfolio Value: ${results['portfolio_value']:,.0f}\n\n")
            
            f.write("CURRENT ALLOCATION:\n")
            for ticker, weight in results['weights'].items():
                f.write(f"  {ticker}: {weight:.1%}\n")
            
            f.write("\nRISK METRICS (Parametric):\n")
            metrics = results['risk_metrics']['parametric']
            f.write(f"  Daily VaR (95%): ${metrics['VaR_0.95']:,.0f}\n")
            f.write(f"  Daily VaR (99%): ${metrics['VaR_0.99']:,.0f}\n")
            f.write(f"  Daily CVaR (95%): {metrics['CVaR_0.95']:.2%}\n")
            f.write(f"  Daily CVaR (99%): {metrics['CVaR_0.99']:.2%}\n")
            f.write(f"  Annual Volatility: {metrics['annual_vol']:.2%}\n")
            f.write(f"  Expected Return: {metrics['expected_return']:.2%}\n")
            
            # Calculate and display Sharpe ratio
            rf_rate = float(os.getenv('RISK_FREE_RATE', 0.02))
            sharpe_ratio = (metrics['expected_return'] - rf_rate) / metrics['annual_vol'] if metrics['annual_vol'] > 0 else 0
            f.write(f"  Sharpe Ratio: {sharpe_ratio:.3f}\n")
            
            if results['risk_decomposition']:
                f.write("\nBARRA FACTOR MODEL ANALYSIS:\n")
                f.write("\nRisk Decomposition:\n")
                f.write(f"  Total Portfolio Risk: {results['risk_decomposition']['total_risk']:.2%} annual volatility\n")
                f.write(f"  Systematic Risk: {results['risk_decomposition']['factor_risk']:.2%} ({results['risk_decomposition']['risk_decomposition']['systematic']:.1%} of variance)\n")
                f.write(f"  Idiosyncratic Risk: {results['risk_decomposition']['specific_risk']:.2%} ({results['risk_decomposition']['risk_decomposition']['idiosyncratic']:.1%} of variance)\n")
                
                f.write("\nFactor Exposures (Portfolio Betas):\n")
                sorted_betas = sorted(results['risk_decomposition']['portfolio_betas'].items(), 
                                    key=lambda x: abs(x[1]), reverse=True)
                for factor, beta in sorted_betas:
                    interpretation = "positive" if beta > 0 else "negative"
                    f.write(f"  {factor:15}: {beta:>7.3f} ({interpretation} exposure)\n")
                
                f.write("\nTop Risk Contributors:\n")
                sorted_contributions = sorted(results['risk_decomposition']['factor_contributions'].items(), 
                                            key=lambda x: abs(x[1]), reverse=True)
                for factor, contrib in sorted_contributions[:5]:
                    f.write(f"  {factor:15}: {contrib:>6.1%} of total variance\n")
                
                # Add model quality section
                if hasattr(self.barra_model, 'r_squared') and self.barra_model.r_squared is not None:
                    f.write("\nModel Quality Assessment (R-squared):\n")
                    avg_r2 = self.barra_model.r_squared.mean()
                    f.write(f"  Portfolio Average: {avg_r2:.1%}")
                    quality = " (Excellent)" if avg_r2 > 0.7 else " (Good)" if avg_r2 > 0.5 else " (Fair)"
                    f.write(quality + "\n")
                    
                    f.write("\n  Individual Holdings:\n")
                    for ticker in sorted(results['weights'].keys()):
                        if ticker in self.barra_model.r_squared.index:
                            r2 = self.barra_model.r_squared[ticker]
                            quality = "High" if r2 > 0.7 else "Moderate" if r2 > 0.5 else "Low"
                            f.write(f"    {ticker}: {r2:.1%} ({quality} explanatory power)\n")
            
            f.write("\nRISK LIMITS COMPLIANCE:\n")
            if results['risk_limits_check']['compliant']:
                f.write("  Status: COMPLIANT\n")
            else:
                f.write("  Status: VIOLATIONS DETECTED\n")
                for violation in results['risk_limits_check']['violations']:
                    if violation['limit'] == 'VaR 95%':
                        f.write(f"    - {violation['limit']}: {violation['current']:.2%} > {violation['limit_value']:.2%}\n")
                    elif violation['limit'] == 'Sharpe Ratio':
                        f.write(f"    - {violation['limit']}: {violation['current']:.3f} < {violation['limit_value']:.3f}\n")
                    else:
                        f.write(f"    - {violation['limit']}: {violation['current']:.2%} > {violation['limit_value']:.2%}\n")
            
            # Add optimization results if available
            if 'optimization' in results and results['optimization']:
                f.write("\nMEAN-VARIANCE OPTIMIZED PORTFOLIO (MAX SHARPE):\n")
                f.write("\nOptimal Allocation:\n")
                for ticker, weight in results['optimization']['weights'].items():
                    f.write(f"  {ticker}: {weight:>6.1%}\n")
                
                # Calculate optimal portfolio metrics
                opt_metrics = results['optimization']['metrics']
                opt_sharpe = opt_metrics.get('sharpe_ratio', 0)
                f.write(f"\nOptimal Portfolio Metrics:\n")
                f.write(f"  Expected Return: {opt_metrics.get('expected_return', 0):.2%}\n")
                f.write(f"  Annual Volatility: {opt_metrics.get('volatility', 0):.2%}\n")
                f.write(f"  Sharpe Ratio: {opt_sharpe:.3f}\n")
                
                # Show improvement over current portfolio
                current_sharpe = sharpe_ratio
                f.write(f"\nImprovement over Current Portfolio:\n")
                f.write(f"  Sharpe Ratio: {current_sharpe:.3f} → {opt_sharpe:.3f} ")
                f.write(f"({(opt_sharpe/current_sharpe - 1)*100:+.1f}%)\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("  1. Review positions exceeding 20% allocation\n")
            f.write("  2. Consider risk parity approach for better diversification\n")
            f.write("  3. Monitor factor exposures during market stress\n")
        
        logger.info(f"Text report saved to {filename}")
    
    def display_results(self, results: Dict):
        """
        Display key results in console.
        """
        print("\n" + "="*60)
        print("PORTFOLIO RISK ANALYSIS SUMMARY")
        print("="*60)
        
        # Portfolio composition
        print(f"\nPortfolio Value: ${results['portfolio_value']:,.0f}")
        print("\nAllocation:")
        for ticker, weight in results['weights'].items():
            print(f"  {ticker}: {weight:.1%}")
        
        # Risk metrics
        metrics = results['risk_metrics']['parametric']
        print(f"\nRisk Metrics:")
        print(f"  95% VaR: ${metrics['VaR_0.95']:,.0f} ({metrics['VaR_0.95']/results['portfolio_value']:.2%})")
        print(f"  99% VaR: ${metrics['VaR_0.99']:,.0f} ({metrics['VaR_0.99']/results['portfolio_value']:.2%})")
        print(f"  95% CVaR: {metrics['CVaR_0.95']:.2%}")
        print(f"  Annual Volatility: {metrics['annual_vol']:.2%}")
        print(f"  Expected Return: {metrics['expected_return']:.2%}")
        rf_rate = float(os.getenv('RISK_FREE_RATE', 0.02))
        sharpe = (metrics['expected_return'] - rf_rate) / metrics['annual_vol'] if metrics['annual_vol'] > 0 else 0
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        
        # Risk limits
        print(f"\nRisk Limits Check: {'✓ COMPLIANT' if results['risk_limits_check']['compliant'] else '✗ VIOLATIONS'}")
        if not results['risk_limits_check']['compliant']:
            for violation in results['risk_limits_check']['violations']:
                print(f"  - {violation['limit']}: {violation['breach']}")
        
        print("\n" + "="*60)


# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    risk_system = RiskManagementSystem(portfolio_value=4_000_000)
    risk_system.initialize()
    
    # Example portfolio
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    weights = [0.25, 0.20, 0.20, 0.20, 0.15]
    
    # Run analysis
    print("Running portfolio analysis...")
    results = risk_system.analyze_portfolio(tickers, weights)
    
    # Display results
    risk_system.display_results(results)
    
    # Run optimization before generating report
    print("\nOptimizing portfolio (Maximum Sharpe Ratio)...")
    optimal_results = risk_system.optimize_portfolio(tickers, method='sharpe')
    
    # Add optimization results to main results for report
    results['optimization'] = optimal_results['optimization']
    
    # Generate report
    print("\nGenerating risk report...")
    risk_system.generate_report(results, 'model_outputs/portfolio_risk_report.png')
    
    # Generate Monte Carlo visualization
    print("\nGenerating Monte Carlo paths...")
    risk_system.mc_engine.plot_monte_carlo_paths(
        results['returns_data']['returns'],
        np.array(weights),
        output_path='model_outputs/monte_carlo_paths.png'
    )
    
    # Generate Efficient Frontier visualization
    print("\nGenerating Efficient Frontier...")
    optimizer = PortfolioOptimizer(results['returns_data']['returns'])
    optimizer.plot_efficient_frontier(
        current_weights=np.array(weights),
        output_path='model_outputs/professional_frontier.png'
    )
    
    # Generate Barra Factor visualizations
    print("\nGenerating Barra Factor visualizations...")
    # Factor exposures
    fig1 = risk_system.barra_model.plot_factor_exposures(ticker_subset=tickers)
    plt.savefig('model_outputs/barra_factor_exposures.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Risk decomposition
    fig2 = risk_system.barra_model.plot_risk_decomposition(np.array(weights))
    plt.savefig('model_outputs/barra_risk_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Factor returns
    fig3 = risk_system.barra_model.plot_factor_returns()
    plt.savefig('model_outputs/barra_factor_returns.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # R-squared distribution
    fig4 = risk_system.barra_model.plot_r_squared_distribution()
    plt.savefig('model_outputs/barra_r_squared.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # Display optimization results
    print("\n" + "="*60)
    print("MEAN-VARIANCE OPTIMIZED PORTFOLIO (MAX SHARPE)")
    print("="*60)
    print("\nOptimal Allocation:")
    for ticker, weight in optimal_results['optimization']['weights'].items():
        print(f"  {ticker}: {weight:>6.1%}")
    
    # Display optimal portfolio metrics
    opt_metrics = optimal_results['risk_metrics']['parametric']
    opt_sharpe = (opt_metrics['expected_return'] - float(os.getenv('RISK_FREE_RATE', 0.02))) / opt_metrics['annual_vol']
    print(f"\nOptimal Portfolio Metrics:")
    print(f"  Expected Return: {opt_metrics['expected_return']:.2%}")
    print(f"  Annual Volatility: {opt_metrics['annual_vol']:.2%}")
    print(f"  Sharpe Ratio: {opt_sharpe:.3f}")
    print(f"  95% VaR: ${optimal_results['risk_metrics']['parametric']['VaR_0.95']:,.0f}")
    print("="*60)
    
    # Create visualization comparing current vs optimal portfolio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Current portfolio pie chart
    colors = plt.cm.Set3(np.arange(len(tickers)))
    ax1.pie(weights, labels=tickers, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Current Portfolio Allocation', fontsize=14, pad=20)
    
    # Optimal portfolio pie chart
    optimal_weights = [optimal_results['optimization']['weights'][ticker] for ticker in tickers]
    ax2.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Mean-Variance Optimized Portfolio\n(Maximum Sharpe Ratio)', fontsize=14, pad=20)
    
    # Add metrics comparison
    current_sharpe = (results['risk_metrics']['parametric']['expected_return'] - float(os.getenv('RISK_FREE_RATE', 0.02))) / results['risk_metrics']['parametric']['annual_vol']
    
    fig.text(0.5, 0.02, 
             f'Current: Sharpe={current_sharpe:.3f}, Vol={results["risk_metrics"]["parametric"]["annual_vol"]:.1%}, Return={results["risk_metrics"]["parametric"]["expected_return"]:.1%}\n'
             f'Optimal: Sharpe={opt_sharpe:.3f}, Vol={opt_metrics["annual_vol"]:.1%}, Return={opt_metrics["expected_return"]:.1%}',
             ha='center', fontsize=11, style='italic')
    
    plt.suptitle('Portfolio Optimization Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('model_outputs/portfolio_optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Run stress tests
    print("\nRunning stress tests...")
    stress_results = risk_system.run_stress_tests(tickers, np.array(weights))
    for scenario, result in stress_results.items():
        if 'VaR_0.95' in result:
            print(f"  {scenario}: VaR = ${result['VaR_0.95']:,.0f}")
    
    print("\nAnalysis complete!")