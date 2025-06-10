import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class BarraFactorModel:
    def __init__(self):
        self.factor_exposures = None
        self.factor_returns = None
        self.specific_returns = None
        self.factor_covariance = None
        self.specific_risk = None
        self.r_squared = None
        
        # Factor ETFs - mix of sectors and style/trend factors
        self.FACTOR_ETFS = {
            'SPY': 'Market',          # S&P 500 market factor
            'XLF': 'Financials',      # Financial sector
            'XLK': 'Technology',      # Technology sector  
            'XLE': 'Energy',          # Energy sector
            'XLV': 'Healthcare',      # Healthcare sector
            'MTUM': 'Momentum',       # Momentum factor
            'QUAL': 'Quality',        # Quality factor
            'USMV': 'Low_Volatility'  # Low volatility factor
        }
        
    def fetch_factor_etf_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch factor ETF returns.
        """
        import yfinance as yf
        
        try:
            # Download ETF data
            etf_tickers = list(self.FACTOR_ETFS.keys())
            logger.info(f"Fetching factor ETF data: {etf_tickers}")
            
            data = yf.download(
                etf_tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )['Close']
            
            # Calculate returns
            factor_returns = np.log(data / data.shift(1)).iloc[1:]
            
            # Rename columns to factor names
            factor_returns.columns = [self.FACTOR_ETFS[ticker] for ticker in etf_tickers]
            
            return factor_returns
            
        except Exception as e:
            logger.error(f"Error fetching ETF data: {e}")
            raise
    
    def estimate_factor_model(self, 
                            returns: pd.DataFrame, 
                            prices: pd.DataFrame) -> Dict:
        """
        Estimate factor model using ETF returns:
        1. Fetch factor ETF returns
        2. Run time-series regression of asset returns on factor returns
        """
        # Get date range from returns
        start_date = returns.index[0] - pd.Timedelta(days=10)
        end_date = returns.index[-1]
        
        # Step 1: Fetch factor ETF returns
        factor_returns = self.fetch_factor_etf_returns(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Step 2: Align data
        common_dates = returns.index.intersection(factor_returns.index)
        if len(common_dates) < 60:
            raise ValueError(f"Insufficient overlapping data: {len(common_dates)} days")
            
        returns_aligned = returns.loc[common_dates]
        factors_aligned = factor_returns.loc[common_dates]
        
        # Store results
        betas = pd.DataFrame(index=returns.columns, columns=factors_aligned.columns)
        alphas = pd.Series(index=returns.columns)
        specific_vols = pd.Series(index=returns.columns)
        r2_values = pd.Series(index=returns.columns)
        
        # Time-series regression for each asset
        for asset in returns.columns:
            y = returns_aligned[asset].values
            X = factors_aligned.values
            
            # Skip if we have NaN values
            mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
            if mask.sum() < 60:
                logger.warning(f"Insufficient data for {asset}")
                continue
                
            y_clean = y[mask]
            X_clean = X[mask]
            
            # Add intercept
            X_with_const = np.column_stack([np.ones(len(X_clean)), X_clean])
            
            # Run regression
            model = LinearRegression(fit_intercept=False)
            model.fit(X_with_const, y_clean)
            
            # Extract results
            alphas[asset] = model.coef_[0]
            betas.loc[asset] = model.coef_[1:]
            
            # Calculate R-squared
            y_pred = model.predict(X_with_const)
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
            r2_values[asset] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Specific risk (residual volatility)
            residuals = y_clean - y_pred
            trading_days = int(os.getenv('TRADING_DAYS', 252))
            specific_vols[asset] = residuals.std() * np.sqrt(trading_days)
        
        # Factor covariance matrix
        trading_days = int(os.getenv('TRADING_DAYS', 252))
        factor_cov = factors_aligned.cov() * trading_days
        
        # Store results
        self.factor_exposures = betas.fillna(0)
        self.factor_covariance = factor_cov
        self.specific_risk = specific_vols.fillna(specific_vols.mean())
        self.r_squared = r2_values.fillna(0)
        self.factor_returns = factors_aligned
        
        logger.info(f"Factor model fitted. Mean R²: {r2_values.mean():.3f}")
        
        return {
            'betas': betas,
            'alphas': alphas,
            'specific_risk': specific_vols,
            'r_squared': r2_values,
            'factor_covariance': self.factor_covariance
        }
    
    
    def calculate_portfolio_risk(self, weights: np.ndarray) -> Dict:
        """
        Decompose portfolio risk into factor and specific components.
        """
        if self.factor_exposures is None:
            raise ValueError("Model not fitted yet")
        
        # Portfolio factor exposures
        portfolio_betas = self.factor_exposures.T @ weights
        
        # Factor risk contribution
        factor_variance = portfolio_betas @ self.factor_covariance @ portfolio_betas
        factor_risk = np.sqrt(factor_variance)
        
        # Specific risk contribution
        specific_variance = np.sum((weights ** 2) * (self.specific_risk ** 2))
        specific_risk = np.sqrt(specific_variance)
        
        # Total risk
        total_variance = factor_variance + specific_variance
        total_risk = np.sqrt(total_variance)
        
        # Risk attribution by factor
        marginal_contributions = self.factor_covariance @ portfolio_betas
        factor_contributions = {}
        
        for i, factor in enumerate(self.factor_covariance.columns):
            # Contribution to variance
            var_contrib = portfolio_betas.iloc[i] * marginal_contributions.iloc[i]
            # Percentage of total variance
            factor_contributions[factor] = var_contrib / total_variance
        
        return {
            'total_risk': total_risk,
            'factor_risk': factor_risk,
            'specific_risk': specific_risk,
            'factor_contributions': factor_contributions,
            'portfolio_betas': dict(zip(self.factor_covariance.columns, portfolio_betas)),
            'risk_decomposition': {
                'systematic': factor_variance / total_variance,
                'idiosyncratic': specific_variance / total_variance
            }
        }
    
    def predict_returns(self, 
                       factor_views: Dict[str, float],
                       horizon: int = 21) -> pd.Series:
        """
        Predict asset returns given factor views.
        Useful for tactical allocation.
        """
        if self.factor_exposures is None:
            raise ValueError("Model not fitted")
        
        # Convert views to array
        factor_predictions = np.array([
            factor_views.get(factor, 0) 
            for factor in self.factor_exposures.columns
        ])
        
        # Scale views to horizon
        trading_days = int(os.getenv('TRADING_DAYS', 252))
        factor_predictions = factor_predictions * horizon / trading_days
        
        # Predict returns: R = Beta * F
        predicted_returns = self.factor_exposures @ factor_predictions
        
        return predicted_returns
    
    def plot_factor_exposures(self, ticker_subset: Optional[List[str]] = None, 
                              figsize: Tuple[int, int] = (14, 7)) -> plt.Figure:
        """
        Visualize factor exposures (betas) as a clean heatmap.
        Shows how much each asset is exposed to each risk factor.
        """
        if self.factor_exposures is None:
            raise ValueError("Model not fitted. Run estimate_factor_model first.")
        
        # Select subset of tickers if specified
        if ticker_subset:
            exposures = self.factor_exposures.loc[ticker_subset]
        else:
            exposures = self.factor_exposures
        
        # Convert to numeric and fill NaNs
        exposures = exposures.astype(float).fillna(0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine appropriate scale based on data
        max_exposure = max(abs(exposures.values.min()), abs(exposures.values.max()))
        vmax = min(2.0, max(1.0, np.ceil(max_exposure * 10) / 10))  # Round up to nearest 0.1
        
        # Use a better colormap and scale
        im = ax.imshow(exposures.values, cmap='RdBu_r', aspect='auto', 
                       vmin=-vmax, vmax=vmax)
        
        # Set ticks
        ax.set_xticks(np.arange(len(exposures.columns)))
        ax.set_yticks(np.arange(len(exposures.index)))
        ax.set_xticklabels(exposures.columns, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(exposures.index, fontsize=12)
        
        # Add text annotations for all values
        for i in range(len(exposures.index)):
            for j in range(len(exposures.columns)):
                value = exposures.iloc[i, j]
                text_color = 'white' if abs(value) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{value:.2f}',
                        ha="center", va="center",
                        color=text_color,
                        fontsize=12, weight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Factor Beta', fontsize=13)
        cbar.ax.tick_params(labelsize=11)
        
        ax.set_title('Portfolio Factor Exposures', fontsize=18, pad=20, weight='bold')
        ax.set_xlabel('Market Factors', fontsize=14)
        ax.set_ylabel('Portfolio Holdings', fontsize=14)
        
        # Remove all grid lines and spines
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(which="both", size=0)
        
        plt.tight_layout()
        return fig
    
    def plot_risk_decomposition(self, weights: np.ndarray,
                                figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Clean portfolio risk decomposition visualization.
        """
        if self.factor_exposures is None:
            raise ValueError("Model not fitted. Run estimate_factor_model first.")
        
        # Calculate risk decomposition
        risk_decomp = self.calculate_portfolio_risk(weights)
        
        # Create figure with single subplot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Calculate values
        total_vol = risk_decomp['total_risk'] * 100
        systematic_vol = risk_decomp['factor_risk'] * 100
        idiosyncratic_vol = risk_decomp['specific_risk'] * 100
        
        # Variance percentages
        systematic_var_pct = risk_decomp['risk_decomposition']['systematic'] * 100
        idiosyncratic_var_pct = risk_decomp['risk_decomposition']['idiosyncratic'] * 100
        
        # Create three vertical bars for volatilities
        x_pos = np.arange(3)
        volatilities = [total_vol, systematic_vol, idiosyncratic_vol]
        labels = ['Total Risk', 'Systematic Risk', 'Idiosyncratic Risk']
        colors = ['#1f77b4', '#004B87', '#7f7f7f']
        
        bars = ax.bar(x_pos, volatilities, color=colors, width=0.6, edgecolor='none')
        
        # Add value labels on bars
        for i, (bar, vol) in enumerate(zip(bars, volatilities)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{vol:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        # Add variance contribution labels for systematic and idiosyncratic
        ax.text(1, systematic_vol/2, f'{systematic_var_pct:.0f}%\nof variance', 
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')
        ax.text(2, idiosyncratic_vol/2, f'{idiosyncratic_var_pct:.0f}%\nof variance', 
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')
        
        # Styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax.set_ylabel('Annual Volatility (%)', fontsize=13, fontweight='bold')
        ax.set_title('Portfolio Risk Decomposition', fontsize=16, pad=20, fontweight='bold')
        ax.set_ylim(0, max(volatilities) * 1.15)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid
        ax.yaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add formula explanation at bottom
        formula_text = f'Total Risk² ({total_vol:.1f}²) = Systematic Risk² ({systematic_vol:.1f}²) + Idiosyncratic Risk² ({idiosyncratic_vol:.1f}²)'
        ax.text(0.5, -0.12, formula_text, transform=ax.transAxes, 
                ha='center', fontsize=11, style='italic', color='#666666')
        
        # Add key factor exposures as text box
        portfolio_betas = risk_decomp['portfolio_betas']
        sorted_betas = sorted(portfolio_betas.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        exposure_text = 'Top Factor Exposures:\n'
        for factor, beta in sorted_betas:
            sign = '+' if beta > 0 else ''
            exposure_text += f'{factor}: {sign}{beta:.2f}\n'
        
        ax.text(0.98, 0.95, exposure_text.strip(), transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#004B87', linewidth=2),
                verticalalignment='top', horizontalalignment='right', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def plot_factor_returns(self, lookback_days: int = None, 
                            figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Plot cumulative factor performance with better visualization.
        """
        if self.factor_returns is None:
            raise ValueError("Model not fitted. Run estimate_factor_model first.")
        
        # Use default trading days if not specified
        if lookback_days is None:
            lookback_days = int(os.getenv('TRADING_DAYS', 252))
        
        # Get recent factor returns
        recent_returns = self.factor_returns.iloc[-lookback_days:]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + recent_returns).cumprod() - 1
        
        # Define better colors for each factor
        factor_colors = {
            'Market': '#1f77b4',      # Blue
            'Financials': '#2ca02c',   # Green
            'Technology': '#ff7f0e',   # Orange
            'Energy': '#d62728',       # Red
            'Healthcare': '#9467bd',   # Purple
            'Momentum': '#8c564b',     # Brown
            'Quality': '#e377c2',      # Pink
            'Low_Volatility': '#7f7f7f' # Gray
        }
        
        # Plot all factors with proper colors
        for factor in cumulative_returns.columns:
            color = factor_colors.get(factor, '#17becf')
            ax.plot(cumulative_returns.index, cumulative_returns[factor], 
                    label=factor, linewidth=2.5, color=color)
        
        # Formatting
        ax.set_title('Risk Factor Performance Over Time\nCumulative returns of factor ETFs', 
                     fontsize=16, pad=20)
        ax.set_xlabel('Date', fontsize=13)
        ax.set_ylabel('Cumulative Return', fontsize=13)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Better legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, ncol=2)
        
        # Grid and zero line
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Highlight significant levels
        ax.axhline(y=0.1, color='green', linestyle=':', alpha=0.5)
        ax.axhline(y=-0.1, color='red', linestyle=':', alpha=0.5)
        
        # Add annotations for best/worst performers
        final_returns = cumulative_returns.iloc[-1]
        best_factor = final_returns.idxmax()
        worst_factor = final_returns.idxmin()
        
        ax.text(0.02, 0.98, f'Best: {best_factor} ({final_returns[best_factor]:.1%})',
                transform=ax.transAxes, va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.text(0.02, 0.91, f'Worst: {worst_factor} ({final_returns[worst_factor]:.1%})',
                transform=ax.transAxes, va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def plot_r_squared_distribution(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Clean institutional R-squared visualization.
        """
        if self.r_squared is None:
            raise ValueError("Model not fitted. Run estimate_factor_model first.")
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Sort R-squared values descending
        r2_sorted = self.r_squared.sort_values(ascending=False)
        
        # Create gradient from dark to light cobalt blue based on R² values
        colors = []
        for r2 in r2_sorted.values:
            if r2 >= 0.7:
                colors.append('#003f7f')  # Dark cobalt for high R²
            elif r2 >= 0.5:
                colors.append('#004B87')  # Medium cobalt
            else:
                colors.append('#5B9BD5')  # Light cobalt for low R²
        
        # Create vertical bar chart
        bars = ax.bar(range(len(r2_sorted)), r2_sorted.values, 
                      color=colors, width=0.7, edgecolor='none')
        
        # Add asset names
        ax.set_xticks(range(len(r2_sorted)))
        ax.set_xticklabels(r2_sorted.index, rotation=0, ha='center', fontsize=12)
        
        # Add value labels on top of bars
        for i, (asset, r2) in enumerate(r2_sorted.items()):
            ax.text(i, r2 + 0.02, f'{r2:.0%}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold', color='#333333')
        
        # Clean styling
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('R² (Explanatory Power)', fontsize=13, fontweight='bold')
        ax.set_title('Factor Model Quality by Asset', fontsize=15, pad=20, fontweight='bold')
        
        # Remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add horizontal grid only
        ax.yaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Add threshold line at 0.5
        ax.axhline(0.5, color='#666666', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add average as text annotation
        avg_r2 = self.r_squared.mean()
        ax.text(0.98, 0.95, f'Portfolio Average\nR² = {avg_r2:.0%}', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#004B87', linewidth=2),
                fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    """Test and display all Barra visualizations."""
    from data_fetch import DataFetcher
    import matplotlib.pyplot as plt
    
    print("Testing Barra Factor Model Visualizations")
    print("="*50)
    
    # Initialize components
    data_fetcher = DataFetcher()
    barra_model = BarraFactorModel()
    
    # Portfolio setup
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
    
    # Fetch data
    print("\nFetching market data...")
    start_date = '2022-06-10'
    end_date = '2025-06-09'
    prices = data_fetcher.fetch_prices(tickers, start_date, end_date)
    returns = data_fetcher.calculate_returns(prices)
    
    # Estimate factor model
    print("Estimating factor model...")
    results = barra_model.estimate_factor_model(returns, prices)
    
    print(f"\nModel Results:")
    print(f"Mean R²: {results['r_squared'].mean():.3f}")
    print(f"Factor Covariance shape: {results['factor_covariance'].shape}")
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    
    # 1. Factor Exposures Heatmap
    print("1. Factor Exposures Heatmap")
    fig1 = barra_model.plot_factor_exposures()
    plt.savefig('barra_factor_exposures.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Risk Decomposition
    print("\n2. Portfolio Risk Decomposition")
    fig2 = barra_model.plot_risk_decomposition(weights)
    plt.savefig('barra_risk_decomposition.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. Factor Returns
    print("\n3. Factor ETF Performance")
    fig3 = barra_model.plot_factor_returns()
    plt.savefig('barra_factor_returns.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4. R-squared Distribution
    print("\n4. Model Fit (R-squared)")
    fig4 = barra_model.plot_r_squared_distribution()
    plt.savefig('barra_r_squared.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print risk decomposition details
    risk_decomp = barra_model.calculate_portfolio_risk(weights)
    print("\n" + "="*50)
    print("PORTFOLIO RISK ANALYSIS")
    print("="*50)
    print(f"Total Risk: {risk_decomp['total_risk']:.2%}")
    print(f"Systematic Risk: {risk_decomp['factor_risk']:.2%} ({risk_decomp['risk_decomposition']['systematic']:.1%})")
    print(f"Idiosyncratic Risk: {risk_decomp['specific_risk']:.2%} ({risk_decomp['risk_decomposition']['idiosyncratic']:.1%})")
    
    print("\nFactor Contributions:")
    for factor, contrib in risk_decomp['factor_contributions'].items():
        print(f"  {factor}: {contrib:.1%}")
    
    print("\nPortfolio Factor Exposures:")
    for factor, beta in risk_decomp['portfolio_betas'].items():
        print(f"  {factor}: {beta:.3f}")
    
    print("\n✓ All visualizations generated and saved!")