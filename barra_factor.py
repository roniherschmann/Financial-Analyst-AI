import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import logging

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
            specific_vols[asset] = residuals.std() * np.sqrt(252)
        
        # Factor covariance matrix
        factor_cov = factors_aligned.cov() * 252
        
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
        factor_predictions = factor_predictions * horizon / 252
        
        # Predict returns: R = Beta * F
        predicted_returns = self.factor_exposures @ factor_predictions
        
        return predicted_returns
    
    def plot_factor_exposures(self, ticker_subset: Optional[List[str]] = None, 
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Visualize factor exposures (betas) as a cleaner heatmap.
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
        
        # Create heatmap with fewer annotations
        im = ax.imshow(exposures.values, cmap='RdBu_r', aspect='auto', 
                       vmin=-1.5, vmax=1.5)
        
        # Set ticks
        ax.set_xticks(np.arange(len(exposures.columns)))
        ax.set_yticks(np.arange(len(exposures.index)))
        ax.set_xticklabels(exposures.columns, rotation=45, ha='right')
        ax.set_yticklabels(exposures.index)
        
        # Add text annotations only for significant values
        for i in range(len(exposures.index)):
            for j in range(len(exposures.columns)):
                value = exposures.iloc[i, j]
                if abs(value) > 0.1:  # Only show significant exposures
                    ax.text(j, i, f'{value:.2f}',
                            ha="center", va="center",
                            color="white" if abs(value) > 0.7 else "black",
                            fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Factor Beta', fontsize=12)
        
        ax.set_title('Factor Exposures by Asset', fontsize=16, pad=20)
        ax.set_xlabel('Factor ETFs', fontsize=12)
        ax.set_ylabel('Assets', fontsize=12)
        
        # Add grid
        ax.set_xticks(np.arange(len(exposures.columns))-.5, minor=True)
        ax.set_yticks(np.arange(len(exposures.index))-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_risk_decomposition(self, weights: np.ndarray,
                                figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Visualize portfolio risk decomposition - simplified version.
        """
        if self.factor_exposures is None:
            raise ValueError("Model not fitted. Run estimate_factor_model first.")
        
        # Calculate risk decomposition
        risk_decomp = self.calculate_portfolio_risk(weights)
        
        # Create figure with just 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Risk decomposition donut chart
        risk_data = [risk_decomp['factor_risk']**2, risk_decomp['specific_risk']**2]
        labels = ['Systematic\nRisk', 'Idiosyncratic\nRisk']
        colors = ['#3498db', '#e74c3c']
        
        _, _, autotexts = ax1.pie(risk_data, labels=labels, colors=colors, 
                                   autopct='%1.0f%%', startangle=90,
                                   wedgeprops=dict(width=0.5))
        # Make percentage text larger
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')
        
        ax1.set_title('Risk Decomposition', fontsize=16, pad=20)
        
        # 2. Top factor exposures (only show significant ones)
        portfolio_betas = risk_decomp['portfolio_betas']
        # Sort by absolute value and take top 5
        sorted_betas = sorted(portfolio_betas.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        factors = [item[0] for item in sorted_betas]
        betas = [item[1] for item in sorted_betas]
        
        colors = ['#2ecc71' if b > 0 else '#e74c3c' for b in betas]
        bars = ax2.barh(factors, betas, color=colors)
        
        ax2.set_xlabel('Portfolio Beta', fontsize=12)
        ax2.set_title('Top Factor Exposures', fontsize=16, pad=20)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for bar, beta in zip(bars, betas):
            width = bar.get_width()
            ax2.text(width + 0.02 if width > 0 else width - 0.02, bar.get_y() + bar.get_height()/2,
                     f'{beta:.2f}', ha='left' if width > 0 else 'right', va='center')
        
        # Add summary text
        fig.text(0.5, 0.02, 
                f'Total Risk: {risk_decomp["total_risk"]:.1%} | '
                f'Systematic: {risk_decomp["factor_risk"]:.1%} | '
                f'Idiosyncratic: {risk_decomp["specific_risk"]:.1%}',
                ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        return fig
    
    def plot_factor_returns(self, lookback_days: int = 252, 
                            figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot cumulative factor performance - simplified.
        """
        if self.factor_returns is None:
            raise ValueError("Model not fitted. Run estimate_factor_model first.")
        
        # Get recent factor returns
        recent_returns = self.factor_returns.iloc[-lookback_days:]
        
        # Create single plot for cumulative returns
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + recent_returns).cumprod() - 1
        
        # Plot only the most relevant factors (exclude market to reduce clutter)
        factors_to_plot = [col for col in cumulative_returns.columns if col != 'Market']
        
        # Use a color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(factors_to_plot)))
        
        for factor, color in zip(factors_to_plot, colors):
            ax.plot(cumulative_returns.index, cumulative_returns[factor], 
                    label=factor, linewidth=2, color=color)
        
        ax.set_title('Factor ETF Performance (1 Year)', fontsize=16, pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Improve legend
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add shaded regions for market regimes (optional)
        ax.axhspan(-0.1, 0, alpha=0.1, color='red')
        ax.axhspan(0, 0.5, alpha=0.1, color='green')
        
        plt.tight_layout()
        return fig
    
    def plot_r_squared_distribution(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot R-squared distribution to assess model fit quality.
        """
        if self.r_squared is None:
            raise ValueError("Model not fitted. Run estimate_factor_model first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Histogram of R-squared values
        ax1.hist(self.r_squared.values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(self.r_squared.mean(), color='red', linestyle='--', 
                    label=f'Mean: {self.r_squared.mean():.3f}')
        ax1.set_title('Distribution of R-squared Values', fontsize=14)
        ax1.set_xlabel('R-squared')
        ax1.set_ylabel('Number of Assets')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sorted R-squared values
        sorted_r2 = self.r_squared.sort_values(ascending=False)
        ax2.bar(range(len(sorted_r2)), sorted_r2.values, color='lightgreen')
        ax2.set_title('R-squared by Asset (Sorted)', fontsize=14)
        ax2.set_xlabel('Asset Rank')
        ax2.set_ylabel('R-squared')
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='R² = 0.5')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add text box with summary statistics
        textstr = f'Mean R²: {self.r_squared.mean():.3f}\n'
        textstr += f'Median R²: {self.r_squared.median():.3f}\n'
        textstr += f'Min R²: {self.r_squared.min():.3f}\n'
        textstr += f'Max R²: {self.r_squared.max():.3f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.65, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        
        plt.suptitle('Barra Factor Model - Goodness of Fit', fontsize=16)
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