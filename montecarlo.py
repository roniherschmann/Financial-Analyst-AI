# montecarlo.py
"""
Production Monte Carlo engine with correct risk metrics.
Key fixes: proper path generation, correct VaR/CVaR calculation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class MonteCarloEngine:
    def __init__(self, 
                 n_simulations: int = 10000,
                 time_horizon: int = 21,  # 21 days = 1 month
                 confidence_levels: List[float] = None):
        self.n_sims = n_simulations
        self.horizon = time_horizon
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        
    def parametric_var(self, 
                      returns: pd.DataFrame, 
                      weights: np.ndarray,
                      portfolio_value: float,
                      method: str = 'normal') -> Dict:
        """
        Analytical VaR calculation - fast and suitable for liquid portfolios.
        """
        # Portfolio returns
        portfolio_returns = returns @ weights
        
        # Calculate moments
        daily_mean = portfolio_returns.mean()
        daily_vol = portfolio_returns.std()
        
        # Scale to horizon
        horizon_mean = daily_mean * self.horizon
        horizon_vol = daily_vol * np.sqrt(self.horizon)
        
        results = {
            'daily_vol': daily_vol,
            'annual_vol': daily_vol * np.sqrt(int(os.getenv('TRADING_DAYS', 252))),
            'expected_return': daily_mean * int(os.getenv('TRADING_DAYS', 252))
        }
        
        if method == 'normal':
            # Normal VaR
            for conf in self.confidence_levels:
                z_score = stats.norm.ppf(1 - conf)
                # VaR as percentage loss over the horizon
                var_pct = -(horizon_mean + z_score * horizon_vol)
                # Convert VaR to dollar amount
                var_dollars = var_pct * portfolio_value
                # CVaR as percentage (expected loss beyond VaR)
                # Correct formula: CVaR = μ - σ * φ(z) / α where α = 1 - conf
                alpha = 1 - conf
                cvar_pct = -(horizon_mean - horizon_vol * stats.norm.pdf(z_score) / alpha)
                
                results[f'VaR_{conf}'] = var_dollars  # Dollar amount
                results[f'CVaR_{conf}'] = cvar_pct    # Percentage
                
        elif method == 'cornish-fisher':
            # Adjusted for skewness and kurtosis
            skew = portfolio_returns.skew()
            kurt = portfolio_returns.kurtosis()
            
            for conf in self.confidence_levels:
                z = stats.norm.ppf(1 - conf)
                
                # Cornish-Fisher expansion
                z_cf = z + (z**2 - 1) * skew / 6 + \
                       (z**3 - 3*z) * (kurt - 3) / 24 - \
                       (2*z**3 - 5*z) * skew**2 / 36
                
                var_pct = -(horizon_mean + z_cf * horizon_vol)
                var_dollars = var_pct * portfolio_value
                
                # CVaR using Cornish-Fisher expansion
                # For non-normal distributions, CVaR needs adjustment
                alpha = 1 - conf
                # Approximate CVaR with skewness and kurtosis adjustments
                cvar_adjustment = 1 + (skew / 6) * (z_cf**2 - 1) + (kurt - 3) / 24 * (z_cf**3 - 3*z_cf)
                cvar_pct = var_pct * cvar_adjustment
                
                results[f'VaR_{conf}'] = var_dollars  # Dollar amount
                results[f'CVaR_{conf}'] = cvar_pct    # Percentage
        
        return results
    
    def historical_simulation(self, 
                            returns: pd.DataFrame, 
                            weights: np.ndarray,
                            portfolio_value: float) -> Dict:
        """
        Historical simulation with proper bootstrap.
        Best for capturing empirical return distribution.
        """
        # Portfolio returns
        portfolio_returns = returns @ weights
        n_historical = len(portfolio_returns)
        
        # Check if we have enough data
        if n_historical < int(os.getenv('TRADING_DAYS', 252)):
            logger.warning("Less than 1 year of data for historical simulation")
        
        # Generate paths using block bootstrap
        block_size = min(20, n_historical // 10)  # Preserve autocorrelation
        n_blocks = (self.horizon // block_size) + 1
        
        final_returns = np.zeros(self.n_sims)
        
        for sim in range(self.n_sims):
            path_returns = []
            
            for _ in range(n_blocks):
                # Random block start
                start_idx = np.random.randint(0, n_historical - block_size)
                block = portfolio_returns.iloc[start_idx:start_idx + block_size].values
                path_returns.extend(block)
            
            # Trim to exact horizon
            path_returns = np.array(path_returns[:self.horizon])
            
            # Compound returns for the period
            final_returns[sim] = np.prod(1 + path_returns) - 1
        
        # Calculate risk metrics
        results = self._calculate_var_cvar(final_returns, portfolio_value)
        results['method'] = 'historical'
        results['n_observations'] = n_historical
        
        return results
    
    def monte_carlo_simulation(self, 
                             returns: pd.DataFrame, 
                             weights: np.ndarray,
                             portfolio_value: float,
                             use_t_dist: bool = True) -> Dict:
        """
        Full Monte Carlo simulation with Student-t distribution option.
        """
        # Portfolio parameters
        portfolio_returns = returns @ weights
        daily_mean = portfolio_returns.mean()
        daily_vol = portfolio_returns.std()
        
        # Fit distribution
        if use_t_dist:
            # Fit Student-t distribution for fat tails
            params = stats.t.fit(portfolio_returns)
            df, loc, scale = params
            logger.info(f"Fitted t-distribution with df={df:.2f}")
        
        # Generate paths
        final_values = np.zeros(self.n_sims)
        
        for sim in range(self.n_sims):
            path_value = 1.0
            
            for t in range(self.horizon):
                if use_t_dist and df < 30:  # Use t-dist only if significantly different from normal
                    daily_return = stats.t.rvs(df, loc=daily_mean, scale=daily_vol)
                else:
                    daily_return = np.random.normal(daily_mean, daily_vol)
                
                path_value *= (1 + daily_return)
            
            final_values[sim] = path_value
        
        # Convert to returns
        final_returns = final_values - 1
        
        # Calculate metrics
        results = self._calculate_var_cvar(final_returns, portfolio_value)
        results['distribution'] = 't-dist' if use_t_dist else 'normal'
        
        # Add path statistics
        results['paths'] = {
            'min': np.min(final_values),
            'max': np.max(final_values),
            'median': np.median(final_values),
            'mean': np.mean(final_values)
        }
        
        return results
    
    def factor_based_simulation(self,
                              returns: pd.DataFrame,
                              weights: np.ndarray,
                              portfolio_value: float,
                              barra_model,
                              stressed_factors: Optional[Dict[str, float]] = None) -> Dict:
        """
        Factor-based Monte Carlo with optional stress scenarios.
        Most sophisticated approach for factor-driven portfolios.
        """
        if not hasattr(barra_model, 'factor_exposures') or barra_model.factor_exposures is None:
            raise ValueError("Barra model must be fitted first")
        
        # Portfolio factor exposures
        portfolio_betas = barra_model.factor_exposures.T @ weights
        
        # Factor covariance
        factor_cov = barra_model.factor_covariance.values
        n_factors = len(factor_cov)
        
        # Specific risk
        specific_risks = barra_model.specific_risk.values
        portfolio_specific_risk = np.sqrt(np.sum((weights ** 2) * (specific_risks ** 2)))
        
        # Cholesky decomposition for correlated factor draws
        try:
            L = np.linalg.cholesky(factor_cov)
        except np.linalg.LinAlgError:
            # If not positive definite, add small diagonal
            factor_cov += np.eye(n_factors) * 1e-6
            L = np.linalg.cholesky(factor_cov)
        
        # Daily scaling
        trading_days = int(os.getenv('TRADING_DAYS', 252))
        daily_L = L / np.sqrt(trading_days)
        daily_specific_vol = portfolio_specific_risk / np.sqrt(trading_days)
        
        # Simulate
        final_values = np.zeros(self.n_sims)
        
        for sim in range(self.n_sims):
            path_value = 1.0
            
            for t in range(self.horizon):
                # Generate factor returns
                if t == 0 and stressed_factors:
                    # Apply stress on first day
                    factor_returns = np.array([
                        stressed_factors.get(factor, 0) 
                        for factor in barra_model.factor_covariance.columns
                    ])
                else:
                    # Normal factor returns (correlated)
                    z = np.random.standard_normal(n_factors)
                    factor_returns = daily_L @ z
                
                # Portfolio return = factor return + specific return
                factor_contribution = portfolio_betas @ factor_returns
                specific_return = np.random.normal(0, daily_specific_vol)
                
                daily_return = factor_contribution + specific_return
                path_value *= (1 + daily_return)
            
            final_values[sim] = path_value
        
        # Calculate metrics
        final_returns = final_values - 1
        results = self._calculate_var_cvar(final_returns, portfolio_value)
        
        # Add factor attribution
        results['risk_attribution'] = {
            'systematic': np.sqrt(portfolio_betas @ factor_cov @ portfolio_betas) / np.sqrt(int(os.getenv('TRADING_DAYS', 252))),
            'idiosyncratic': portfolio_specific_risk
        }
        
        if stressed_factors:
            results['stress_scenario'] = stressed_factors
        
        return results
    
    def _calculate_var_cvar(self, returns: np.ndarray, portfolio_value: float = None) -> Dict:
        """
        Calculate VaR and CVaR from return distribution.
        VaR: Value at Risk (percentile)
        CVaR: Conditional Value at Risk (expected loss beyond VaR)
        
        Note: VaR and CVaR are reported as positive numbers representing losses
        """
        results = {
            'mean_return': np.mean(returns),
            'volatility': np.std(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns)
        }
        
        # Calculate VaR and CVaR for each confidence level
        for conf in self.confidence_levels:
            # VaR is the loss at the (1-conf) percentile
            # For 95% confidence, we look at the 5th percentile
            var_percentile = (1 - conf) * 100
            var_threshold = np.percentile(returns, var_percentile)
            
            # VaR is the absolute loss value
            # If threshold is negative (a loss), convert to positive
            var_pct = -var_threshold if var_threshold < 0 else 0
            
            # Convert VaR to dollars if portfolio value provided
            if portfolio_value:
                var = var_pct * portfolio_value  # Dollar amount
            else:
                var = var_pct  # Keep as percentage if no portfolio value
            
            # CVaR is the expected percentage loss in the tail
            tail_returns = returns[returns <= var_threshold]
            if len(tail_returns) > 0:
                # Average of the tail returns (as percentage)
                cvar_pct = -np.mean(tail_returns) if np.mean(tail_returns) < 0 else 0
            else:
                # If no returns in tail, CVaR equals VaR percentage
                cvar_pct = var_pct
            
            results[f'VaR_{conf}'] = var  # Dollar amount or percentage
            results[f'CVaR_{conf}'] = cvar_pct  # Always percentage
            
            # Add number of VaR breaches for validation
            n_breaches = np.sum(returns <= var_threshold)
            results[f'VaR_breaches_{conf}'] = n_breaches
            results[f'VaR_breach_rate_{conf}'] = n_breaches / len(returns)
            
            # Verify CVaR >= VaR (mathematical requirement)
            # Note: We're comparing percentages for CVaR vs dollars/percentage for VaR
            # So we need to compare apples to apples
            if portfolio_value and (cvar_pct < var_pct):
                logger.warning(f"CVaR ({cvar_pct:.4f}) < VaR ({var_pct:.4f}) - mathematical inconsistency")
                # Don't arbitrarily adjust - this indicates a calculation error
        
        return results
    
    def calculate_marginal_var(self,
                             returns: pd.DataFrame,
                             weights: np.ndarray,
                             confidence: float = 0.95,
                             delta: float = 0.01) -> pd.Series:
        """
        Calculate marginal VaR contribution of each asset.
        Useful for risk budgeting.
        """
        base_var = self.parametric_var(returns, weights)['VaR_' + str(confidence)]
        marginal_vars = pd.Series(index=returns.columns)
        
        for i, asset in enumerate(returns.columns):
            # Increase weight slightly
            weights_up = weights.copy()
            weights_up[i] += delta
            weights_up /= weights_up.sum()  # Renormalize
            
            # Calculate new VaR
            new_var = self.parametric_var(returns, weights_up)['VaR_' + str(confidence)]
            
            # Marginal VaR
            marginal_vars[asset] = (new_var - base_var) / delta
        
        return marginal_vars
    
    def plot_monte_carlo_paths(self, 
                               returns: pd.DataFrame,
                               weights: np.ndarray,
                               n_paths: int = 1000,
                               n_days: int = None,
                               initial_value: float = 100,
                               figsize: Tuple[int, int] = (10, 7),
                               output_path: Optional[str] = None) -> plt.Figure:
        """
        Generate and plot Monte Carlo simulation paths for portfolio value.
        Clean visualization showing paths with percentiles.
        """
        if n_days is None:
            n_days = int(os.getenv('TRADING_DAYS', 252))
        
        # Calculate portfolio parameters
        portfolio_returns = returns @ weights
        daily_mean = portfolio_returns.mean()
        daily_vol = portfolio_returns.std()
        
        # Generate random paths
        dt = 1  # Daily time step
        paths = np.zeros((n_days + 1, n_paths))
        paths[0] = initial_value
        
        # Generate all random shocks at once for efficiency
        shocks = np.random.normal(daily_mean, daily_vol, size=(n_days, n_paths))
        
        # Calculate paths
        for t in range(1, n_days + 1):
            paths[t] = paths[t-1] * (1 + shocks[t-1])
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Create colormap for paths
        colors = plt.cm.rainbow(np.linspace(0, 1, min(100, n_paths)))
        
        # Plot a sample of paths with different colors
        n_display = min(100, n_paths)  # Show up to 100 paths
        indices = np.random.choice(n_paths, n_display, replace=False)
        for i, idx in enumerate(indices):
            ax.plot(paths[:, idx], color=colors[i], alpha=0.6, linewidth=0.8)
        
        # Calculate and plot statistics
        mean_path = np.mean(paths, axis=1)
        percentiles = np.percentile(paths, [5, 25, 75, 95], axis=1)
        
        # Plot mean and percentiles with enhanced visibility
        ax.plot(mean_path, color='black', linewidth=3, label='Expected Path', zorder=10)
        ax.plot(percentiles[0], color='darkred', linestyle='--', linewidth=2, 
                label='5th/95th Percentile', zorder=9)
        ax.plot(percentiles[3], color='darkred', linestyle='--', linewidth=2, zorder=9)
        
        # Fill between percentiles for better visualization
        ax.fill_between(range(n_days + 1), percentiles[0], percentiles[3], 
                       alpha=0.1, color='red', label='90% Confidence Interval')
        ax.fill_between(range(n_days + 1), percentiles[1], percentiles[2], 
                       alpha=0.2, color='blue', label='50% Confidence Interval')
        
        # Formatting
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Portfolio Value', fontsize=12)
        ax.set_title('Monte Carlo Simulation - 1 Year Portfolio Paths', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Set y-axis to start from a reasonable minimum
        ax.set_ylim(bottom=max(0, np.min(percentiles[0]) * 0.9))
        
        # Add summary statistics
        final_mean = mean_path[-1]
        final_std = np.std(paths[-1])
        ax.text(0.02, 0.98, 
                f'Final Value: ${final_mean:.2f} ± ${final_std:.2f}\\n'
                f'Paths: {n_paths:,}',
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Monte Carlo paths visualization to {output_path}")
        
        return fig