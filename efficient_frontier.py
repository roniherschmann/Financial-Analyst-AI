# efficient_frontier.py
"""
Production-ready portfolio optimization with numerical stability.
Simplified but robust implementations.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import cvxpy as cp
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, 
                 returns: pd.DataFrame,
                 min_weight: float = 0.0,
                 max_weight: float = 0.25):  # 25% max for $4M portfolio
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        
        # Annualized parameters
        self.expected_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        
        # Add regularization for numerical stability
        self.cov_matrix += np.eye(self.n_assets) * 1e-8
        
        # Constraints
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Validate inputs
        if not self._is_positive_definite(self.cov_matrix):
            logger.warning("Covariance matrix not positive definite, adding regularization")
            self.cov_matrix += np.eye(self.n_assets) * 1e-6
    
    def _is_positive_definite(self, matrix: np.ndarray) -> bool:
        """Check if matrix is positive definite."""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def optimize_portfolio(self, 
                          objective: str = 'sharpe',
                          target_return: Optional[float] = None,
                          risk_free_rate: float = 0.02) -> Dict:
        """
        Main optimization interface supporting multiple objectives.
        """
        if objective == 'min_variance':
            return self._minimize_variance()
        elif objective == 'sharpe':
            return self._maximize_sharpe(risk_free_rate)
        elif objective == 'target_return':
            if target_return is None:
                raise ValueError("target_return required for this objective")
            return self._target_return_optimization(target_return)
        elif objective == 'risk_parity':
            return self._risk_parity()
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def _minimize_variance(self) -> Dict:
        """
        Minimum variance portfolio using quadratic programming.
        """
        w = cp.Variable(self.n_assets)
        
        # Objective: minimize portfolio variance
        portfolio_variance = cp.quad_form(w, self.cov_matrix.values)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w >= self.min_weight,  # Long only or minimum position
            w <= self.max_weight   # Concentration limit
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status != 'optimal':
            logger.error(f"Optimization failed: {problem.status}")
            # Return equal weight as fallback
            weights = np.ones(self.n_assets) / self.n_assets
        else:
            weights = w.value
        
        return self._create_portfolio_stats(weights)
    
    def _maximize_sharpe(self, risk_free_rate: float = 0.02) -> Dict:
        """
        Maximum Sharpe ratio portfolio.
        Note: This is a non-convex problem, solved via sequential convex optimization.
        """
        # Initial guess: equal weights
        w0 = np.ones(self.n_assets) / self.n_assets
        
        # Objective function (negative Sharpe for minimization)
        def negative_sharpe(w):
            returns = w @ self.expected_returns.values
            volatility = np.sqrt(w @ self.cov_matrix.values @ w)
            sharpe = (returns - risk_free_rate) / volatility
            return -sharpe
        
        # Gradient
        def sharpe_gradient(w):
            mu = self.expected_returns.values
            Sigma = self.cov_matrix.values
            
            portfolio_return = w @ mu
            portfolio_var = w @ Sigma @ w
            portfolio_vol = np.sqrt(portfolio_var)
            
            # Gradient of Sharpe ratio
            excess_return = portfolio_return - risk_free_rate
            
            dmu_dw = mu
            dvol_dw = (Sigma @ w) / portfolio_vol
            
            gradient = (dmu_dw * portfolio_vol - excess_return * dvol_dw) / (portfolio_var)
            return -gradient  # Negative because we're minimizing
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
        ]
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(self.n_assets)]
        
        # Optimize
        result = minimize(
            negative_sharpe,
            w0,
            method='SLSQP',
            jac=sharpe_gradient,
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning("Sharpe optimization failed, using minimum variance")
            return self._minimize_variance()
        
        weights = result.x
        return self._create_portfolio_stats(weights)
    
    def _target_return_optimization(self, target_return: float) -> Dict:
        """
        Minimize risk for a target return level.
        """
        w = cp.Variable(self.n_assets)
        
        # Objective: minimize variance
        portfolio_variance = cp.quad_form(w, self.cov_matrix.values)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w @ self.expected_returns.values >= target_return,
            w >= self.min_weight,
            w <= self.max_weight
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status != 'optimal':
            logger.error(f"Target return {target_return:.2%} not achievable")
            return None
        
        weights = w.value
        return self._create_portfolio_stats(weights)
    
    def _risk_parity(self) -> Dict:
        """
        Risk parity portfolio - equal risk contribution from each asset.
        """
        # Initial guess
        w0 = np.ones(self.n_assets) / self.n_assets
        
        def risk_parity_objective(w):
            # Calculate risk contributions
            Sigma_w = self.cov_matrix.values @ w
            portfolio_vol = np.sqrt(w @ Sigma_w)
            
            # Marginal contributions to risk
            marginal_contrib = Sigma_w / portfolio_vol
            
            # Risk contributions
            contrib = w * marginal_contrib
            
            # Target: equal contribution
            target_contrib = portfolio_vol / self.n_assets
            
            # Minimize squared deviations
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds - risk parity typically requires all positive weights
        bounds = [(0.001, self.max_weight) for _ in range(self.n_assets)]
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )
        
        if not result.success:
            logger.warning("Risk parity optimization failed")
            weights = w0
        else:
            weights = result.x
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return self._create_portfolio_stats(weights)
    
    def efficient_frontier(self, n_portfolios: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier by varying target returns.
        """
        # Get return range
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        
        # Add some buffer
        target_returns = np.linspace(min_ret * 0.9, max_ret * 1.1, n_portfolios)
        
        portfolios = []
        
        for target in target_returns:
            try:
                result = self._target_return_optimization(target)
                if result:
                    portfolios.append({
                        'return': result['expected_return'],
                        'volatility': result['volatility'],
                        'sharpe': result['sharpe_ratio']
                    })
            except:
                continue
        
        return pd.DataFrame(portfolios)
    
    def black_litterman(self,
                       market_weights: np.ndarray,
                       views: Dict[str, float],
                       view_confidence: float = 0.25,
                       market_tau: float = 0.05) -> Dict:
        """
        Black-Litterman model for combining market equilibrium with views.
        Simplified implementation focusing on robustness.
        """
        # Prior (market equilibrium returns)
        # Reverse optimization: given weights, what returns justify them?
        market_risk_aversion = 2.5  # Typical value
        equilibrium_returns = market_risk_aversion * self.cov_matrix @ market_weights
        
        # Build views matrix P and vector Q
        n_views = len(views)
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros(n_views)
        
        view_idx = 0
        for assets, view_return in views.items():
            if isinstance(assets, str):
                # Absolute view on single asset
                asset_idx = self.assets.index(assets)
                P[view_idx, asset_idx] = 1
                Q[view_idx] = view_return
            else:
                # Relative view (asset1 vs asset2)
                asset1_idx = self.assets.index(assets[0])
                asset2_idx = self.assets.index(assets[1])
                P[view_idx, asset1_idx] = 1
                P[view_idx, asset2_idx] = -1
                Q[view_idx] = view_return
            view_idx += 1
        
        # View uncertainty matrix (diagonal)
        omega = np.diag(np.diag(P @ (market_tau * self.cov_matrix) @ P.T)) / view_confidence
        
        # Posterior estimates (Black-Litterman formula) with stability checks
        try:
            # Check condition number before inversion
            scaled_cov = market_tau * self.cov_matrix
            cond_num = np.linalg.cond(scaled_cov)
            if cond_num > 1e10:
                logger.warning(f"Ill-conditioned covariance matrix: condition number = {cond_num:.2e}")
            
            inv_cov = np.linalg.inv(scaled_cov)
            
            # Check omega matrix condition
            omega_cond = np.linalg.cond(omega)
            if omega_cond > 1e10:
                logger.warning(f"Ill-conditioned uncertainty matrix: condition number = {omega_cond:.2e}")
            
            # Posterior covariance with stability check
            posterior_precision = inv_cov + P.T @ np.linalg.inv(omega) @ P
            posterior_cond = np.linalg.cond(posterior_precision)
            if posterior_cond > 1e10:
                logger.warning(f"Ill-conditioned posterior precision: condition number = {posterior_cond:.2e}")
                
            posterior_cov = np.linalg.inv(posterior_precision)
            
            # Posterior expected returns
            posterior_returns = posterior_cov @ (inv_cov @ equilibrium_returns + P.T @ np.linalg.inv(omega) @ Q)
            
        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion failed in Black-Litterman: {e}")
            raise ValueError("Black-Litterman optimization failed due to singular matrix")
        
        # Store for optimization
        original_returns = self.expected_returns.copy()
        self.expected_returns = pd.Series(posterior_returns, index=self.assets)
        
        # Optimize with new returns
        result = self._maximize_sharpe()
        
        # Restore original returns
        self.expected_returns = original_returns
        
        result['bl_returns'] = posterior_returns
        result['equilibrium_returns'] = equilibrium_returns
        
        return result
    
    def _create_portfolio_stats(self, weights: np.ndarray) -> Dict:
        """
        Calculate comprehensive portfolio statistics.
        """
        # Ensure weights sum to 1 (numerical precision)
        weights = weights / weights.sum()
        
        # Expected return and volatility
        expected_return = weights @ self.expected_returns
        variance = weights @ self.cov_matrix @ weights
        volatility = np.sqrt(variance)
        
        # Sharpe ratio
        sharpe_ratio = (expected_return - 0.02) / volatility if volatility > 0 else 0
        
        # Diversification ratio
        weighted_avg_vol = np.sum(weights * np.sqrt(np.diag(self.cov_matrix)))
        diversification_ratio = weighted_avg_vol / volatility if volatility > 0 else 1
        
        # Effective number of assets (inverse HHI)
        hhi = np.sum(weights ** 2)
        effective_n = 1 / hhi if hhi > 0 else self.n_assets
        
        # Create result dictionary
        result = {
            'weights': pd.Series(weights, index=self.assets),
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': diversification_ratio,
            'effective_n_assets': effective_n,
            'max_weight': np.max(weights),
            'min_weight': np.min(weights[weights > 0.001])  # Exclude near-zero
        }
        
        # Risk contributions
        marginal_contrib = self.cov_matrix @ weights / volatility
        risk_contrib = weights * marginal_contrib
        result['risk_contributions'] = pd.Series(risk_contrib, index=self.assets)
        
        return result
    
    def backtest_allocation(self,
                           weights: np.ndarray,
                           out_sample_returns: pd.DataFrame) -> Dict:
        """
        Simple backtest of allocation on out-of-sample data.
        """
        # Portfolio returns
        portfolio_returns = out_sample_returns @ weights
        
        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'n_days': len(portfolio_returns)
        }