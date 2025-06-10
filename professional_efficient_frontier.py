# professional_efficient_frontier.py
"""
Professional Efficient Frontier Visualization
Clean, minimalist design with proper financial aesthetics
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.patches as mpatches

def calculate_portfolio_stats(weights, returns, cov_matrix):
    """Calculate portfolio return and volatility."""
    portfolio_return = np.dot(weights, returns)
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_vol

def generate_efficient_frontier(returns_data, n_points=100):
    """Generate the efficient frontier."""
    
    # Annualized statistics
    annual_returns = returns_data.mean() * 252
    annual_cov = returns_data.cov() * 252
    n_assets = len(annual_returns)
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Find minimum variance portfolio
    def portfolio_variance(w):
        return np.dot(w, np.dot(annual_cov, w))
    
    x0 = np.ones(n_assets) / n_assets
    min_var_result = minimize(portfolio_variance, x0, method='SLSQP', 
                             bounds=bounds, constraints=constraints)
    min_var_weights = min_var_result.x
    min_var_return, min_var_vol = calculate_portfolio_stats(min_var_weights, annual_returns, annual_cov)
    
    # Find maximum return portfolio
    max_return = annual_returns.max()
    
    # Generate efficient frontier
    target_returns = np.linspace(min_var_return, max_return * 0.95, n_points)
    efficient_vols = []
    efficient_weights = []
    
    for target in target_returns:
        constraints_with_return = constraints + [
            {'type': 'eq', 'fun': lambda w, r=target: np.dot(w, annual_returns) - r}
        ]
        
        result = minimize(portfolio_variance, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints_with_return,
                         options={'disp': False, 'maxiter': 1000})
        
        if result.success:
            vol = np.sqrt(portfolio_variance(result.x))
            efficient_vols.append(vol)
            efficient_weights.append(result.x)
    
    return (np.array(efficient_vols), target_returns, 
            np.array(efficient_weights), min_var_weights)

def generate_monte_carlo_portfolios(returns_data, n_portfolios=5000):
    """Generate random portfolios for the scatter plot."""
    
    annual_returns = returns_data.mean() * 252
    annual_cov = returns_data.cov() * 252
    n_assets = len(annual_returns)
    
    weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
    
    portfolio_returns = np.dot(weights, annual_returns)
    portfolio_vols = np.sqrt(np.array([np.dot(w, np.dot(annual_cov, w)) for w in weights]))
    portfolio_sharpes = (portfolio_returns - 0.02) / portfolio_vols
    
    return portfolio_vols, portfolio_returns, portfolio_sharpes, weights

def create_professional_efficient_frontier(returns_data, current_weights, output_path='professional_frontier.png'):
    """Create a professional, clean efficient frontier visualization."""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Generate data
    print("Generating random portfolios...")
    mc_vols, mc_returns, mc_sharpes, mc_weights = generate_monte_carlo_portfolios(returns_data, 10000)
    
    print("Calculating efficient frontier...")
    ef_vols, ef_returns, ef_weights, min_var_weights = generate_efficient_frontier(returns_data)
    
    # Color scheme
    # Use Sharpe ratio for coloring
    scatter = ax.scatter(mc_vols * 100, mc_returns * 100, 
                        c=mc_sharpes, cmap='coolwarm', 
                        s=8, alpha=0.5, edgecolors='none',
                        vmin=0, vmax=2)
    
    # Plot efficient frontier
    ax.plot(ef_vols * 100, ef_returns * 100, 
            color='navy', linewidth=3.5, 
            label='Efficient Frontier', zorder=5)
    
    # Calculate current portfolio stats
    annual_returns = returns_data.mean() * 252
    annual_cov = returns_data.cov() * 252
    current_return, current_vol = calculate_portfolio_stats(current_weights, annual_returns, annual_cov)
    current_sharpe = (current_return - 0.02) / current_vol
    
    # Mark special portfolios
    # Current portfolio
    ax.scatter(current_vol * 100, current_return * 100, 
              color='red', s=200, marker='D', 
              edgecolors='darkred', linewidth=2,
              label=f'Current Portfolio (SR: {current_sharpe:.2f})', 
              zorder=10)
    
    # Minimum variance
    min_var_return, min_var_vol = calculate_portfolio_stats(min_var_weights, annual_returns, annual_cov)
    ax.scatter(min_var_vol * 100, min_var_return * 100, 
              color='green', s=200, marker='s',
              edgecolors='darkgreen', linewidth=2,
              label='Minimum Variance', zorder=10)
    
    # Maximum Sharpe ratio
    sharpe_ratios = (ef_returns - 0.02) / ef_vols
    max_sharpe_idx = np.argmax(sharpe_ratios)
    ax.scatter(ef_vols[max_sharpe_idx] * 100, ef_returns[max_sharpe_idx] * 100,
              color='gold', s=200, marker='^',
              edgecolors='darkgoldenrod', linewidth=2,
              label=f'Maximum Sharpe (SR: {sharpe_ratios[max_sharpe_idx]:.2f})', 
              zorder=10)
    
    # Add Capital Allocation Line
    rf_rate = 2  # 2% risk-free rate
    max_x = ef_vols[max_sharpe_idx] * 100 * 1.5
    cal_y = rf_rate + sharpe_ratios[max_sharpe_idx] * np.array([0, max_x])
    ax.plot([0, max_x], cal_y, 'k--', alpha=0.5, linewidth=1.5, 
            label='Capital Allocation Line')
    
    # Formatting
    ax.set_xlabel('Volatility (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Expected Return (%)', fontsize=14, fontweight='bold')
    ax.set_title('Efficient Frontier Analysis', fontsize=18, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(0, max(mc_vols) * 100 * 1.1)
    ax.set_ylim(min(mc_returns) * 100 * 0.9, max(mc_returns) * 100 * 1.1)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, 
                      shadow=True, fontsize=11)
    legend.get_frame().set_alpha(0.9)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Sharpe Ratio', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add text annotations
    ax.text(0.95, 0.05, 'Size = 10,000 portfolios', 
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved professional efficient frontier to {output_path}")
    
    plt.show()
    
    return fig

def cleanup_directory():
    """Remove unnecessary files from the directory."""
    import os
    
    files_to_remove = [
        'enhanced_visualizations.py',
        'fixed_barra_analysis.py',
        'proper_factor_analysis.png',
        'risk_report_20250609_214407.txt',
        'portfolio_risk_report.png'  # if exists
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed: {file}")
    
    print("\nDirectory cleaned!")

if __name__ == "__main__":
    from main import RiskManagementSystem
    
    # Initialize system
    print("Initializing system...")
    risk_system = RiskManagementSystem(portfolio_value=4_000_000)
    risk_system.initialize()
    
    # Portfolio
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
    
    # Get returns data
    print("Fetching data...")
    prices = risk_system.data_fetcher.fetch_prices(tickers, '2022-06-10', '2025-06-09')
    returns = risk_system.data_fetcher.calculate_returns(prices)
    
    # Create visualization
    print("\nCreating professional efficient frontier...")
    create_professional_efficient_frontier(returns, weights, 'professional_frontier.png')
    
    # Cleanup
    print("\nCleaning up directory...")
    cleanup_directory()