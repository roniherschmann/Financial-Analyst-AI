# clean_visualizations.py
"""
Clean visualizations for Monte Carlo paths and Efficient Frontier
"""
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def generate_monte_carlo_paths(returns, weights, n_paths=1000, n_days=252, initial_value=100):
    """Generate Monte Carlo simulation paths for portfolio value."""
    
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
    
    return paths




def create_monte_carlo_plot(risk_system, results, output_path='monte_carlo_paths.png'):
    """Create clean visualization of Monte Carlo paths only."""
    
    # Extract data
    returns = results['returns_data']['returns']
    weights = np.array(list(results['weights'].values()))
    
    # Create single figure
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    
    # ============ Monte Carlo Paths ============
    print("Generating Monte Carlo paths...")
    paths = generate_monte_carlo_paths(returns, weights, n_paths=1000, n_days=252)
    
    # Create colormap for paths
    colors = plt.cm.rainbow(np.linspace(0, 1, 100))
    
    # Plot a sample of paths with different colors
    n_display = 100  # Show 100 paths
    for i in range(n_display):
        ax1.plot(paths[:, i], color=colors[i], alpha=0.6, linewidth=0.8)
    
    # Calculate and plot statistics
    mean_path = np.mean(paths, axis=1)
    percentiles = np.percentile(paths, [5, 25, 75, 95], axis=1)
    
    # Plot mean and percentiles
    ax1.plot(mean_path, color='black', linewidth=3, label='Expected Path', zorder=10)
    ax1.plot(percentiles[0], color='darkred', linestyle='--', linewidth=2, label='5th/95th Percentile')
    ax1.plot(percentiles[3], color='darkred', linestyle='--', linewidth=2)
    
    ax1.set_xlabel('Trading Days', fontsize=12)
    ax1.set_ylabel('Portfolio Value', fontsize=12)
    ax1.set_title('Monte Carlo Simulation - 1 Year Portfolio Paths', fontsize=14, weight='bold')
    ax1.grid(False)  # No grid
    ax1.legend(loc='upper left', frameon=False)
    
    # Set y-axis to start from a reasonable minimum
    ax1.set_ylim(bottom=50)  # Start from 50 instead of 0 for better visualization
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    from main import RiskManagementSystem
    
    # Initialize system
    print("Initializing risk management system...")
    risk_system = RiskManagementSystem(portfolio_value=4_000_000)
    risk_system.initialize()
    
    # Example portfolio
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    weights = [0.25, 0.20, 0.20, 0.20, 0.15]
    
    # Run analysis
    print("Running portfolio analysis...")
    results = risk_system.analyze_portfolio(tickers, weights)
    
    # Create visualization
    print("Creating Monte Carlo visualization...")
    create_monte_carlo_plot(risk_system, results, 'monte_carlo_paths.png')
    
    print("Complete!")