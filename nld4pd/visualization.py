"""
Visualization functions for non-linear dynamics analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings

# Set style
plt.style.use('default')
sns.set_palette("husl")


def plot_results(results: Dict[str, Any], 
                save_path: Optional[str] = None,
                figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot comprehensive analysis results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results dictionary from GaitAnalyzer
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(15, 10)
        Figure size in inches
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Non-Linear Dynamics Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Lyapunov Exponent
    ax1 = axes[0, 0]
    if ('lyapunov_exponent' in results and 
        not (isinstance(results['lyapunov_exponent'], dict) and 'error' in results['lyapunov_exponent']) and
        not np.isnan(results['lyapunov_exponent'])):
        lyap_exp = results['lyapunov_exponent']
        ax1.bar(['Lyapunov Exponent'], [lyap_exp], color='skyblue', alpha=0.7)
        ax1.set_ylabel('Value')
        ax1.set_title('Largest Lyapunov Exponent')
        ax1.grid(True, alpha=0.3)
        
        # Add interpretation
        if lyap_exp > 0:
            ax1.text(0, lyap_exp/2, 'Chaotic\n(λ > 0)', ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        elif lyap_exp < 0:
            ax1.text(0, lyap_exp/2, 'Stable\n(λ < 0)', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    else:
        ax1.text(0.5, 0.5, 'Lyapunov Exponent\nNot Available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Largest Lyapunov Exponent')
    
    # Plot 2: Fractal Dimension
    ax2 = axes[0, 1]
    if ('fractal_dimension' in results and 
        not (isinstance(results['fractal_dimension'], dict) and 'error' in results['fractal_dimension']) and
        not np.isnan(results['fractal_dimension'])):
        frac_dim = results['fractal_dimension']
        ax2.bar(['Fractal Dimension'], [frac_dim], color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Dimension')
        ax2.set_title('Higuchi Fractal Dimension')
        ax2.grid(True, alpha=0.3)
        
        # Add reference lines
        ax2.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5, label='Regular (1.5)')
        ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Random Walk (2.0)')
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'Fractal Dimension\nNot Available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Higuchi Fractal Dimension')
    
    # Plot 3: Correlation Dimension
    ax3 = axes[1, 0]
    if ('correlation_dimension' in results and 
        not (isinstance(results['correlation_dimension'], dict) and 'error' in results['correlation_dimension']) and
        not np.isnan(results['correlation_dimension'])):
        corr_dim = results['correlation_dimension']
        ax3.bar(['Correlation Dimension'], [corr_dim], color='lightgreen', alpha=0.7)
        ax3.set_ylabel('Dimension')
        ax3.set_title('Correlation Dimension')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Correlation Dimension\nNot Available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Correlation Dimension')
    
    # Plot 4: RQA Results
    ax4 = axes[1, 1]
    if 'rqa' in results and 'error' not in results['rqa']:
        rqa_data = results['rqa']
        measures = ['recurrence_rate', 'determinism', 'laminarity']
        values = [rqa_data.get(measure, 0) for measure in measures]
        labels = ['RR', 'DET', 'LAM']
        
        bars = ax4.bar(labels, values, color=['purple', 'orange', 'brown'], alpha=0.7)
        ax4.set_ylabel('Value')
        ax4.set_title('Recurrence Quantification Analysis')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'RQA Analysis\nFailed', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Recurrence Quantification Analysis')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results plot saved to: {save_path}")
    else:
        plt.show()


def plot_time_series(data: np.ndarray,
                    sampling_rate: float = 100.0,
                    title: str = "Gait Time Series",
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Plot gait time series data.
    
    Parameters:
    -----------
    data : np.ndarray
        1D time series data
    sampling_rate : float, default=100.0
        Sampling rate in Hz
    title : str, default="Gait Time Series"
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(12, 4)
        Figure size in inches
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    time = np.arange(len(data)) / sampling_rate
    ax.plot(time, data, 'b-', linewidth=0.8, alpha=0.8)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Mean: {np.mean(data):.3f}\nStd: {np.std(data):.3f}\nLength: {len(data)} samples'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to: {save_path}")
    else:
        plt.show()


def plot_phase_space(data: np.ndarray,
                    embed_dim: int = 3,
                    tau: int = 1,
                    title: str = "Phase Space Reconstruction",
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot phase space reconstruction of the time series.
    
    Parameters:
    -----------
    data : np.ndarray
        1D time series data
    embed_dim : int, default=3
        Embedding dimension
    tau : int, default=1
        Time delay
    title : str, default="Phase Space Reconstruction"
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(10, 8)
        Figure size in inches
    """
    from .utils import create_phase_space
    
    try:
        embedded = create_phase_space(data, embed_dim, tau)
    except ValueError as e:
        print(f"Error creating phase space: {e}")
        return
    
    if embed_dim == 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(embedded[:, 0], embedded[:, 1], 'b-', alpha=0.6, linewidth=0.5)
        ax.scatter(embedded[0, 0], embedded[0, 1], c='green', s=50, label='Start', zorder=5)
        ax.scatter(embedded[-1, 0], embedded[-1, 1], c='red', s=50, label='End', zorder=5)
        ax.set_xlabel(f'x(t)')
        ax.set_ylabel(f'x(t+{tau})')
        ax.legend()
        
    elif embed_dim == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2], 'b-', alpha=0.6, linewidth=0.5)
        ax.scatter(embedded[0, 0], embedded[0, 1], embedded[0, 2], c='green', s=50, label='Start')
        ax.scatter(embedded[-1, 0], embedded[-1, 1], embedded[-1, 2], c='red', s=50, label='End')
        ax.set_xlabel(f'x(t)')
        ax.set_ylabel(f'x(t+{tau})')
        ax.set_zlabel(f'x(t+{2*tau})')
        ax.legend()
        
    else:
        # For higher dimensions, plot 2D projections
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        projections = [(0, 1), (0, 2), (1, 2), (0, 3)] if embed_dim > 3 else [(0, 1), (0, 2), (1, 2), (0, 1)]
        
        for idx, (i, j) in enumerate(projections):
            ax = axes[idx // 2, idx % 2]
            if j < embed_dim:
                ax.plot(embedded[:, i], embedded[:, j], 'b-', alpha=0.6, linewidth=0.5)
                ax.scatter(embedded[0, i], embedded[0, j], c='green', s=30, zorder=5)
                ax.scatter(embedded[-1, i], embedded[-1, j], c='red', s=30, zorder=5)
                ax.set_xlabel(f'x(t+{i*tau})')
                ax.set_ylabel(f'x(t+{j*tau})')
            else:
                ax.axis('off')
    
    plt.suptitle(f'{title} (dim={embed_dim}, τ={tau})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phase space plot saved to: {save_path}")
    else:
        plt.show()


def plot_recurrence_matrix(data: np.ndarray,
                          embed_dim: int = 3,
                          tau: int = 1,
                          threshold: Optional[float] = None,
                          title: str = "Recurrence Plot",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (8, 8)) -> None:
    """
    Plot recurrence matrix.
    
    Parameters:
    -----------
    data : np.ndarray
        1D time series data
    embed_dim : int, default=3
        Embedding dimension
    tau : int, default=1
        Time delay
    threshold : float, optional
        Recurrence threshold
    title : str, default="Recurrence Plot"
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(8, 8)
        Figure size in inches
    """
    from .utils import create_phase_space
    from scipy.spatial.distance import pdist, squareform
    
    try:
        embedded = create_phase_space(data, embed_dim, tau)
        distances = squareform(pdist(embedded))
        
        if threshold is None:
            threshold = 0.1 * np.max(distances)
            
        recurrence_matrix = (distances <= threshold).astype(int)
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(recurrence_matrix, cmap='binary', origin='lower')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Time Index')
        ax.set_title(f'{title}\n(threshold={threshold:.3f})')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Recurrence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Recurrence plot saved to: {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error creating recurrence plot: {e}")


def plot_comparison(pd_results: List[Dict[str, Any]],
                   control_results: List[Dict[str, Any]],
                   measures: Optional[List[str]] = None,
                   title: str = "PD vs Control Comparison",
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot comparison between PD patients and controls.
    
    Parameters:
    -----------
    pd_results : List[Dict[str, Any]]
        Results from PD patients
    control_results : List[Dict[str, Any]]
        Results from controls
    measures : List[str], optional
        Measures to compare
    title : str, default="PD vs Control Comparison"
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(12, 8)
        Figure size in inches
    """
    if measures is None:
        measures = ['lyapunov_exponent', 'fractal_dimension', 'correlation_dimension']
    
    # Extract values for each measure
    comparison_data = {}
    
    for measure in measures:
        pd_values = []
        control_values = []
        
        for result in pd_results:
            if measure in result and 'error' not in result[measure]:
                if isinstance(result[measure], dict):
                    # For RQA, take recurrence rate as example
                    if 'recurrence_rate' in result[measure]:
                        pd_values.append(result[measure]['recurrence_rate'])
                else:
                    pd_values.append(result[measure])
        
        for result in control_results:
            if measure in result and 'error' not in result[measure]:
                if isinstance(result[measure], dict):
                    if 'recurrence_rate' in result[measure]:
                        control_values.append(result[measure]['recurrence_rate'])
                else:
                    control_values.append(result[measure])
        
        if pd_values and control_values:
            comparison_data[measure] = {
                'PD': pd_values,
                'Control': control_values
            }
    
    if not comparison_data:
        print("No valid data for comparison")
        return
    
    # Create subplot for each measure
    n_measures = len(comparison_data)
    cols = min(3, n_measures)
    rows = (n_measures + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_measures == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for idx, (measure, data) in enumerate(comparison_data.items()):
        ax = axes[idx] if n_measures > 1 else axes[0]
        
        # Create box plot
        box_data = [data['PD'], data['Control']]
        labels = ['PD', 'Control']
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')
        
        # Add individual points
        for i, values in enumerate(box_data):
            y = values
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.6, s=20)
        
        ax.set_title(measure.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Add statistical annotation
        from scipy import stats
        if len(data['PD']) > 1 and len(data['Control']) > 1:
            stat, p_value = stats.ttest_ind(data['PD'], data['Control'])
            ax.text(0.5, 0.95, f'p = {p_value:.3f}', transform=ax.transAxes,
                   ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="yellow" if p_value < 0.05 else "white", alpha=0.7))
    
    # Hide unused subplots
    for idx in range(n_measures, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()


def plot_lyapunov_evolution(data: np.ndarray,
                           embed_dim: int = 3,
                           tau: int = 1,
                           max_time: int = 50,
                           title: str = "Lyapunov Exponent Evolution",
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot the evolution of divergence for Lyapunov exponent calculation.
    
    Parameters:
    -----------
    data : np.ndarray
        1D time series data
    embed_dim : int, default=3
        Embedding dimension
    tau : int, default=1
        Time delay
    max_time : int, default=50
        Maximum evolution time to plot
    title : str, default="Lyapunov Exponent Evolution"
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(10, 6)
        Figure size in inches
    """
    from .nonlinear_methods import _embed_time_series
    from scipy.spatial.distance import pdist, squareform
    
    try:
        # Embed the time series
        embedded = _embed_time_series(data, embed_dim, tau)
        N = len(embedded)
        
        # Find nearest neighbors
        distances = squareform(pdist(embedded))
        np.fill_diagonal(distances, np.inf)
        nearest_neighbors = np.argmin(distances, axis=1)
        
        # Calculate divergence evolution
        max_time = min(max_time, N // 4)
        mean_log_divergence = []
        
        for t in range(1, max_time):
            log_divergences = []
            
            for i in range(N - t):
                j = nearest_neighbors[i]
                if j < N - t:
                    initial_sep = np.linalg.norm(embedded[i] - embedded[j])
                    evolved_sep = np.linalg.norm(embedded[i + t] - embedded[j + t])
                    
                    if initial_sep > 0 and evolved_sep > 0:
                        log_divergences.append(np.log(evolved_sep / initial_sep))
            
            if log_divergences:
                mean_log_divergence.append(np.mean(log_divergences))
            else:
                mean_log_divergence.append(np.nan)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        time_steps = np.arange(1, len(mean_log_divergence) + 1)
        
        ax.plot(time_steps, mean_log_divergence, 'b-', marker='o', markersize=3, alpha=0.7)
        
        # Fit linear region for Lyapunov exponent
        valid_indices = ~np.isnan(mean_log_divergence)
        if np.sum(valid_indices) >= 2:
            coeffs = np.polyfit(time_steps[valid_indices], 
                              np.array(mean_log_divergence)[valid_indices], 1)
            fit_line = np.polyval(coeffs, time_steps)
            ax.plot(time_steps, fit_line, 'r--', alpha=0.8, 
                   label=f'Linear fit: λ = {coeffs[0]:.4f}')
            ax.legend()
        
        ax.set_xlabel('Evolution Time')
        ax.set_ylabel('ln(divergence)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Lyapunov evolution plot saved to: {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error creating Lyapunov evolution plot: {e}")