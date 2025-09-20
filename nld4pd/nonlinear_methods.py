"""
Non-linear dynamics analysis methods for gait data.

This module implements various non-linear dynamics methods commonly used
in gait analysis for Parkinson's disease research.
"""

import numpy as np
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Any, Tuple, Optional
import warnings


def lyapunov_exponent(data: np.ndarray, 
                     embed_dim: int = 3, 
                     tau: int = 1, 
                     min_tsep: int = 0, 
                     max_tsep: Optional[int] = None) -> float:
    """
    Calculate the largest Lyapunov exponent using the algorithm by Rosenstein et al.
    
    Parameters:
    -----------
    data : np.ndarray
        1D time series data
    embed_dim : int, default=3
        Embedding dimension
    tau : int, default=1
        Time delay for embedding
    min_tsep : int, default=0
        Minimum temporal separation for nearest neighbors
    max_tsep : int, optional
        Maximum temporal separation to consider
        
    Returns:
    --------
    float
        Largest Lyapunov exponent
    """
    try:
        # Embed the time series
        embedded = _embed_time_series(data, embed_dim, tau)
        N = len(embedded)
        
        if max_tsep is None:
            max_tsep = N // 4
            
        # Find nearest neighbors
        distances = squareform(pdist(embedded))
        
        # Exclude self-distances and apply temporal separation constraint
        for i in range(N):
            for j in range(max(0, i - min_tsep), min(N, i + min_tsep + 1)):
                distances[i, j] = np.inf
                
        nearest_neighbors = np.argmin(distances, axis=1)
        
        # Calculate divergence rates
        divergence_rates = []
        max_evolution_time = min(max_tsep, N - max(range(N)) - 1)
        
        for t in range(1, max_evolution_time):
            valid_pairs = []
            
            for i in range(N - t):
                j = nearest_neighbors[i]
                if j < N - t and j != i:
                    # Calculate initial separation
                    initial_sep = np.linalg.norm(embedded[i] - embedded[j])
                    
                    # Calculate separation after time t
                    evolved_sep = np.linalg.norm(embedded[i + t] - embedded[j + t])
                    
                    if initial_sep > 0 and evolved_sep > 0:
                        valid_pairs.append(np.log(evolved_sep / initial_sep))
                        
            if valid_pairs:
                divergence_rates.append(np.mean(valid_pairs))
            else:
                divergence_rates.append(np.nan)
                
        # Fit linear regression to get Lyapunov exponent
        valid_indices = ~np.isnan(divergence_rates)
        if np.sum(valid_indices) < 2:
            return np.nan
            
        time_steps = np.arange(1, len(divergence_rates) + 1)[valid_indices]
        valid_rates = np.array(divergence_rates)[valid_indices]
        
        coefficients = np.polyfit(time_steps, valid_rates, 1)
        lyap_exp = coefficients[0]  # Slope of the linear fit
        
        return lyap_exp
        
    except Exception as e:
        warnings.warn(f"Error calculating Lyapunov exponent: {str(e)}")
        return np.nan


def correlation_dimension(data: np.ndarray, 
                         embed_dim: int = 10, 
                         tau: int = 1,
                         r_min: Optional[float] = None,
                         r_max: Optional[float] = None,
                         n_points: int = 50) -> float:
    """
    Calculate the correlation dimension using the Grassberger-Procaccia algorithm.
    
    Parameters:
    -----------
    data : np.ndarray
        1D time series data
    embed_dim : int, default=10
        Embedding dimension
    tau : int, default=1
        Time delay for embedding
    r_min : float, optional
        Minimum radius for correlation sum calculation
    r_max : float, optional
        Maximum radius for correlation sum calculation
    n_points : int, default=50
        Number of radius points to evaluate
        
    Returns:
    --------
    float
        Correlation dimension estimate
    """
    try:
        # Embed the time series
        embedded = _embed_time_series(data, embed_dim, tau)
        N = len(embedded)
        
        # Calculate pairwise distances
        distances = pdist(embedded)
        
        if r_min is None:
            r_min = np.percentile(distances, 1)
        if r_max is None:
            r_max = np.percentile(distances, 50)
            
        # Define radius range
        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
        
        # Calculate correlation sums
        correlation_sums = []
        for r in radii:
            c_r = np.sum(distances <= r) / (N * (N - 1) / 2)
            correlation_sums.append(c_r)
            
        # Remove zeros for log calculation
        valid_indices = np.array(correlation_sums) > 0
        if np.sum(valid_indices) < 5:
            return np.nan
            
        log_radii = np.log(radii[valid_indices])
        log_correlation_sums = np.log(np.array(correlation_sums)[valid_indices])
        
        # Find linear region and calculate slope
        # Use middle portion of the curve for more stable estimation
        mid_start = len(log_radii) // 4
        mid_end = 3 * len(log_radii) // 4
        
        if mid_end - mid_start < 3:
            # Use all points if too few
            coefficients = np.polyfit(log_radii, log_correlation_sums, 1)
        else:
            coefficients = np.polyfit(log_radii[mid_start:mid_end], 
                                    log_correlation_sums[mid_start:mid_end], 1)
        
        correlation_dim = coefficients[0]  # Slope
        
        return correlation_dim
        
    except Exception as e:
        warnings.warn(f"Error calculating correlation dimension: {str(e)}")
        return np.nan


def fractal_dimension(data: np.ndarray, method: str = 'higuchi') -> float:
    """
    Calculate fractal dimension using various methods.
    
    Parameters:
    -----------
    data : np.ndarray
        1D time series data
    method : str, default='higuchi'
        Method to use: 'higuchi', 'box_counting', or 'detrended_fluctuation'
        
    Returns:
    --------
    float
        Fractal dimension estimate
    """
    if method == 'higuchi':
        return _higuchi_fractal_dimension(data)
    elif method == 'box_counting':
        return _box_counting_dimension(data)
    elif method == 'detrended_fluctuation':
        return _detrended_fluctuation_analysis(data)
    else:
        raise ValueError(f"Unknown method: {method}")


def _higuchi_fractal_dimension(data: np.ndarray, k_max: int = 10) -> float:
    """Calculate fractal dimension using Higuchi's method."""
    try:
        N = len(data)
        L = []
        x = []
        
        for k in range(1, k_max + 1):
            Lk = []
            for m in range(k):
                Lmk = 0
                for i in range(1, int((N - m) / k)):
                    Lmk += abs(data[m + i * k] - data[m + (i - 1) * k])
                Lmk = Lmk * (N - 1) / (((N - m) / k) * k) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk)))
            x.append(np.log(1.0 / k))
            
        # Linear regression
        coefficients = np.polyfit(x, L, 1)
        return coefficients[0]  # Slope is the fractal dimension
        
    except Exception as e:
        warnings.warn(f"Error calculating Higuchi fractal dimension: {str(e)}")
        return np.nan


def _box_counting_dimension(data: np.ndarray) -> float:
    """Calculate fractal dimension using box counting method."""
    try:
        # Normalize data to [0, 1]
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Create different box sizes
        scales = np.logspace(0.01, 0.2, num=20, base=2)
        Ns = []
        
        for scale in scales:
            # Quantize the data
            H = int(1 / scale)
            boxes = set()
            
            for i in range(len(data_norm) - 1):
                box_x = int(data_norm[i] * H)
                box_y = int(data_norm[i + 1] * H)
                boxes.add((box_x, box_y))
                
            Ns.append(len(boxes))
            
        # Linear regression in log-log space
        coefficients = np.polyfit(np.log(scales), np.log(Ns), 1)
        return -coefficients[0]  # Negative slope is the dimension
        
    except Exception as e:
        warnings.warn(f"Error calculating box counting dimension: {str(e)}")
        return np.nan


def _detrended_fluctuation_analysis(data: np.ndarray) -> float:
    """Calculate scaling exponent using detrended fluctuation analysis."""
    try:
        N = len(data)
        
        # Integrate the data
        y = np.cumsum(data - np.mean(data))
        
        # Define scales
        scales = np.unique(np.logspace(0.5, np.log10(N // 4), 20).astype(int))
        
        fluctuations = []
        
        for n in scales:
            # Divide into non-overlapping segments
            n_segments = N // n
            segments = y[:n_segments * n].reshape(n_segments, n)
            
            # Detrend each segment
            F_n = 0
            for segment in segments:
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                F_n += np.mean((segment - trend) ** 2)
                
            F_n = np.sqrt(F_n / n_segments)
            fluctuations.append(F_n)
            
        # Linear regression in log-log space
        log_scales = np.log(scales)
        log_fluctuations = np.log(fluctuations)
        
        coefficients = np.polyfit(log_scales, log_fluctuations, 1)
        return coefficients[0]  # Scaling exponent
        
    except Exception as e:
        warnings.warn(f"Error calculating DFA scaling exponent: {str(e)}")
        return np.nan


def recurrence_quantification_analysis(data: np.ndarray, 
                                     embed_dim: int = 3, 
                                     tau: int = 1,
                                     threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Perform Recurrence Quantification Analysis (RQA).
    
    Parameters:
    -----------
    data : np.ndarray
        1D time series data
    embed_dim : int, default=3
        Embedding dimension
    tau : int, default=1
        Time delay for embedding
    threshold : float, optional
        Recurrence threshold. If None, uses 10% of max distance
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing RQA measures
    """
    try:
        # Embed the time series
        embedded = _embed_time_series(data, embed_dim, tau)
        N = len(embedded)
        
        # Calculate distance matrix
        distances = squareform(pdist(embedded))
        
        # Set threshold
        if threshold is None:
            threshold = 0.1 * np.max(distances)
            
        # Create recurrence matrix
        recurrence_matrix = (distances <= threshold).astype(int)
        
        # Remove main diagonal
        np.fill_diagonal(recurrence_matrix, 0)
        
        # Calculate RQA measures
        rqa_measures = {}
        
        # Recurrence Rate (RR)
        rqa_measures['recurrence_rate'] = np.sum(recurrence_matrix) / (N * (N - 1))
        
        # Determinism (DET)
        line_lengths = _get_diagonal_line_lengths(recurrence_matrix, min_length=2)
        if line_lengths:
            rqa_measures['determinism'] = np.sum(line_lengths) / np.sum(recurrence_matrix)
            rqa_measures['average_diagonal_line'] = np.mean(line_lengths)
            rqa_measures['max_diagonal_line'] = np.max(line_lengths)
        else:
            rqa_measures['determinism'] = 0.0
            rqa_measures['average_diagonal_line'] = 0.0
            rqa_measures['max_diagonal_line'] = 0.0
            
        # Laminarity (LAM)
        vertical_lengths = _get_vertical_line_lengths(recurrence_matrix, min_length=2)
        if vertical_lengths:
            rqa_measures['laminarity'] = np.sum(vertical_lengths) / np.sum(recurrence_matrix)
            rqa_measures['average_vertical_line'] = np.mean(vertical_lengths)
        else:
            rqa_measures['laminarity'] = 0.0
            rqa_measures['average_vertical_line'] = 0.0
            
        # Entropy
        if line_lengths:
            p_l = np.bincount(line_lengths) / len(line_lengths)
            p_l = p_l[p_l > 0]  # Remove zeros
            rqa_measures['entropy'] = -np.sum(p_l * np.log(p_l))
        else:
            rqa_measures['entropy'] = 0.0
            
        return rqa_measures
        
    except Exception as e:
        warnings.warn(f"Error in RQA analysis: {str(e)}")
        return {'error': str(e)}


def _embed_time_series(data: np.ndarray, embed_dim: int, tau: int) -> np.ndarray:
    """Embed time series using method of delays."""
    N = len(data)
    embedded_length = N - (embed_dim - 1) * tau
    
    if embedded_length <= 0:
        raise ValueError("Time series too short for given embedding parameters")
        
    embedded = np.zeros((embedded_length, embed_dim))
    
    for i in range(embed_dim):
        embedded[:, i] = data[i * tau:i * tau + embedded_length]
        
    return embedded


def _get_diagonal_line_lengths(recurrence_matrix: np.ndarray, min_length: int = 2) -> list:
    """Get lengths of diagonal lines in recurrence matrix."""
    N = recurrence_matrix.shape[0]
    line_lengths = []
    
    # Check all diagonals
    for offset in range(-(N-1), N):
        diagonal = np.diagonal(recurrence_matrix, offset=offset)
        lengths = _get_line_lengths_in_sequence(diagonal, min_length)
        line_lengths.extend(lengths)
        
    return line_lengths


def _get_vertical_line_lengths(recurrence_matrix: np.ndarray, min_length: int = 2) -> list:
    """Get lengths of vertical lines in recurrence matrix."""
    line_lengths = []
    
    for col in range(recurrence_matrix.shape[1]):
        column = recurrence_matrix[:, col]
        lengths = _get_line_lengths_in_sequence(column, min_length)
        line_lengths.extend(lengths)
        
    return line_lengths


def _get_line_lengths_in_sequence(sequence: np.ndarray, min_length: int) -> list:
    """Get lengths of consecutive 1s in a binary sequence."""
    lengths = []
    current_length = 0
    
    for value in sequence:
        if value == 1:
            current_length += 1
        else:
            if current_length >= min_length:
                lengths.append(current_length)
            current_length = 0
            
    # Check final sequence
    if current_length >= min_length:
        lengths.append(current_length)
        
    return lengths