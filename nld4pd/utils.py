"""
Utility functions for data loading and preprocessing.
"""

import numpy as np
from scipy import signal
from typing import Optional, Union, Tuple
import warnings


def load_gait_data(filepath: str, 
                  column: Optional[int] = None,
                  skip_header: int = 0,
                  delimiter: str = ',') -> np.ndarray:
    """
    Load gait data from a text file.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    column : int, optional
        Column index to load (0-based). If None, loads all columns
    skip_header : int, default=0
        Number of header lines to skip
    delimiter : str, default=','
        Delimiter used in the file
        
    Returns:
    --------
    np.ndarray
        Loaded gait data
    """
    try:
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skip_header)
        
        if data.ndim == 1:
            return data
        elif data.ndim == 2:
            if column is not None:
                return data[:, column]
            else:
                return data
        else:
            raise ValueError("Data must be 1D or 2D array")
            
    except Exception as e:
        raise IOError(f"Error loading data from {filepath}: {str(e)}")


def preprocess_data(data: np.ndarray,
                   detrend: bool = True,
                   normalize: bool = True,
                   filter_data: bool = True,
                   lowpass_freq: Optional[float] = None,
                   sampling_rate: float = 100.0) -> np.ndarray:
    """
    Preprocess gait data for non-linear analysis.
    
    Parameters:
    -----------
    data : np.ndarray
        Input gait time series data
    detrend : bool, default=True
        Whether to detrend the data
    normalize : bool, default=True
        Whether to normalize the data to zero mean and unit variance
    filter_data : bool, default=True
        Whether to apply low-pass filtering
    lowpass_freq : float, optional
        Low-pass filter cutoff frequency in Hz. If None, uses sampling_rate/4
    sampling_rate : float, default=100.0
        Sampling rate in Hz
        
    Returns:
    --------
    np.ndarray
        Preprocessed data
    """
    processed_data = data.copy()
    
    # Remove NaN values
    if np.any(np.isnan(processed_data)):
        warnings.warn("NaN values detected. Interpolating missing values.")
        processed_data = _interpolate_nan(processed_data)
    
    # Detrend the data
    if detrend:
        processed_data = signal.detrend(processed_data, type='linear')
    
    # Apply low-pass filter
    if filter_data:
        if lowpass_freq is None:
            lowpass_freq = sampling_rate / 4
            
        # Design Butterworth filter
        nyquist = sampling_rate / 2
        normalized_freq = lowpass_freq / nyquist
        
        if normalized_freq >= 1.0:
            warnings.warn("Cutoff frequency too high. Skipping filtering.")
        elif len(processed_data) < 18:  # Minimum length for filtfilt
            warnings.warn("Data too short for filtering. Skipping filtering.")
        else:
            b, a = signal.butter(4, normalized_freq, btype='low')
            processed_data = signal.filtfilt(b, a, processed_data)
    
    # Normalize data
    if normalize:
        processed_data = (processed_data - np.mean(processed_data)) / np.std(processed_data)
    
    return processed_data


def _interpolate_nan(data: np.ndarray) -> np.ndarray:
    """Interpolate NaN values using linear interpolation."""
    valid_indices = ~np.isnan(data)
    
    if not np.any(valid_indices):
        raise ValueError("All values are NaN")
    
    # Use linear interpolation
    interpolated_data = np.interp(
        np.arange(len(data)),
        np.where(valid_indices)[0],
        data[valid_indices]
    )
    
    return interpolated_data


def create_phase_space(data: np.ndarray, 
                      embed_dim: int = 3, 
                      tau: int = 1) -> np.ndarray:
    """
    Create phase space reconstruction using method of delays.
    
    Parameters:
    -----------
    data : np.ndarray
        1D time series data
    embed_dim : int, default=3
        Embedding dimension
    tau : int, default=1
        Time delay
        
    Returns:
    --------
    np.ndarray
        Phase space coordinates (N x embed_dim array)
    """
    N = len(data)
    embedded_length = N - (embed_dim - 1) * tau
    
    if embedded_length <= 0:
        raise ValueError("Time series too short for given embedding parameters")
    
    embedded = np.zeros((embedded_length, embed_dim))
    
    for i in range(embed_dim):
        embedded[:, i] = data[i * tau:i * tau + embedded_length]
    
    return embedded


def estimate_embedding_parameters(data: np.ndarray,
                                max_dim: int = 10,
                                max_tau: int = 50) -> Tuple[int, int]:
    """
    Estimate optimal embedding dimension and time delay.
    
    Parameters:
    -----------
    data : np.ndarray
        1D time series data
    max_dim : int, default=10
        Maximum embedding dimension to test
    max_tau : int, default=50
        Maximum time delay to test
        
    Returns:
    --------
    Tuple[int, int]
        Optimal (embedding_dimension, time_delay)
    """
    # Estimate time delay using first minimum of autocorrelation
    tau_opt = _estimate_tau_autocorr(data, max_tau)
    
    # Estimate embedding dimension using false nearest neighbors
    embed_dim_opt = _estimate_embedding_dim_fnn(data, tau_opt, max_dim)
    
    return embed_dim_opt, tau_opt


def _estimate_tau_autocorr(data: np.ndarray, max_tau: int) -> int:
    """Estimate time delay using first minimum of autocorrelation."""
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Take positive lags only
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Find first minimum
    for tau in range(1, min(max_tau, len(autocorr) - 1)):
        if autocorr[tau] < autocorr[tau - 1] and autocorr[tau] < autocorr[tau + 1]:
            return tau
    
    return 1  # Default fallback


def _estimate_embedding_dim_fnn(data: np.ndarray, tau: int, max_dim: int) -> int:
    """Estimate embedding dimension using false nearest neighbors method."""
    
    def false_nearest_neighbors_fraction(data, embed_dim, tau):
        """Calculate fraction of false nearest neighbors."""
        try:
            # Create embedded vectors
            embedded = create_phase_space(data, embed_dim, tau)
            embedded_plus = create_phase_space(data, embed_dim + 1, tau)
            
            N = len(embedded)
            if N < 100:  # Need sufficient points
                return 1.0
            
            # Find nearest neighbors in m-dimensional space
            false_neighbors = 0
            total_neighbors = 0
            
            for i in range(N - 1):
                # Find nearest neighbor
                distances = np.linalg.norm(embedded - embedded[i], axis=1)
                distances[i] = np.inf  # Exclude self
                nearest_idx = np.argmin(distances)
                
                if nearest_idx < len(embedded_plus) - 1:
                    # Check if still nearest in (m+1)-dimensional space
                    dist_m = distances[nearest_idx]
                    dist_m_plus = np.linalg.norm(embedded_plus[i] - embedded_plus[nearest_idx])
                    
                    # Ratio test
                    if dist_m > 0:
                        ratio = dist_m_plus / dist_m
                        if ratio > 2.0:  # Typical threshold
                            false_neighbors += 1
                    
                    total_neighbors += 1
            
            if total_neighbors == 0:
                return 1.0
                
            return false_neighbors / total_neighbors
            
        except:
            return 1.0
    
    # Find dimension where FNN fraction drops below threshold
    for dim in range(1, max_dim + 1):
        fnn_fraction = false_nearest_neighbors_fraction(data, dim, tau)
        if fnn_fraction < 0.1:  # 10% threshold
            return dim
    
    return max_dim  # Fallback


def segment_data(data: np.ndarray, 
                segment_length: int,
                overlap: float = 0.0) -> list:
    """
    Segment time series data into overlapping or non-overlapping windows.
    
    Parameters:
    -----------
    data : np.ndarray
        1D time series data
    segment_length : int
        Length of each segment
    overlap : float, default=0.0
        Overlap fraction between segments (0.0 to 1.0)
        
    Returns:
    --------
    list
        List of data segments
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError("Overlap must be between 0 and 1")
    
    step_size = int(segment_length * (1 - overlap))
    segments = []
    
    start = 0
    while start + segment_length <= len(data):
        segments.append(data[start:start + segment_length])
        start += step_size
    
    return segments


def calculate_stride_intervals(gait_data: np.ndarray,
                             sampling_rate: float = 100.0,
                             method: str = 'peak_detection') -> np.ndarray:
    """
    Calculate stride intervals from gait time series data.
    
    Parameters:
    -----------
    gait_data : np.ndarray
        Gait time series (e.g., heel strike signals)
    sampling_rate : float, default=100.0
        Sampling rate in Hz
    method : str, default='peak_detection'
        Method for detecting stride events
        
    Returns:
    --------
    np.ndarray
        Array of stride intervals in seconds
    """
    if method == 'peak_detection':
        # Find peaks (heel strikes)
        peaks, _ = signal.find_peaks(gait_data, 
                                   height=np.mean(gait_data) + 0.5 * np.std(gait_data),
                                   distance=int(0.5 * sampling_rate))  # Min 0.5s between strides
        
        # Calculate intervals
        if len(peaks) < 2:
            return np.array([])
        
        intervals = np.diff(peaks) / sampling_rate
        return intervals
    
    else:
        raise ValueError(f"Unknown method: {method}")


def generate_synthetic_gait_data(length: int = 1000,
                                sampling_rate: float = 100.0,
                                noise_level: float = 0.1,
                                nonlinear_component: bool = True) -> np.ndarray:
    """
    Generate synthetic gait data for testing purposes.
    
    Parameters:
    -----------
    length : int, default=1000
        Length of the time series
    sampling_rate : float, default=100.0
        Sampling rate in Hz
    noise_level : float, default=0.1
        Level of added noise
    nonlinear_component : bool, default=True
        Whether to include nonlinear dynamics
        
    Returns:
    --------
    np.ndarray
        Synthetic gait time series
    """
    t = np.arange(length) / sampling_rate
    
    # Base periodic component (typical gait frequency ~1.2 Hz)
    base_freq = 1.2
    gait_signal = np.sin(2 * np.pi * base_freq * t)
    
    # Add harmonics for more realistic gait pattern
    gait_signal += 0.3 * np.sin(2 * np.pi * 2 * base_freq * t + np.pi/4)
    gait_signal += 0.1 * np.sin(2 * np.pi * 3 * base_freq * t - np.pi/3)
    
    if nonlinear_component:
        # Add chaotic component (simplified Lorenz-like dynamics)
        x, y, z = 1.0, 1.0, 1.0
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        dt = 1.0 / sampling_rate
        
        chaos_component = np.zeros(length)
        for i in range(length):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            x += dx * dt
            y += dy * dt
            z += dz * dt
            
            chaos_component[i] = x
        
        # Normalize and add to gait signal
        chaos_component = 0.2 * (chaos_component - np.mean(chaos_component)) / np.std(chaos_component)
        gait_signal += chaos_component
    
    # Add noise
    if noise_level > 0:
        noise = noise_level * np.random.randn(length)
        gait_signal += noise
    
    return gait_signal