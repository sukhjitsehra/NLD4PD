"""
NLD4PD: Non-Linear Dynamics for Parkinson's Disease Gait Analysis

A Python library for analyzing gait patterns in Parkinson's disease using 
non-linear dynamics methods including chaos theory, fractal analysis, 
and recurrence quantification analysis.
"""

from .gait_analyzer import GaitAnalyzer
from .nonlinear_methods import (
    lyapunov_exponent,
    correlation_dimension,
    fractal_dimension,
    recurrence_quantification_analysis
)
from .utils import load_gait_data, preprocess_data
from .visualization import plot_results, plot_time_series, plot_phase_space

__version__ = "0.1.0"
__author__ = "NLD4PD Contributors"

__all__ = [
    'GaitAnalyzer',
    'lyapunov_exponent',
    'correlation_dimension', 
    'fractal_dimension',
    'recurrence_quantification_analysis',
    'load_gait_data',
    'preprocess_data',
    'plot_results',
    'plot_time_series',
    'plot_phase_space'
]