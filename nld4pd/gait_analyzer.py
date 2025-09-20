"""
Main GaitAnalyzer class for comprehensive gait analysis using non-linear dynamics.
"""

import numpy as np
from typing import Dict, Any, Optional, Union
from .nonlinear_methods import (
    lyapunov_exponent,
    correlation_dimension,
    fractal_dimension,
    recurrence_quantification_analysis
)
from .utils import preprocess_data
from .visualization import plot_results


class GaitAnalyzer:
    """
    Main class for analyzing gait data using non-linear dynamics methods.
    
    This class provides a unified interface for applying various non-linear
    dynamics analysis methods to gait time series data from Parkinson's disease
    patients and healthy controls.
    """
    
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize the GaitAnalyzer.
        
        Parameters:
        -----------
        sampling_rate : float, default=100.0
            Sampling rate of the gait data in Hz
        """
        self.sampling_rate = sampling_rate
        self.results = {}
        
    def analyze(self, 
                data: np.ndarray, 
                methods: Optional[list] = None,
                preprocess: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive non-linear dynamics analysis on gait data.
        
        Parameters:
        -----------
        data : np.ndarray
            1D array of gait time series data
        methods : list, optional
            List of methods to apply. If None, applies all methods.
            Options: ['lyapunov', 'correlation_dim', 'fractal_dim', 'rqa']
        preprocess : bool, default=True
            Whether to preprocess the data (detrend, normalize)
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing analysis results for each method
        """
        if methods is None:
            methods = ['lyapunov', 'correlation_dim', 'fractal_dim', 'rqa']
            
        # Preprocess data if requested
        if preprocess:
            data = preprocess_data(data)
            
        results = {}
        
        # Calculate Lyapunov exponent
        if 'lyapunov' in methods:
            try:
                results['lyapunov_exponent'] = lyapunov_exponent(data)
            except Exception as e:
                results['lyapunov_exponent'] = {'error': str(e)}
                
        # Calculate correlation dimension
        if 'correlation_dim' in methods:
            try:
                results['correlation_dimension'] = correlation_dimension(data)
            except Exception as e:
                results['correlation_dimension'] = {'error': str(e)}
                
        # Calculate fractal dimension
        if 'fractal_dim' in methods:
            try:
                results['fractal_dimension'] = fractal_dimension(data)
            except Exception as e:
                results['fractal_dimension'] = {'error': str(e)}
                
        # Perform recurrence quantification analysis
        if 'rqa' in methods:
            try:
                results['rqa'] = recurrence_quantification_analysis(data)
            except Exception as e:
                results['rqa'] = {'error': str(e)}
                
        self.results = results
        return results
        
    def plot_results(self, 
                    results: Optional[Dict[str, Any]] = None,
                    save_path: Optional[str] = None) -> None:
        """
        Plot the analysis results.
        
        Parameters:
        -----------
        results : Dict[str, Any], optional
            Results dictionary. If None, uses self.results
        save_path : str, optional
            Path to save the plot. If None, displays the plot
        """
        if results is None:
            results = self.results
            
        plot_results(results, save_path=save_path)
        
    def compare_groups(self, 
                      pd_data: list, 
                      control_data: list,
                      methods: Optional[list] = None) -> Dict[str, Any]:
        """
        Compare gait dynamics between Parkinson's disease patients and controls.
        
        Parameters:
        -----------
        pd_data : list
            List of gait time series from PD patients
        control_data : list
            List of gait time series from healthy controls
        methods : list, optional
            List of methods to apply for comparison
            
        Returns:
        --------
        Dict[str, Any]
            Comparison results including statistics and effect sizes
        """
        if methods is None:
            methods = ['lyapunov', 'correlation_dim', 'fractal_dim', 'rqa']
            
        # Analyze each group
        pd_results = []
        for data in pd_data:
            pd_results.append(self.analyze(data, methods=methods))
            
        control_results = []
        for data in control_data:
            control_results.append(self.analyze(data, methods=methods))
            
        # Compile group statistics
        comparison = {}
        for method in methods:
            method_key = self._get_method_key(method)
            
            # Extract values for each group
            pd_values = []
            control_values = []
            
            for r in pd_results:
                if method_key in r:
                    if isinstance(r[method_key], dict):
                        if 'error' not in r[method_key]:
                            # For RQA, extract specific measures
                            if 'recurrence_rate' in r[method_key]:
                                pd_values.append(r[method_key]['recurrence_rate'])
                    else:
                        pd_values.append(r[method_key])
                        
            for r in control_results:
                if method_key in r:
                    if isinstance(r[method_key], dict):
                        if 'error' not in r[method_key]:
                            # For RQA, extract specific measures
                            if 'recurrence_rate' in r[method_key]:
                                control_values.append(r[method_key]['recurrence_rate'])
                    else:
                        control_values.append(r[method_key])
            
            if pd_values and control_values:
                comparison[method] = {
                    'pd_mean': np.mean(pd_values),
                    'pd_std': np.std(pd_values),
                    'control_mean': np.mean(control_values),
                    'control_std': np.std(control_values),
                    'effect_size': self._cohen_d(pd_values, control_values)
                }
                
        return comparison
        
    def _get_method_key(self, method: str) -> str:
        """Get the dictionary key for a given method."""
        method_map = {
            'lyapunov': 'lyapunov_exponent',
            'correlation_dim': 'correlation_dimension',
            'fractal_dim': 'fractal_dimension',
            'rqa': 'rqa'
        }
        return method_map.get(method, method)
        
    def _cohen_d(self, group1: list, group2: list) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std