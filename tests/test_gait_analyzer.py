#!/usr/bin/env python3
"""
Test suite for GaitAnalyzer class.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import nld4pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nld4pd import GaitAnalyzer
from nld4pd.utils import generate_synthetic_gait_data


class TestGaitAnalyzer(unittest.TestCase):
    """Test cases for GaitAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GaitAnalyzer(sampling_rate=100.0)
        self.test_data = generate_synthetic_gait_data(length=500, noise_level=0.1)
        
    def test_analyzer_initialization(self):
        """Test GaitAnalyzer initialization."""
        analyzer = GaitAnalyzer(sampling_rate=50.0)
        self.assertEqual(analyzer.sampling_rate, 50.0)
        self.assertEqual(analyzer.results, {})
        
    def test_analyze_basic(self):
        """Test basic analysis functionality."""
        results = self.analyzer.analyze(self.test_data)
        
        # Check that results dictionary is returned
        self.assertIsInstance(results, dict)
        
        # Check that expected methods are present
        expected_methods = ['lyapunov_exponent', 'correlation_dimension', 
                           'fractal_dimension', 'rqa']
        
        for method in expected_methods:
            self.assertIn(method, results)
            
    def test_analyze_subset_methods(self):
        """Test analysis with subset of methods."""
        methods = ['lyapunov', 'fractal_dim']
        results = self.analyzer.analyze(self.test_data, methods=methods)
        
        # Should only contain requested methods
        self.assertIn('lyapunov_exponent', results)
        self.assertIn('fractal_dimension', results)
        self.assertNotIn('correlation_dimension', results)
        self.assertNotIn('rqa', results)
        
    def test_analyze_no_preprocessing(self):
        """Test analysis without preprocessing."""
        results = self.analyzer.analyze(self.test_data, preprocess=False)
        self.assertIsInstance(results, dict)
        
    def test_short_data_handling(self):
        """Test handling of very short data."""
        short_data = np.random.randn(10)
        results = self.analyzer.analyze(short_data)
        
        # Some methods should fail with short data, but should not crash
        self.assertIsInstance(results, dict)
        
    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        # Test with NaN values
        nan_data = self.test_data.copy()
        nan_data[10:20] = np.nan
        
        results = self.analyzer.analyze(nan_data)
        self.assertIsInstance(results, dict)
        
        # Test with constant data
        constant_data = np.ones(500)
        results = self.analyzer.analyze(constant_data)
        self.assertIsInstance(results, dict)
        
    def test_method_key_mapping(self):
        """Test internal method key mapping."""
        # Test private method
        self.assertEqual(self.analyzer._get_method_key('lyapunov'), 'lyapunov_exponent')
        self.assertEqual(self.analyzer._get_method_key('fractal_dim'), 'fractal_dimension')
        self.assertEqual(self.analyzer._get_method_key('correlation_dim'), 'correlation_dimension')
        self.assertEqual(self.analyzer._get_method_key('rqa'), 'rqa')
        
    def test_cohen_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [6, 7, 8, 9, 10]
        
        effect_size = self.analyzer._cohen_d(group1, group2)
        
        # Should be a negative value (group1 < group2)
        self.assertLess(effect_size, 0)
        self.assertIsInstance(effect_size, float)
        
    def test_compare_groups(self):
        """Test group comparison functionality."""
        # Generate test data for two groups
        pd_data = [generate_synthetic_gait_data(300, noise_level=0.2) for _ in range(3)]
        control_data = [generate_synthetic_gait_data(300, noise_level=0.1) for _ in range(3)]
        
        comparison = self.analyzer.compare_groups(pd_data, control_data)
        
        # Check structure of comparison results
        self.assertIsInstance(comparison, dict)
        
        # Each comparison should have group statistics
        for method_results in comparison.values():
            self.assertIn('pd_mean', method_results)
            self.assertIn('pd_std', method_results)
            self.assertIn('control_mean', method_results)
            self.assertIn('control_std', method_results)
            self.assertIn('effect_size', method_results)
            
    def test_results_storage(self):
        """Test that results are stored in analyzer instance."""
        results = self.analyzer.analyze(self.test_data)
        
        # Results should be stored in instance
        self.assertEqual(self.analyzer.results, results)
        
    def test_different_sampling_rates(self):
        """Test analyzer with different sampling rates."""
        analyzers = [
            GaitAnalyzer(sampling_rate=50.0),
            GaitAnalyzer(sampling_rate=100.0),
            GaitAnalyzer(sampling_rate=200.0)
        ]
        
        for analyzer in analyzers:
            results = analyzer.analyze(self.test_data)
            self.assertIsInstance(results, dict)


class TestGaitAnalyzerIntegration(unittest.TestCase):
    """Integration tests for GaitAnalyzer with real-world scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GaitAnalyzer()
        
    def test_pd_vs_control_simulation(self):
        """Test realistic PD vs control simulation."""
        # Generate more realistic synthetic data
        control_data = generate_synthetic_gait_data(
            length=1000, 
            nonlinear_component=True, 
            noise_level=0.05
        )
        
        pd_data = generate_synthetic_gait_data(
            length=1000, 
            nonlinear_component=True, 
            noise_level=0.15
        )
        
        # Analyze both datasets
        control_results = self.analyzer.analyze(control_data)
        pd_results = self.analyzer.analyze(pd_data)
        
        # Both should complete without errors
        self.assertIsInstance(control_results, dict)
        self.assertIsInstance(pd_results, dict)
        
        # Check that at least some methods completed successfully
        successful_methods = 0
        for method in control_results:
            if not isinstance(control_results[method], dict) or 'error' not in control_results[method]:
                successful_methods += 1
                
        self.assertGreater(successful_methods, 0, "At least one method should complete successfully")
        
    def test_batch_analysis(self):
        """Test analysis of multiple datasets."""
        datasets = [
            generate_synthetic_gait_data(500, noise_level=0.1) for _ in range(5)
        ]
        
        results_list = []
        for data in datasets:
            results = self.analyzer.analyze(data)
            results_list.append(results)
            
        # All analyses should complete
        self.assertEqual(len(results_list), 5)
        
        # Each result should be a dictionary
        for results in results_list:
            self.assertIsInstance(results, dict)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)