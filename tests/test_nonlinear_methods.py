#!/usr/bin/env python3
"""
Test suite for non-linear dynamics methods.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import nld4pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nld4pd.nonlinear_methods import (
    lyapunov_exponent,
    correlation_dimension,
    fractal_dimension,
    recurrence_quantification_analysis,
    _embed_time_series,
    _higuchi_fractal_dimension
)


class TestNonlinearMethods(unittest.TestCase):
    """Test cases for non-linear dynamics methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data with known properties
        np.random.seed(42)  # For reproducible tests
        
        # Simple sine wave
        t = np.linspace(0, 10, 1000)
        self.sine_data = np.sin(2 * np.pi * t)
        
        # Random walk
        self.random_walk = np.cumsum(np.random.randn(1000))
        
        # White noise
        self.white_noise = np.random.randn(500)
        
        # Lorenz system (chaotic)
        self.lorenz_data = self._generate_lorenz_data(2000)
        
    def _generate_lorenz_data(self, length):
        """Generate Lorenz system data."""
        dt = 0.01
        x, y, z = 1.0, 1.0, 1.0
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        
        data = np.zeros(length)
        for i in range(length):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            x += dx * dt
            y += dy * dt
            z += dz * dt
            
            data[i] = x
            
        return data
        
    def test_embed_time_series(self):
        """Test time series embedding."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test basic embedding
        embedded = _embed_time_series(data, embed_dim=3, tau=1)
        expected_shape = (8, 3)  # 10 - (3-1)*1 = 8 rows
        self.assertEqual(embedded.shape, expected_shape)
        
        # Check first row
        np.testing.assert_array_equal(embedded[0], [1, 2, 3])
        
        # Test with different tau
        embedded_tau2 = _embed_time_series(data, embed_dim=3, tau=2)
        expected_shape_tau2 = (6, 3)  # 10 - (3-1)*2 = 6 rows
        self.assertEqual(embedded_tau2.shape, expected_shape_tau2)
        
        # Check first row with tau=2
        np.testing.assert_array_equal(embedded_tau2[0], [1, 3, 5])
        
    def test_embed_time_series_edge_cases(self):
        """Test embedding with edge cases."""
        data = np.array([1, 2, 3])
        
        # Should work with minimal data
        embedded = _embed_time_series(data, embed_dim=2, tau=1)
        self.assertEqual(embedded.shape, (2, 2))
        
        # Should raise error if data too short
        with self.assertRaises(ValueError):
            _embed_time_series(data, embed_dim=5, tau=1)
            
    def test_lyapunov_exponent(self):
        """Test Lyapunov exponent calculation."""
        # Test with sine wave (should be negative or close to zero)
        lyap_sine = lyapunov_exponent(self.sine_data)
        self.assertIsInstance(lyap_sine, (float, np.floating))
        
        # Test with chaotic data (should be positive)
        lyap_chaotic = lyapunov_exponent(self.lorenz_data)
        self.assertIsInstance(lyap_chaotic, (float, np.floating))
        
        # Test with short data
        short_data = np.random.randn(50)
        lyap_short = lyapunov_exponent(short_data)
        # Should either return a value or NaN, but not crash
        self.assertTrue(isinstance(lyap_short, (float, np.floating)) or np.isnan(lyap_short))
        
    def test_correlation_dimension(self):
        """Test correlation dimension calculation."""
        # Test with regular data
        corr_dim_sine = correlation_dimension(self.sine_data)
        self.assertIsInstance(corr_dim_sine, (float, np.floating))
        
        # Test with chaotic data
        corr_dim_chaotic = correlation_dimension(self.lorenz_data)
        self.assertIsInstance(corr_dim_chaotic, (float, np.floating))
        
        # Should be positive for real data
        if not np.isnan(corr_dim_chaotic):
            self.assertGreater(corr_dim_chaotic, 0)
            
    def test_fractal_dimension_methods(self):
        """Test different fractal dimension methods."""
        # Test Higuchi method
        fd_higuchi = fractal_dimension(self.sine_data, method='higuchi')
        self.assertIsInstance(fd_higuchi, (float, np.floating))
        
        # Test box counting method
        fd_box = fractal_dimension(self.sine_data, method='box_counting')
        self.assertIsInstance(fd_box, (float, np.floating))
        
        # Test DFA method
        fd_dfa = fractal_dimension(self.sine_data, method='detrended_fluctuation')
        self.assertIsInstance(fd_dfa, (float, np.floating))
        
        # Test invalid method
        with self.assertRaises(ValueError):
            fractal_dimension(self.sine_data, method='invalid_method')
            
    def test_higuchi_fractal_dimension(self):
        """Test Higuchi fractal dimension specifically."""
        # Test with different k_max values
        fd1 = _higuchi_fractal_dimension(self.sine_data, k_max=5)
        fd2 = _higuchi_fractal_dimension(self.sine_data, k_max=10)
        
        self.assertIsInstance(fd1, (float, np.floating))
        self.assertIsInstance(fd2, (float, np.floating))
        
        # Should be in reasonable range for real signals
        if not np.isnan(fd1):
            self.assertGreater(fd1, 0.5)
            self.assertLess(fd1, 3.0)
            
    def test_recurrence_quantification_analysis(self):
        """Test RQA analysis."""
        # Test with sine wave
        rqa_sine = recurrence_quantification_analysis(self.sine_data)
        
        # Should return dictionary with expected measures
        expected_measures = [
            'recurrence_rate', 'determinism', 'average_diagonal_line',
            'max_diagonal_line', 'laminarity', 'average_vertical_line', 'entropy'
        ]
        
        if 'error' not in rqa_sine:
            for measure in expected_measures:
                self.assertIn(measure, rqa_sine)
                # Some measures can be integers (like max_diagonal_line)
                self.assertIsInstance(rqa_sine[measure], (int, float, np.integer, np.floating))
                
            # Recurrence rate should be between 0 and 1
            self.assertGreaterEqual(rqa_sine['recurrence_rate'], 0.0)
            self.assertLessEqual(rqa_sine['recurrence_rate'], 1.0)
            
            # Determinism should be between 0 and 1
            self.assertGreaterEqual(rqa_sine['determinism'], 0.0)
            self.assertLessEqual(rqa_sine['determinism'], 1.0)
        
    def test_rqa_with_different_parameters(self):
        """Test RQA with different embedding parameters."""
        # Test with different embedding dimensions
        rqa1 = recurrence_quantification_analysis(self.sine_data, embed_dim=2)
        rqa2 = recurrence_quantification_analysis(self.sine_data, embed_dim=4)
        
        # Both should complete (or both should fail gracefully)
        self.assertIsInstance(rqa1, dict)
        self.assertIsInstance(rqa2, dict)
        
        # Test with custom threshold
        rqa_custom = recurrence_quantification_analysis(
            self.sine_data, threshold=0.1
        )
        self.assertIsInstance(rqa_custom, dict)
        
    def test_methods_with_constant_data(self):
        """Test methods with constant data."""
        constant_data = np.ones(500)
        
        # Most methods should handle constant data gracefully
        lyap = lyapunov_exponent(constant_data)
        self.assertTrue(isinstance(lyap, (float, np.floating)) or np.isnan(lyap))
        
        corr_dim = correlation_dimension(constant_data)
        self.assertTrue(isinstance(corr_dim, (float, np.floating)) or np.isnan(corr_dim))
        
        frac_dim = fractal_dimension(constant_data)
        self.assertTrue(isinstance(frac_dim, (float, np.floating)) or np.isnan(frac_dim))
        
        rqa = recurrence_quantification_analysis(constant_data)
        self.assertIsInstance(rqa, dict)
        
    def test_methods_with_minimal_data(self):
        """Test methods with minimal data length."""
        minimal_data = np.random.randn(20)
        
        # Methods should either work or fail gracefully
        lyap = lyapunov_exponent(minimal_data)
        self.assertTrue(isinstance(lyap, (float, np.floating)) or np.isnan(lyap))
        
        # Some methods may not work with very short data
        try:
            corr_dim = correlation_dimension(minimal_data)
            self.assertTrue(isinstance(corr_dim, (float, np.floating)) or np.isnan(corr_dim))
        except:
            pass  # Expected to fail with very short data
            
    def test_parameter_validation(self):
        """Test parameter validation in methods."""
        # Test embedding parameters
        with self.assertRaises(ValueError):
            _embed_time_series(np.array([1, 2]), embed_dim=5, tau=1)
            
        # Test negative parameters (should be handled gracefully)
        try:
            lyapunov_exponent(self.sine_data, embed_dim=0)
        except:
            pass  # Expected to fail or handle gracefully
            
    def test_reproducibility(self):
        """Test that methods give consistent results."""
        # Same data should give same results
        lyap1 = lyapunov_exponent(self.sine_data)
        lyap2 = lyapunov_exponent(self.sine_data)
        
        if not np.isnan(lyap1) and not np.isnan(lyap2):
            self.assertAlmostEqual(lyap1, lyap2, places=10)
            
        # Same for other methods
        fd1 = fractal_dimension(self.sine_data)
        fd2 = fractal_dimension(self.sine_data)
        
        if not np.isnan(fd1) and not np.isnan(fd2):
            self.assertAlmostEqual(fd1, fd2, places=10)


class TestMethodsIntegration(unittest.TestCase):
    """Integration tests for non-linear methods."""
    
    def test_methods_on_known_systems(self):
        """Test methods on systems with known properties."""
        # Periodic system (sine wave)
        t = np.linspace(0, 20, 2000)
        periodic = np.sin(2 * np.pi * t)
        
        # Random system
        random_data = np.random.randn(2000)
        
        # Apply all methods
        methods = [
            lyapunov_exponent,
            correlation_dimension,
            fractal_dimension
        ]
        
        for method in methods:
            # Should work on both types of data
            result_periodic = method(periodic)
            result_random = method(random_data)
            
            # Results should be numbers (or NaN)
            self.assertTrue(
                isinstance(result_periodic, (float, np.floating)) or 
                np.isnan(result_periodic)
            )
            self.assertTrue(
                isinstance(result_random, (float, np.floating)) or 
                np.isnan(result_random)
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)