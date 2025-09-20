#!/usr/bin/env python3
"""
Basic usage example for NLD4PD library.

This script demonstrates how to use the NLD4PD library for analyzing
gait data using non-linear dynamics methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from nld4pd import GaitAnalyzer
from nld4pd.utils import generate_synthetic_gait_data, preprocess_data
from nld4pd.visualization import plot_time_series, plot_phase_space


def main():
    """Main demonstration function."""
    print("NLD4PD - Non-Linear Dynamics for Parkinson's Disease Gait Analysis")
    print("=" * 70)
    
    # Generate synthetic gait data for demonstration
    print("\n1. Generating synthetic gait data...")
    
    # Generate data for a healthy control
    control_data = generate_synthetic_gait_data(
        length=2000, 
        nonlinear_component=True, 
        noise_level=0.05
    )
    
    # Generate data simulating PD gait (more irregular)
    pd_data = generate_synthetic_gait_data(
        length=2000, 
        nonlinear_component=True, 
        noise_level=0.15  # Higher noise for PD simulation
    )
    
    print(f"Generated control data: {len(control_data)} samples")
    print(f"Generated PD data: {len(pd_data)} samples")
    
    # Visualize the time series
    print("\n2. Visualizing time series...")
    plot_time_series(control_data, title="Control Subject - Gait Time Series")
    plot_time_series(pd_data, title="PD Patient - Gait Time Series")
    
    # Initialize the analyzer
    print("\n3. Initializing GaitAnalyzer...")
    analyzer = GaitAnalyzer(sampling_rate=100.0)
    
    # Analyze control data
    print("\n4. Analyzing control subject data...")
    control_results = analyzer.analyze(control_data)
    
    print("Control Results:")
    for method, result in control_results.items():
        if isinstance(result, dict) and 'error' in result:
            print(f"  {method}: ERROR - {result['error']}")
        elif isinstance(result, dict):
            print(f"  {method}: {result}")
        else:
            print(f"  {method}: {result:.6f}")
    
    # Analyze PD data
    print("\n5. Analyzing PD patient data...")
    pd_results = analyzer.analyze(pd_data)
    
    print("PD Results:")
    for method, result in pd_results.items():
        if isinstance(result, dict) and 'error' in result:
            print(f"  {method}: ERROR - {result['error']}")
        elif isinstance(result, dict):
            print(f"  {method}: {result}")
        else:
            print(f"  {method}: {result:.6f}")
    
    # Plot analysis results
    print("\n6. Plotting analysis results...")
    analyzer.plot_results(control_results)
    plt.suptitle("Control Subject - Analysis Results")
    plt.show()
    
    analyzer.plot_results(pd_results)
    plt.suptitle("PD Patient - Analysis Results")
    plt.show()
    
    # Visualize phase space
    print("\n7. Creating phase space reconstructions...")
    plot_phase_space(control_data, embed_dim=3, tau=10, 
                    title="Control Subject - Phase Space")
    plot_phase_space(pd_data, embed_dim=3, tau=10, 
                    title="PD Patient - Phase Space")
    
    # Compare specific measures
    print("\n8. Comparing key measures:")
    print("-" * 40)
    
    measures_to_compare = [
        ('lyapunov_exponent', 'Lyapunov Exponent'),
        ('fractal_dimension', 'Fractal Dimension'),
        ('correlation_dimension', 'Correlation Dimension')
    ]
    
    for measure_key, measure_name in measures_to_compare:
        if (measure_key in control_results and measure_key in pd_results and
            'error' not in str(control_results[measure_key]) and
            'error' not in str(pd_results[measure_key])):
            
            control_val = control_results[measure_key]
            pd_val = pd_results[measure_key]
            
            # Handle dict results (like RQA)
            if isinstance(control_val, dict):
                if 'recurrence_rate' in control_val:
                    control_val = control_val['recurrence_rate']
                    pd_val = pd_val['recurrence_rate']
                    measure_name += ' (RR)'
                else:
                    continue
            
            print(f"{measure_name}:")
            print(f"  Control: {control_val:.6f}")
            print(f"  PD:      {pd_val:.6f}")
            print(f"  Difference: {pd_val - control_val:.6f}")
            print()
    
    print("\nExample completed successfully!")
    print("\nNote: This example uses synthetic data for demonstration.")
    print("For real analysis, load your gait data using:")
    print("  from nld4pd.utils import load_gait_data")
    print("  data = load_gait_data('your_data_file.txt')")


if __name__ == "__main__":
    main()