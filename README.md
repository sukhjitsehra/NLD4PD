# NLD4PD
Non-Linear Dynamics for Parkinson's Disease Gait Analysis

A Python library for analyzing gait patterns in Parkinson's disease using non-linear dynamics methods including chaos theory, fractal analysis, and recurrence quantification analysis.

## Features

- Lyapunov exponents calculation for gait time series
- Fractal dimension analysis using various methods
- Correlation dimension estimation
- Recurrence Quantification Analysis (RQA)
- Visualization tools for analysis results
- Data preprocessing utilities for gait data

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from nld4pd import GaitAnalyzer
import numpy as np

# Load your gait data
gait_data = np.loadtxt('gait_data.txt')

# Initialize analyzer
analyzer = GaitAnalyzer()

# Perform non-linear dynamics analysis
results = analyzer.analyze(gait_data)

# Visualize results
analyzer.plot_results(results)
```

## Methods

### Lyapunov Exponents
Quantifies the sensitivity to initial conditions in the gait dynamics.

### Fractal Dimension
Measures the complexity and self-similarity of gait patterns.

### Correlation Dimension
Estimates the dimensionality of the underlying attractor in gait dynamics.

### Recurrence Quantification Analysis (RQA)
Analyzes recurring patterns in gait time series data.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
