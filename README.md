# Analyzing state transitions with marginal models

This repository contains a Python module for analyzing state transitions in sequential data.  Additionally, code is also included for running numerical experiments investigating the validity of these procedures.

# Requirements

The module requires a Python installation with NumPy, SciPy, statsmodels, and joblib.  It has been tested on Python 3.8.

# Usage
## Section 2.4
The default parameters of the `run_simulations` function execute experiment 1 (Figures 1 and 2), while the parameters for experiment 2 (Figure 3) are given below.
```Python
sim_data = transition_analysis.run_simulations(
    num_trials=10000,
    base_rates=np.array([0.6, 0.2, 0.1, 0.1]),
    seq_lengths=np.arange(3, 151)
)
```

## Section 3.2

The command below runs the no self-transition simulations with three states.
```Python
sim_data = transition_analysis.run_no_self_simulations(
    num_trials=10000,
    base_rates=np.ones(3) / 3,
    seq_lengths=np.arange(3, 151)
)
```
The command below runs the no self-transition simulations with four states.
```Python
sim_data = transition_analysis.run_no_self_simulations(
    num_trials=10000,
    base_rates=np.ones(4) / 4,
    seq_lengths=np.arange(3, 151)
)
```

## Section 4.5
An example command for running a multiple comparisons analysis with no false nulls, using n = 200 sequences.
```python
sim_data = transition_analysis.run_sequence_sims(
      rate=0., n_jobs=1, num_trials=200)
fdr_marginal = transition_analysis.analyze_sequence_results(
      sim_data, dependence=False, L_star=False)
fdr_L_star = transition_analysis.analyze_sequence_results(
      sim_data, dependence=False, L_star=True)
```

## Section 4.6
An example command for running a multiple comparisons analysis with false nulls, using n = 200 sequences.
```python
sim_data = transition_analysis.run_sequence_sims(
      rate=0.05, n_jobs=1, num_trials=200)
fdr_marginal = transition_analysis.analyze_sequence_results(
      sim_data, dependence=True, L_star=False)
fdr_L_star = transition_analysis.analyze_sequence_results(
      sim_data, dependence=True, L_star=True)
```    
