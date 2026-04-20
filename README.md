# ORBIT: Oil-Basis Reversion with Bivariate Ito Theory

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Math: SDE & Lyapunov Theory](https://img.shields.io/badge/Math-SDE_%7C_Lyapunov_Theory-blueviolet)](https://github.com/stevetab03/ORBIT)
[![Methods: KS Calibration & Differential Evolution](https://img.shields.io/badge/Methods-KS_Calibration_%7C_Differential_Evolution-success)](https://github.com/stevetab03/ORBIT)
[![ML: Neural SDE · Bayesian Opt · LSTM](https://img.shields.io/badge/ML-Neural_SDE_%7C_Bayesian_Opt_%7C_LSTM-orange)](https://github.com/stevetab03/ORBIT)
[![BI: Power BI](https://img.shields.io/badge/BI-Power_BI-yellow)](https://github.com/stevetab03/ORBIT)

**Author:** Liyuan Zhang  
**Status:** Active Research & Development / Portfolio Showcase  

_To protect my IP, all Jupyter notebooks have been converted to HTML format using the nbconvert --no-input command. This process ensures that only the research outputs and visualizations are shared within the repository, while the underlying source code remains restricted._

---

## Executive Summary

Classical commodity futures models treat the convergence of futures prices toward spot
as a symmetric, constant-speed process. This assumption breaks down under geopolitical
supply disruption — most dramatically in April 2026, when the closure of the Strait of
Hormuz drove a $30/barrel gap between Brent spot and the June futures contract, a basis
anomaly with no precedent in modern market history.

**ORBIT** models the futures-spot basis as a directed cointegration system. The key
architectural insight is that convergence speed is not constant — it accelerates as
expiration approaches, driven by the contractual obligation of physical delivery. The
theoretical framework proves that basis variance must collapse to zero at expiry, at
a rate governed by the integrated convergence speed. The mathematics and the economics
are mutually consistent by construction.

---

## What the Model Does

- Formalizes the futures-spot basis as a two-factor stochastic system with directed
  mean reversion
- Proves that basis variance collapses to zero at expiration under general conditions
- Quantifies the approximation error when the system is simplified to a
  time-inhomogeneous mean-reverting process
- Calibrates the model to historical WTI data using a distributional objective that
  prioritizes fidelity near expiration — where it matters most
- Detects when the basis has entered a crisis regime that departs from historical
  convergence behavior

---

## Classical Engineering Pipeline

### Data Ingestion (`pipeline.py`)
Pulls WTI spot and futures price series from EIA and CME sources. Computes the basis,
time-to-expiry, and data quality flags for each observation.

### Basis Construction (`basis.py`)
Constructs the cointegrating basis series, assigns the time-to-expiry parameter tau,
and applies quality controls.

### Simulation (`sde.py`)
Simulates the bivariate stochastic system using the Milstein numerical scheme.
Retained as the classical baseline for benchmarking against the Neural SDE.

### Calibration (`calibrator.py`)
Calibrates the seven-parameter model using a distributional objective function
weighted toward near-expiry observations, with a two-stage global-then-local
optimization strategy. Retained as classical baseline.

### Nonparametric Validation (`mlp_atau.py`)
A shallow neural network provides a nonparametric estimate of convergence speed
as a function of time-to-expiry and market state. Agreement with the parametric
model validates the theoretical framework. Divergence identifies where the
parametric assumptions bind.

### Validation (`validation.py`)
Stationarity tests, distributional tests, and benchmark comparisons against the
Schwartz (1997) model.

---

## ML Enhancement Layer

The classical layer provides mathematical guarantees. The ML layer removes parametric
assumptions where the data can do better. Each classical component has a modern
counterpart. Both run in parallel — the repo benchmarks them rather than blindly
replacing one with the other.

### Neural SDE — Learned Dynamics (`neural_sde.py`)

The parametric model imposes specific functional forms on the drift and diffusion of
the bivariate system. A Neural SDE (implemented via `torchsde`) replaces these with
neural networks trained end-to-end on historical basis data, using the adjoint
sensitivity method for memory-efficient gradient computation.

Where the learned dynamics agree with the parametric model, the theoretical framework
is validated data-adaptively. Where they diverge, the mis-specification is identified
and quantified — which is more useful than assuming it is not there.

### Bayesian Optimization — Efficient Calibration (`bayes_calibrator.py`)

The classical calibration explores the parameter space using Differential Evolution,
which requires thousands of objective evaluations. Bayesian Optimization via `optuna`
(TPE sampler) builds a probabilistic surrogate of the loss surface and concentrates
evaluations where improvement is likely.

This reduces calibration from roughly five thousand evaluations to roughly one hundred,
and produces uncertainty estimates on calibrated parameters rather than point estimates
only. The DE pipeline is retained as a fallback for high-dimensional extensions.

### LSTM Convergence Speed Estimator — Path-Dependent Signal (`lstm_atau.py`)

The shallow MLP estimates convergence speed from instantaneous features — it is
memoryless. An LSTM takes the full recent path of basis observations as input,
capturing path-dependence that no instantaneous snapshot can encode.

During crisis regimes, the trajectory of the basis — whether it has been widening
rapidly, oscillating, or trending — carries predictive information about future
convergence speed. The LSTM learns this structure from data. Architecture: 2-layer
LSTM, hidden dimension 64, dropout 0.2, sequence length 20 trading days, trained
on rolling windows with expanding out-of-sample evaluation.

### Benchmarking Philosophy

No component is replaced without evidence. Each ML module runs alongside its classical
counterpart. The classical model wins on interpretability and theoretical guarantees.
The ML model wins when the data is informative enough to justify the complexity.
Both results are reported.

---

## Model Benchmarking

Calibrated on WTI data 2020–2025, evaluated on the 2026 Strait of Hormuz crisis
window (out-of-sample).

| Model | Basis RMSE | Variance Collapse | Calibration Speed | Interpretability |
|---|---|---|---|---|
| **ORBIT — Classical** | TBD | Yes — proven | ~5,000 evals (DE) | Full |
| **ORBIT — Bayesian Opt** | TBD | Yes — proven | ~100 evals (TPE) | Full |
| **ORBIT — Neural SDE** | TBD | Empirical | Gradient (adjoint) | Partial |
| **ORBIT — LSTM a(τ)** | TBD | Yes — proven | Gradient | Full + path |
| MLP Baseline | TBD | None | — | None |
| Schwartz (1997) | TBD | Partial | — | Medium |
| Random Walk | — | None | — | None |

*Results to be populated upon full calibration run.*

---

## Power BI Integration

The Python pipeline exports structured CSVs consumed by a Power BI dashboard via a
star schema data model. This bridges the quantitative research layer with enterprise
analytics — the same architectural pattern used in upstream BI deployments.

**CSV exports:**

| File | Contents |
|---|---|
| `basis_panel.csv` | Daily basis, tau, contract metadata |
| `variance_by_tau.csv` | Model vs empirical variance by time-to-expiry |
| `calibration_results.csv` | Parameter estimates, classical vs Bayesian |
| `regime_signals.csv` | LSTM convergence speed estimates, regime flags |
| `model_comparison.csv` | RMSE benchmarks across all model variants |

**Dashboard pages (Power BI):**
1. Basis Overview — spread, tau, rolling volatility
2. Variance Collapse — modeled vs empirical term structure
3. Calibration — parameter estimates and uncertainty bands
4. ML Signals — LSTM path-dependent convergence speed, regime overlay
5. Model Comparison — classical vs ML RMSE benchmarks

---

## Repository Structure

```
ORBIT/
├── README.md
├── theory/
│   └── orbit_monograph.pdf           full mathematical derivation (compiled)
├── orbit/
│   ├── pipeline.py                   EIA + CME data ingestion
│   ├── basis.py                      basis computation, tau, quality flags
│   ├── sde.py                        Milstein simulator (classical)
│   ├── calibrator.py                 KS-weighted differential evolution
│   ├── mlp_atau.py                   nonparametric a(tau) via shallow MLP
│   ├── validation.py                 ADF, KS tests, Schwartz benchmark
│   ├── neural_sde.py                 Neural SDE via torchsde (ML)
│   ├── bayes_calibrator.py           Bayesian optimization via optuna (ML)
│   └── lstm_atau.py                  path-dependent a(tau) via LSTM (ML)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_calibration_demo.ipynb
│   ├── 03_results_and_charts.ipynb   Power BI export
│   └── 04_ml_benchmarking.ipynb      classical vs ML comparison
└── outputs/
    ├── figures/
    └── powerbi/
        ├── basis_panel.csv
        ├── variance_by_tau.csv
        ├── calibration_results.csv
        ├── regime_signals.csv
        └── model_comparison.csv
```

---

## Retrospective

The SVMA project ([stevetab03/SVMA](https://github.com/stevetab03/SVMA)) established
that a structurally grounded SDE system can outperform pure deep learning for
high-fidelity modeling of market microstructure dynamics. ORBIT applies the same
philosophy to a domain where the driving mechanism is not latent flow but a harder,
more fundamental law: contractual delivery obligation.

The ML layer is not a replacement for the mathematics. It is an honesty check. Where
Neural SDE and LSTM agree with the classical framework, the theory is validated
data-adaptively. Where they diverge, the parametric assumptions are exposed — which
is more useful than pretending they are not there.

---

## Contact

**Monograph:** Available upon request for full derivations, proofs, and theorem statements.  
**LinkedIn:** https://www.linkedin.com/in/hlzhang/  
**GitHub:** https://github.com/stevetab03
