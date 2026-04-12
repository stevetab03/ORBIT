# ORBIT: Oil-Basis Reversion with Bivariate Ito Theory

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Math: SDE & Lyapunov Theory](https://img.shields.io/badge/Math-SDE_%7C_Lyapunov_Theory-blueviolet)](https://github.com/hlzhang/orbit/blob/main)
[![Methods: KS Calibration & Differential Evolution](https://img.shields.io/badge/Methods-KS_Calibration_%7C_Differential_Evolution-success)](https://github.com/hlzhang/orbit/blob/main)

**Author:** Liyuan Zhang  
**Status:** Active Research / Portfolio Showcase

---

## Executive Summary

Classical commodity futures models (Schwartz 1997, Gibson-Schwartz 1990) treat the convergence of futures prices toward spot as a symmetric, constant-speed process. This assumption breaks down under geopolitical supply disruption — most dramatically in April 2026, when the closure of the Strait of Hormuz drove a $30/barrel gap between Brent spot ($124.68) and June futures ($94.75), a basis anomaly with no precedent in modern market history.

**ORBIT** formalizes the futures-spot basis as a directed cointegration system in which the futures price is the follower and the spot price is the anchor. The convergence speed $a(\tau)$ is not constant — it accelerates as expiration approaches, driven by the contractual obligation of delivery. The central result is a provable basis collapse: $\sigma^2_e(\tau) = O(1/a(\tau)) \to 0$ as $\tau \to 0$, with a rate that is exponential in the integrated convergence speed.

---

## The Mathematical Engine

### Core SDE System

ORBIT models the state vector $Z_t = (F_t, S_t)^\top$ where $F_t$ is the futures price and $S_t$ is the spot price, coupled through directed mean reversion:

$$dF_t = a(\tau)(S_t - F_t)\,dt + \sigma_F\,dW^1_t$$

$$dS_t = b(\mu_S - S_t)\,dt + \sigma_S\,dW^2_t$$

$$d\langle W^1, W^2 \rangle_t = \rho\,dt$$

The architectural contribution is the **time-varying convergence speed**:

$$a(\tau) = a_0 + \frac{\kappa}{\tau + \varepsilon}, \qquad \tau = T - t$$

As expiration approaches ($\tau \to 0$), $a(\tau) \to \infty$ — the futures price is pulled toward spot with infinite force, consistent with the contractual delivery obligation. Far from expiration ($\tau \to \infty$), $a(\tau) \to a_0$ — the baseline independent reversion speed.

### Basis Process via Ito Product Rule

Define the cointegrating parameter $\lambda(\tau) = a(\tau)/(a(\tau)+b)$ and the basis $e_t = F_t - \lambda(\tau) S_t$. Applying Ito's product rule with $d\tau = -dt$:

$$de_t = \underbrace{-a(\tau) e_t\,dt}_{\text{directed pull}}
       + \underbrace{b\lambda(\tau)(\mu_S - S_t)\,dt}_{\text{slow spot drift}}
       - \underbrace{S_t \lambda'(\tau)\,dt}_{\text{Ito correction}}
       + \sigma_{\text{eff}}(\tau)\,dB_t$$

The Ito correction term $-S_t \lambda' dt$ arises from the time-dependence of $\lambda(\tau)$ and vanishes as $\tau \to 0$ (see Corollary 1 below).

---

## Time-Varying Lyapunov Equation & Variance Collapse

The covariance matrix $V(t) = \text{Cov}(Z_t)$ satisfies:

$$\frac{dV}{dt} = A(\tau)V + VA(\tau)^\top + D$$

where:

```math
A(\tau) = \begin{pmatrix} -a(\tau) & a(\tau) \\ 0 & -b \end{pmatrix}, \qquad
D = \begin{pmatrix} \sigma_F^2 & \rho\sigma_F\sigma_S \\ \rho\sigma_F\sigma_S & \sigma_S^2 \end{pmatrix}
```

**Theorem 1 (Variance Collapse).** Under the fast-slow regime
$a(\tau) \gg b$, the basis variance satisfies:

```math
\sigma_e^2(\tau) = O\left(\frac{1}{a(\tau)}\right) + O\left(e^{-\int_0^t a(s)\,ds}\right) \to 0 \quad \text{as } \tau \to 0
```

**Theorem 2 (Fast-Slow Approximation).** The basis is approximated
by a time-inhomogeneous OU process with error:

$$\|e_t - e_t^{\text{approx}}\| \leq C \cdot \left(\frac{b}{a(\tau)} + \frac{|\lambda'|}{a(\tau)}\right)$$

**Corollary 1 (Exactness at Expiry).** As $\tau \to 0$:

$$\frac{|\lambda'|}{a(\tau)} = \frac{b/\kappa \cdot (\tau+\varepsilon)}{\kappa} \to 0$$

The approximation error vanishes precisely when contractual convergence is most binding — the mathematics and the economics align.

---

## Engineering & Calibration Pipeline

### KS-Weighted Objective Function

Standard SDE calibration minimizes MSE between model and empirical moments. ORBIT uses a distributional objective that prioritizes fidelity of the full basis CDF near expiration:

$$\mathcal{L}(\theta) = \sup_x \left| F_n^{\text{MC}}(x;\theta) - F_n^{\text{emp}}(x) \right| \cdot w(\tau)$$

$$w(\tau) = \frac{1}{(\tau + \varepsilon)^\alpha}, \qquad \alpha > 0$$

The KS statistic captures the full distributional shape — critical during crisis regimes where the basis distribution departs sharply from its pre-disruption form.

### Global/Local Hybrid Calibration

- **Stage 1 (Global):** Differential Evolution explores the 7-parameter space $\theta = (a_0, \kappa, b, \sigma_F, \sigma_S, \rho, \lambda)$ to avoid local minima.
- **Stage 2 (Local):** L-BFGS-B refines the DE output to gradient-level accuracy within stability bounds.

### Numerical Integration

- **Method:** Milstein scheme for the bivariate correlated SDE.
- **Design:** For additive noise the Milstein correction vanishes, so Euler-Maruyama and Milstein coincide — the Milstein framework is retained for correctness under multiplicative noise extensions.

### Nonparametric Validation of $a(\tau)$

A shallow MLP regressor trained on $(\tau, z_t, \sigma_{\text{realized}})$ provides a nonparametric estimate $a_{\text{MLP}}(\tau)$. Agreement with the parametric form validates Theorem 2. Divergence identifies regimes where the parametric assumption is binding.

---

## Model Benchmarking

Calibrated on WTI data 2020–2025, evaluated on the 2026 Strait of Hormuz crisis window (out-of-sample).

| Model | Basis RMSE | Variance Collapse Prediction | Interpretability |
|---|---|---|---|
| **ORBIT (Full Framework)** | **TBD** | **Yes — exact** | **Full** |
| MLP Baseline | TBD | None | None |
| Schwartz (1997) | TBD | Partial | Medium |
| Random Walk | — | None | None |

*Results to be populated upon full calibration run.*

---

## Retrospective: Why This Model and Why Now

The SVMA project (see [stevetab03/SVMA](https://github.com/stevetab03/SVMA)) established that a structurally grounded, activation-driven SDE system can outperform LSTM deep learning for high-fidelity modeling of microstructure dynamics. ORBIT applies that same mathematical philosophy to a domain where the driving mechanism is not options dealer flow but a harder, more fundamental law: **contractual delivery obligation**.

Futures and spot prices must converge. The question is not whether but how fast, how noisily, and how that structure deforms under the largest supply disruption in the history of the global oil market. The current crisis — with Brent spot trading $30 above the June futures contract — is not an obstacle to the model. It is the most rigorous possible test of it.

---

## Repository Structure
```
ORBIT/
├── README.md
├── theory/
│   └── orbit_monograph.pdf          full mathematical derivation
├── orbit/
│   ├── pipeline.py                  EIA + CME data ingestion
│   ├── basis.py                     basis computation, tau, quality flags
│   ├── sde.py                       Milstein simulator
│   ├── calibrator.py                KS-weighted differential evolution
│   ├── mlp_atau.py                  nonparametric a(tau) via MLP
│   └── validation.py                ADF, KS tests, Schwartz benchmark
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_calibration_demo.ipynb
│   └── 03_results_and_charts.ipynb  Power BI export
└── outputs/
├── figures/
└── powerbi/
├── basis_panel.csv
├── variance_by_tau.csv
└── calibration_results.csv
```

---

## Contact

**LinkedIn:** https://www.linkedin.com/in/hlzhang/  
**GitHub:** https://github.com/stevetab03
