# Recurrent Keypoint SLDS (rSLDS) for `jax_moseq`

This repository contains a modified version of the Keypoint SLDS model from the `jax_moseq` package. The standard Switching Linear Dynamical System (SLDS) has been extended into a **Recurrent Switching Linear Dynamical System (rSLDS)**, based on the framework introduced by *Linderman et al., 2016*.

## Overview

In a standard SLDS, the discrete states ($z_t$) follow open-loop Markovian dynamics, meaning the transition to the next state $z_{t+1}$ depends entirely on $z_t$. 

In this **rSLDS** implementation, the discrete state transition probabilities are conditionally dependent on the continuous latent state ($x_t$). This allows the model to learn localized, location-dependent behavioral states (e.g., specific dynamics that trigger only when an agent is in a specific region of the state space).

## Architectural Changes from Standard SLDS

### 1. Recurrent Sticky Transitions
Rather than a static Markov transition matrix $\pi$, the model utilizes dynamic, time-varying transitions. Specifically, it implements the "Recurrent Sticky SLDS" formulation:
* **Stay Probability**: The probability of remaining in the current state $z_t$ is determined by a logistic function of the continuous state $x_t$: 
  $$P(	ext{stay}) = \sigma(x_t^T W_{	ext{stay}} + b_{	ext{stay}})$$
* **Leave Probability**: If the model determines to leave the current state, it transitions according to a base distribution $\pi_{	ext{other}}$ (which has zero probability on the diagonal).

### 2. Hybrid Inference Pipeline
The original paper utilizes Pólya-gamma augmentation to perform exact Bayesian conjugate updates (Gibbs sampling) for the recurrence weights. Because a native JAX Pólya-gamma sampler is intractable/unavailable, this pipeline substitutes exact weight sampling with gradient-based Maximum a Posteriori (MAP) optimization.
* **Weights (`W_stay`, `b_stay`)**: Updated via MAP optimization using `optax.adam` with an L2 prior to allow sharp linear decision boundaries.
* **Discrete States (`z`)**: Resampled using a custom time-varying forward-backward algorithm (`resample_time_varying_discrete_stateseqs`) that integrates the dynamic transition matrices $\pi_t$.
* **Continuous States (`x`)**: Resampled using the standard continuous state block-sampler. (Note: Because Pólya-gamma augmentation is omitted, the continuous state updates do not explicitly factor in the non-Gaussian recurrent potentials $\psi(x_t, z_{t+1})$ during the Kalman smoothing pass).

### 3. State Initialization via Decision Lists
Due to the stick-breaking construction's sensitivity to the ordering of output dimensions, the initialization pipeline utilizes a greedy decision list. It iteratively fits logistic regressions to find the sequence of states most amenable to stick-breaking isolation, returning an optimal permutation matrix $P$.

## File Modifications

* **`decision_list.py`**: *(New)* Implements the `greedy_decision_list_permutation` function to initialize the sequence of latent states for the recurrent framework.
* **`log_prob.py`**: *(Modified)* Replaces the static HMM joint likelihood with `rslds_log_joint_likelihood` and `compute_sticky_rslds_transition_active_row` to calculate the time-varying probability matrices $\pi_t$.
* **`initialize.py`**: *(Modified)* Added `init_rslds_params` to initialize the recurrent weights `W_stay` (shape: `num_states x latent_dim`), biases `b_stay`, and the off-diagonal base transition matrix `pi_other`.
* **`gibbs.py`**: *(Modified)* Overhauled the resampling loop (`resample_rslds_model`). Introduced `update_sticky_weights_map` for Adam-based optimization of the recurrent weights and implemented a custom scan-based forward-backward sampler for the time-varying categorical transitions.
* **`alignment.py`**: Handles egocentric alignment and spatial embeddings (functionally equivalent to the original `jax_moseq` pipeline).
