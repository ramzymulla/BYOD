import numpy as np
import jax.numpy as jnp
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

def _fit_single_state(k, X, z_active):
    """Helper function to fit a single candidate state concurrently."""
    y_binary = (z_active == k).astype(int)
    
    if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
        return np.inf, k, None
        
    # max_iter reduced for speed; precision is not critical for initialization
    lr = LogisticRegression(max_iter=50, solver='lbfgs', class_weight='balanced', C=1.0)
    lr.fit(X, y_binary)
    
    probs = lr.predict_proba(X)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    loss = -np.mean(y_binary * np.log(probs[:, 1]) + (1 - y_binary) * np.log(probs[:, 0]))
    
    return loss, k, lr

def greedy_decision_list_permutation(x, z, num_states, max_samples=10000, n_jobs=-1):
    """
    Implements the greedy decision list initialization with subsampling 
    and parallelization for efficient execution.
    """
    T_z = z.shape[-1]
    n_lags = x.shape[-2] - T_z
    x_aligned = x[..., n_lags - 1 : n_lags - 1 + T_z - 1, :]
    
    x_prev = np.array(x_aligned.reshape(-1, x.shape[-1]))
    z_curr = np.array(z[..., 1:].reshape(-1))
    
    # Subsample to prevent massive computational bottlenecks
    if len(x_prev) > max_samples:
        idx = np.random.choice(len(x_prev), max_samples, replace=False)
        x_eval = x_prev[idx]
        z_eval = z_curr[idx]
    else:
        x_eval = x_prev
        z_eval = z_curr
    
    U = list(range(num_states))
    P = []
    
    R_init = []
    r_init = []
    
    for step in range(num_states - 1):
        active_mask = np.isin(z_eval, U)
        X_active = x_eval[active_mask]
        z_active = z_eval[active_mask]
        
        if len(X_active) == 0:
            break
            
        # Execute independent logistic regressions in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_single_state)(k, X_active, z_active) for k in U
        )
        
        # Find the state that yielded the lowest loss
        best_loss = np.inf
        best_k = -1
        best_model = None
        
        for loss, k, model in results:
            if loss < best_loss:
                best_loss = loss
                best_k = k
                best_model = model
                
        if best_k != -1 and best_model is not None:
            P.append(best_k)
            U.remove(best_k)
            
            R_init.append(best_model.coef_[0])
            r_init.append(best_model.intercept_[0])
        else:
            break
            
    P.extend(U)
    
    M = x_prev.shape[-1]
    while len(R_init) < num_states - 1:
        R_init.append(np.zeros(M))
        r_init.append(0.0) 
    
    print(f"Optimal stick-breaking permutation: {P}")
        
    return np.array(P), np.array(R_init), np.array(r_init)

def permute_model_states(model, P, R_init=None, r_init=None):
    """
    Reorders the discrete states z and all parameter matrices 
    according to the optimal permutation P. Initializes recurrence 
    weights if extracted from the decision list.
    """
    P = jnp.asarray(P)
    inv_P = jnp.argsort(P)
    
    states = model["states"]
    params = model["params"]
    
    states["z"] = inv_P[states["z"]]
    
    params["Ab"] = params["Ab"][P]
    params["Q"] = params["Q"][P]
    
    if "pi" in params:
        params["pi"] = params["pi"][P][:, P]
        
    if "pi_other" in params:
        params["pi_other"] = params["pi_other"][P][:, P]
        
    if R_init is not None and r_init is not None:
        R_init_jnp = jnp.asarray(R_init)
        r_init_jnp = jnp.asarray(r_init).flatten()
        
        if "W" in params and "b" in params:
            if params["W"].ndim == 3:
                params["W"] = jnp.tile(R_init_jnp[None, :, :], (len(P), 1, 1))
                params["b"] = jnp.tile(r_init_jnp[None, :], (len(P), 1))
            else:
                params["W"] = R_init_jnp
                params["b"] = r_init_jnp
                
    elif "W" in params and "b" in params:
        if params["W"].ndim == 3:
            params["W"] = params["W"][P]
            params["b"] = params["b"][P]
            
    if "W_stay" in params:
        params["W_stay"] = params["W_stay"][P]
        params["b_stay"] = params["b_stay"][P]
        
    return model