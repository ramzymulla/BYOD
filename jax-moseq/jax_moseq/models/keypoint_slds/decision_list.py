import numpy as np
import jax.numpy as jnp
from sklearn.linear_model import LogisticRegression

def greedy_decision_list_permutation(x, z, num_states):
    """
    Implements the greedy decision list initialization from Linderman et al. 2016.
    Iteratively finds the state easiest to isolate via logistic regression.
    """
    T_z = z.shape[-1]
    n_lags = x.shape[-2] - T_z
    x_aligned = x[..., n_lags - 1 : n_lags - 1 + T_z - 1, :]
    
    # Flatten batch and time dimensions
    x_prev = np.array(x_aligned.reshape(-1, x.shape[-1]))
    z_curr = np.array(z[..., 1:].reshape(-1))
    
    U = list(range(num_states))
    P = []
    
    print("Fitting greedy decision list for stick-breaking...")
    for step in range(num_states - 1):
        best_loss = np.inf
        best_k = -1
        
        # Consider only transitions where the destination state is unassigned
        active_mask = np.isin(z_curr, U)
        X_active = x_prev[active_mask]
        z_active = z_curr[active_mask]
        
        if len(X_active) == 0:
            break
            
        for k in U:
            y_binary = (z_active == k).astype(int)
            
            # Defer states with no representation
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                loss = np.inf
            else:
                # Use class_weight='balanced' to prevent rare states from being ignored
                lr = LogisticRegression(max_iter=1000, class_weight='balanced')
                lr.fit(X_active, y_binary)
                
                probs = lr.predict_proba(X_active)
                probs = np.clip(probs, 1e-15, 1 - 1e-15)
                loss = -np.mean(y_binary * np.log(probs[:, 1]) + (1 - y_binary) * np.log(probs[:, 0]))
            
            if loss < best_loss:
                best_loss = loss
                best_k = k
                
        if best_k != -1:
            P.append(best_k)
            U.remove(best_k)
        else:
            break
            
    # Append any remaining states
    P.extend(U)
    print(f"Optimal stick-breaking permutation: {P}")
    return np.array(P)

def permute_model_states(model, P):
    """
    Reorders the discrete states z and all parameter matrices 
    according to the optimal permutation P.
    """
    P = jnp.asarray(P)
    inv_P = jnp.argsort(P)
    
    states = model["states"]
    params = model["params"]
    
    # Remap the discrete state sequence z
    states["z"] = inv_P[states["z"]]
    
    # Permute AR parameters (Ab, Q)
    params["Ab"] = params["Ab"][P]
    params["Q"] = params["Q"][P]
    
    # Permute static transition matrices if they exist
    if "pi" in params:
        params["pi"] = params["pi"][P][:, P]
        
    if "pi_other" in params:
        params["pi_other"] = params["pi_other"][P][:, P]
        
    # Permute Recurrent Weights depending on the formulation used
    if "W_stay" in params:
        params["W_stay"] = params["W_stay"][P]
        params["b_stay"] = params["b_stay"][P]
        
    if "W" in params:
        params["W"] = params["W"][P]
        params["b"] = params["b"][P]
        
    return model