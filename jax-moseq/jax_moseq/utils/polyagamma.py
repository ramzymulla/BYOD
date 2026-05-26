import jax
import jax.numpy as jnp
import jax.random as jr

@jax.jit
def sample_polyagamma_jax(seed, b, c, truncation=20):
    """
    Pure JAX approximation of Pólya-Gamma sampling using a 
    truncated sum of Gamma distributions.
    
    Parameters
    ----------
    seed : jr.PRNGKey
    b : float or jax array
        Number of trials (for stick-breaking, usually 1.0).
    c : float or jax array
        Logistic log-odds (logits).
    truncation : int, default=20
        Number of terms to keep in the infinite sum. 20 is typically 
        sufficient for high accuracy in Bayesian logistic regression.
    """
    b = jnp.asarray(b, dtype=jnp.float32)
    c = jnp.asarray(c, dtype=jnp.float32)
    shape = jnp.broadcast_shapes(b.shape, c.shape)
    
    # Create the sequence n = [1, 2, ..., truncation]
    n = jnp.arange(1, truncation + 1, dtype=jnp.float32)
    
    # Reshape n to broadcast against the target batched shape
    n_reshape = (1,) * len(shape) + (truncation,)
    n = n.reshape(n_reshape)
    
    # Expand b and c for the truncation dimension
    b_expanded = jnp.expand_dims(jnp.broadcast_to(b, shape), axis=-1)
    c_expanded = jnp.expand_dims(jnp.broadcast_to(c, shape), axis=-1)
    
    # Sample gammas: g_n ~ Gamma(b, 1)
    g = jr.gamma(seed, b_expanded)
    
    # Compute denominator: (n - 0.5)^2 + c^2 / (4 * pi^2)
    denominator = (n - 0.5)**2 + (c_expanded**2) / (4 * jnp.pi**2)
    
    # Compute the truncated sum
    omega = (1.0 / (2 * jnp.pi**2)) * jnp.sum(g / denominator, axis=-1)
    
    return omega