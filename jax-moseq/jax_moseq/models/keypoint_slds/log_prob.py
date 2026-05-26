import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from jax_moseq.models import arhmm, slds
from jax_moseq.models.keypoint_slds.alignment import estimate_coordinates

na = jnp.newaxis


def location_log_prob(v, sigmasq_loc):
    """
    Calculate the log probability of the centroid location at each
    time-step, given the prior on centroid movement.

    Parameters
    ----------
    v : jax array of shape (..., T, d)
        Centroid positions.
    sigmasq_loc : float
        Assumed variance in centroid displacements.

    Returns
    -------
    log_pv: jax array of shape (..., T - 1)
        Log probability of `v`.
    """
    v0 = v[..., :-1, :]
    v1 = v[..., 1:, :]
    sigma = jnp.sqrt(sigmasq_loc) * jnp.ones_like(v0)
    return tfd.MultivariateNormalDiag(v0, sigma).log_prob(v1)


def obs_log_prob(Y, x, v, h, s, Cd, sigmasq, **kwargs):
    """
    Calculate the log probability of keypoint coordinates at each
    time-step, given continuous latent trajectories, centroids, heading
    angles, noise scales, and observation parameters.

    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    x : jax array of shape (..., latent_dim)
        Latent trajectories.
    v : jax array of shape (..., d)
        Centroid positions.
    h : jax array
        Heading angles.
    s : jax array of shape (..., k)
        Noise scales.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    log_pY: jax array of shape (..., k)
        Log probability of `Y`.
    """
    Y_bar = estimate_coordinates(x, v, h, Cd)
    sigma = jnp.broadcast_to(jnp.sqrt(s * sigmasq)[..., na], Y.shape)
    return tfd.MultivariateNormalDiag(Y_bar, sigma).log_prob(Y)


@jax.jit
def log_joint_likelihood(
    Y, mask, x, v, h, s, z, pi, Ab, Q, Cd, sigmasq, sigmasq_loc, s_0, nu_s, **kwargs
):
    """
    Calculate the total log probability for each latent state (standard SLDS).

    Parameters
    ----------
    Y : jax array of shape (..., T, k, d)
        Keypoint observations.
    mask : jax array of shape (..., T)
        Binary indicator for valid frames.
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    v : jax array of shape (..., T, d)
        Centroid positions.
    h : jax array of shape (..., T)
        Heading angles.
    s : jax array of shape (..., T, k)
        Noise scales.
    z : jax_array of shape (..., T - n_lags)
        Discrete state sequences.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    sigmasq_loc : float
        Assumed variance in centroid displacements.
    s_0 : scalar or jax array broadcastable to `Y`
        Prior on noise scale.
    nu_s : int
        Chi-squared degrees of freedom in noise prior.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    ll: dict
        Dictionary mapping the name of each state variable to
        its total log probability.
    """
    ll = arhmm.log_joint_likelihood(x, mask, z, pi, Ab, Q)

    log_pY = obs_log_prob(Y, x, v, h, s, Cd, sigmasq)
    log_ps = slds.scale_log_prob(s, s_0, nu_s)
    log_pv = location_log_prob(v, sigmasq_loc)

    ll["Y"] = (log_pY * mask[..., na]).sum()
    ll["s"] = (log_ps * mask[..., na]).sum()
    ll["v"] = (log_pv * mask[..., 1:]).sum()
    return ll


def model_likelihood(data, states, params, hypparams, noise_prior, **kwargs):
    """
    Convenience wrapper that invokes `log_joint_likelihood`.

    Parameters
    ----------
    data : dict
        Data dictionary containing the observations and mask.
    states : dict
        State values for each latent variable.
    params : dict
        Values for each model parameter.
    hypparams : dict
        Values for each group of hyperparameters.
    noise_prior : scalar or jax array broadcastable to `s`
        Prior on noise scale.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    ll : dict
        Dictionary mapping state variable name to its
        total log probability.
    """
    return log_joint_likelihood(
        **data,
        **states,
        **params,
        **hypparams["obs_hypparams"],
        **hypparams["cen_hypparams"],
        s_0=noise_prior,
    )


def compute_rslds_transitions(x, W, b):
    """
    Compute dynamic transition matrices using previous continuous states
    via stick-breaking logistic regression.

    Parameters
    ----------
    x : jax array of shape (..., T, latent_dim)
        Continuous latent trajectories.
    W : jax array of shape (num_states, num_states - 1, latent_dim)
        Recurrent weight matrices.
    b : jax array of shape (num_states, num_states - 1)
        Recurrent bias vectors.

    Returns
    -------
    pi_t : jax array of shape (..., T - 1, num_states, num_states)
        Time-varying transition probability matrices.
        pi_t[..., t, i, j] = P(z_{t+1}=j | z_t=i, x_t).
    """
    x_prev = x[..., :-1, :]  # (..., T-1, latent_dim)

    # psi[..., t, from_state, stick_idx]: log-odds for each stick
    psi = jnp.einsum("...tl, jkl -> ...tjk", x_prev, W) + b

    stick_probs = jax.nn.sigmoid(psi)  # (..., T-1, K, K-1)

    # Stick-breaking construction: π_j = σ(ψ_j) ∏_{k<j} (1 - σ(ψ_k))
    remainders = jnp.concatenate(
        [jnp.ones_like(stick_probs[..., :1]), 1.0 - stick_probs], axis=-1
    )  # (..., T-1, K, K)
    cum_remainders = jnp.cumprod(remainders, axis=-1)

    sticks_full = jnp.concatenate(
        [stick_probs, jnp.ones_like(stick_probs[..., :1])], axis=-1
    )  # (..., T-1, K, K)

    pi_t = sticks_full * cum_remainders  # (..., T-1, K, K)
    return pi_t

def compute_sticky_rslds_transition_active_row(x_prev, z_prev, W_stay, b_stay, pi_other):
    """Compute transitions originating only from the active state."""
    W_active = W_stay[z_prev]
    b_active = b_stay[z_prev]
    
    # P(stay)
    logits = jnp.einsum("...td, ...td -> ...t", x_prev, W_active) + b_active
    
    # Keep likelihood bounds consistent with the Gibbs sampler
    logits = jnp.clip(logits, -5.0, 5.0)
    
    p_stay = jax.nn.sigmoid(logits)
    
    # Base transition row for the active state
    pi_other_active = pi_other[z_prev]  # (..., t, K)
    
    # Trans_probs: (1 - p_stay) * pi_other if leaving, p_stay if staying
    trans_probs = (1.0 - p_stay[..., jnp.newaxis]) * pi_other_active
    
    K = pi_other.shape[-1]
    z_prev_oh = jax.nn.one_hot(z_prev, K)
    trans_probs = trans_probs * (1.0 - z_prev_oh) + p_stay[..., jnp.newaxis] * z_prev_oh
    
    return trans_probs

def rslds_log_joint_likelihood(
    Y, mask, x, v, h, s, z, W_stay, b_stay, pi_other, Ab, Q, Cd, sigmasq, sigmasq_loc, s_0, nu_s, **kwargs
):
    T_z = z.shape[-1]
    n_lags = x.shape[-2] - T_z

    x_prev = x[..., n_lags - 1 : n_lags - 1 + T_z - 1, :]
    z_curr = z[..., 1:]
    z_prev = z[..., :-1]
    num_states = W_stay.shape[0]

    trans_probs = compute_sticky_rslds_transition_active_row(
        x_prev, z_prev, W_stay, b_stay, pi_other
    )

    z_curr_oh = jax.nn.one_hot(z_curr, num_states)
    log_pz = jnp.sum(jnp.log(trans_probs + 1e-12) * z_curr_oh, axis=-1)

    log_pY = obs_log_prob(Y, x, v, h, s, Cd, sigmasq)
    log_ps = slds.scale_log_prob(s, s_0, nu_s)
    log_pv = location_log_prob(v, sigmasq_loc)

    ll = {}
    ll["z"] = (log_pz * mask[..., n_lags + 1 :]).sum()
    ll["Y"] = (log_pY * mask[..., na]).sum()
    ll["s"] = (log_ps * mask[..., na]).sum()
    ll["v"] = (log_pv * mask[..., 1:]).sum()
    return ll
