import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
import optax

from jax_moseq.utils.kalman import kalman_sample
from jax_moseq.utils.distributions import sample_vonmises_fisher

from jax_moseq.models import arhmm, slds
from jax_moseq.models.keypoint_slds.alignment import (
    to_vanilla_slds,
    estimate_coordinates,
    estimate_aligned,
    apply_rotation,
    vector_to_angle,
)
from jax_moseq.models.keypoint_slds.log_prob import compute_rslds_transitions
from jax_moseq.utils.autoregression import get_nlags, ar_log_likelihood


na = jnp.newaxis


@partial(jax.jit, static_argnames=("parallel_message_passing",))
def resample_continuous_stateseqs(
    seed,
    Y,
    mask,
    v,
    h,
    s,
    z,
    Cd,
    sigmasq,
    Ab,
    Q,
    jitter=1e-3,
    parallel_message_passing=True,
    **kwargs
):
    """
    Resamples the latent trajectories ``x``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    v : jax array of shape (N, T, d)
        Centroid positions.
    h : jax array of shape (N, T)
        Heading angles.
    s : jax array of shape (N, T, k)
        Noise scales.
    z : jax_array of shape (N, T - n_lags)
        Discrete state sequences.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    jitter : float, default=1e-3
        Amount to boost the diagonal of the covariance matrix
        during backward-sampling of the continuous states.
    parallel_message_passing : bool, default=True,
        Use associative scan for Kalman sampling, which is faster on
        a GPU but has a significantly longer jit time.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    """
    Y, s, Cd, sigmasq = to_vanilla_slds(Y, v, h, s, Cd, sigmasq)
    x = slds.resample_continuous_stateseqs(
        seed,
        Y,
        mask,
        z,
        s,
        Ab,
        Q,
        Cd,
        sigmasq,
        jitter=jitter,
        parallel_message_passing=parallel_message_passing,
    )
    return x


@jax.jit
def resample_obs_variance(seed, Y, mask, Cd, x, v, h, s, nu_sigma, sigmasq_0, **kwargs):
    """
    Resample the observation variance ``sigmasq``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    v : jax array of shape (N, T, d)
        Centroid positions.
    h : jax array of shape (N, T)
        Heading angles.
    s : jax array of shape (N, T, k)
        Noise scales.
    nu_sigma : float
        Chi-squared degrees of freedom in sigmasq.
    sigmasq_0 : float
        Scaled inverse chi-squared scaling parameter for sigmasq.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    sigmasq : jax_array of shape k
        Unscaled noise.
    """
    sqerr = compute_squared_error(Y, x, v, h, Cd, mask)
    return slds.resample_obs_variance_from_sqerr(
        seed, sqerr, mask, s, nu_sigma, sigmasq_0
    )


@jax.jit
def resample_scales(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, **kwargs):
    """
    Resample the scale values ``s``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    v : jax array of shape (N, T, d)
        Centroid positions.
    h : jax array of shape (N, T)
        Heading angles.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    nu_s : int
        Chi-squared degrees of freedom in noise prior.
    s_0 : scalar or jax array broadcastable to ``Y``
        Prior on noise scale.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    s : jax array of shape (N, T, k)
        Noise scales.
    """
    sqerr = compute_squared_error(Y, x, v, h, Cd)
    return slds.resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)


@jax.jit
def compute_squared_error(Y, x, v, h, Cd, mask=None):
    """
    Computes the squared error between model predicted
    and true observations.

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
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    mask : jax array, optional
        Binary indicator for valid frames.

    Returns
    ------
    sqerr : jax array of shape (..., k)
        Squared error between model predicted and
        true observations.
    """
    Y_est = estimate_coordinates(x, v, h, Cd)
    sqerr = ((Y - Y_est) ** 2).sum(-1)
    if mask is not None:
        sqerr = mask[..., na] * sqerr
    return sqerr


@jax.jit
def resample_heading(seed, Y, x, v, s, Cd, sigmasq, **kwargs):
    """
    Resample the heading angles ``h``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    v : jax array of shape (N, T, d)
        Centroid positions.
    s : jax array of shape (N, T, k)
        Noise scales.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    h : jax array of shape (N, T)
        Heading angles.
    """
    k = Y.shape[-2]

    Y_bar = estimate_aligned(x, Cd, k)
    Y_cent = Y - v[..., na, :]
    variance = s * sigmasq

    # [(..., t, k, d, na) * (..., t, k, na, d) / (..., t, k, na, na)] -> (..., t, d, d)
    S = (Y_bar[..., :2, na] * Y_cent[..., na, :2] / variance[..., na, na]).sum(-3)
    del Y_bar, Y_cent, variance  # free up memory

    kappa_cos = S[..., 0, 0] + S[..., 1, 1]
    kappa_sin = S[..., 0, 1] - S[..., 1, 0]
    del S

    mean_direction = jnp.stack([kappa_cos, kappa_sin], axis=-1)
    sampled_direction = sample_vonmises_fisher(seed, mean_direction)
    h = vector_to_angle(sampled_direction)
    return h


@partial(jax.jit, static_argnames=("parallel_message_passing",))
def resample_location(
    seed,
    Y,
    mask,
    x,
    h,
    s,
    Cd,
    sigmasq,
    sigmasq_loc,
    parallel_message_passing=True,
    **kwargs
):
    """
    Resample the centroid positions ``v``.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    x : jax array of shape (N, T, latent_dim)
        Latent trajectories.
    h : jax array of shape (N, T)
        Heading angles.
    s : jax array of shape (N, T, k)
        Noise scales.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    sigmasq_loc : float
        Assumed variance in centroid displacements.
    parallel_message_passing : bool, default=True,
        Use associative scan for Kalman sampling, which is faster on
        a GPU but has a significantly longer jit time.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    v : jax array of shape (N, T, d)
        Centroid positions.
    """
    k, d = Y.shape[-2:]

    Y_rot = apply_rotation(estimate_aligned(x, Cd, k), h)

    variance = s * sigmasq
    gammasq = 1 / (1 / variance).sum(-1, keepdims=True)

    mu = jnp.einsum("...tkd, ...tk->...td", Y - Y_rot, gammasq / variance)

    # Apply Kalman filter to get smooth headings
    seed = jr.split(seed, mask.shape[0])
    m0 = jnp.zeros(d)
    S0 = jnp.eye(d) * 1e4
    A = jnp.eye(d)[na]
    B = jnp.zeros(d)[na]
    Q = jnp.eye(d)[na] * sigmasq_loc
    C = jnp.eye(d)
    D = jnp.zeros(d)
    R = jnp.repeat(gammasq, d, axis=-1)
    zz = jnp.zeros_like(mask[:, 1:], dtype=int)

    masked_dynamics_noise = sigmasq_loc * 10
    masked_obs_noise = sigmasq.max() * 10

    masked_dynamics_params = {
        "weights": jnp.eye(d),
        "bias": jnp.zeros(d),
        "cov": jnp.eye(d) * masked_dynamics_noise,
    }

    masked_obs_noise_diag = jnp.ones(d) * masked_obs_noise

    in_axes = (0, 0, 0, 0, na, na, na, na, na, na, na, 0, na, na)
    v = jax.vmap(partial(kalman_sample, parallel=parallel_message_passing), in_axes)(
        seed,
        mu,
        mask,
        zz,
        m0,
        S0,
        A,
        B,
        Q,
        C,
        D,
        R,
        masked_dynamics_params,
        masked_obs_noise_diag,
    )
    return v


def resample_model(
    data,
    seed,
    states,
    params,
    hypparams,
    noise_prior,
    ar_only=False,
    states_only=False,
    resample_global_noise_scale=False,
    resample_local_noise_scale=True,
    fix_heading=False,
    verbose=False,
    jitter=1e-3,
    parallel_message_passing=False,
    **kwargs
):
    """
    Resamples the Keypoint SLDS model given the hyperparameters,
    data, noise prior, current states, and current parameters.

    Parameters
    ----------
    data : dict
        Data dictionary containing the observations and mask.
    seed : jr.PRNGKey
        JAX random seed.
    states : dict
        State values for each latent variable.
    params : dict
        Values for each model parameter.
    hypparams : dict
        Values for each group of hyperparameters.
    noise_prior : scalar or jax array broadcastable to ``s``
        Prior on noise scale.
    ar_only : bool, default=False
        Whether to restrict sampling to ARHMM components.
    states_only : bool, default=False
        Whether to restrict sampling to states.
    resample_global_noise_scale : bool, default=False
        Whether to resample the global noise scales (``sigmasq``)
    resample_local_noise_scale : bool, default=True
        Whether to resample the local noise scales (``s``)
    fix_heading : bool, default=False
        Whether to exclude ``h`` from resampling.
    jitter : float, default=1e-3
        Amount to boost the diagonal of the covariance matrix
        during backward-sampling of the continuous states.
    verbose : bool, default=False
        Whether to print progress info during resampling.
    parallel_message_passing : bool, default=False,
        Use associative scan for Kalman sampling, which is faster on
        a GPU but has a significantly longer jit time.

    Returns
    ------
    model : dict
        Dictionary containing the hyperparameters and
        updated seed, states, and parameters of the model.
    """
    model = arhmm.resample_model(
        data, seed, states, params, hypparams, states_only, verbose=verbose
    )
    if ar_only:
        model["noise_prior"] = noise_prior
        return model

    seed = model["seed"]
    params = model["params"].copy()
    states = model["states"].copy()

    if (not states_only) and resample_global_noise_scale:
        if verbose:
            print("Resampling sigmasq (global noise scales)")
        params["sigmasq"] = resample_obs_variance(
            seed,
            **data,
            **states,
            **params,
            s_0=noise_prior,
            **hypparams["obs_hypparams"]
        )

    if verbose:
        print("Resampling x (continuous latent states)")
    states["x"] = resample_continuous_stateseqs(
        seed,
        **data,
        **states,
        **params,
        jitter=jitter,
        parallel_message_passing=parallel_message_passing
    )

    if not fix_heading:
        if verbose:
            print("Resampling h (heading)")
        states["h"] = resample_heading(seed, **data, **states, **params)

    if verbose:
        print("Resampling v (location)")
    states["v"] = resample_location(
        seed, **data, **states, **params, **hypparams["cen_hypparams"]
    )

    if resample_local_noise_scale:
        if verbose:
            print("Resampling s (local noise scales)")
        states["s"] = resample_scales(
            seed,
            **data,
            **states,
            **params,
            s_0=noise_prior,
            **hypparams["obs_hypparams"]
        )

    return {
        "seed": seed,
        "states": states,
        "params": params,
        "hypparams": hypparams,
        "noise_prior": noise_prior,
    }


# ============================================================
# rSLDS-specific Gibbs steps
# ============================================================
def sticky_rslds_loss(params, x_prev, z_prev, z_curr, kappa):
    W_stay, b_stay = params["W_stay"], params["b_stay"]
    
    W_active = W_stay[z_prev]
    b_active = b_stay[z_prev]
    
    logits = jnp.einsum("td, td -> t", x_prev, W_active) + b_active
    stay_mask = (z_prev == z_curr).astype(jnp.float32)
    
    # Binary Cross Entropy for stay vs. leave
    nll = -jnp.sum(stay_mask * jax.nn.log_sigmoid(logits) + 
                   (1.0 - stay_mask) * jax.nn.log_sigmoid(-logits))
    
    # Do NOT penalize b_stay. It must be free to absorb the mean of x_prev.
    # Reduce L2 on W_stay to allow sharp linear decision boundaries.
    l2_W = 0.5 * 1.0 * jnp.sum(W_stay ** 2)
    
    return nll + l2_W

@partial(jax.jit, static_argnames=("num_iters",))
def update_sticky_weights_map(x_prev, z_prev, z_curr, W_init, b_init, kappa, num_iters=50):
    optimizer = optax.adam(learning_rate=1e-2)
    params = {"W_stay": W_init, "b_stay": b_init}
    opt_state = optimizer.init(params)
    
    def step(carry, _):
        p, state = carry
        loss, grads = jax.value_and_grad(sticky_rslds_loss)(p, x_prev, z_prev, z_curr, kappa)
        updates, state = optimizer.update(grads, state)
        p = optax.apply_updates(p, updates)
        return (p, state), loss

    (final_params, _), _ = jax.lax.scan(step, (params, opt_state), None, length=num_iters)
    return final_params["W_stay"], final_params["b_stay"]

@partial(jax.jit, static_argnames=("num_states",))
def resample_pi_other(seed, z, num_states, alpha=10.0):
    z_prev = z[..., :-1].reshape(-1)
    z_curr = z[..., 1:].reshape(-1)
    
    leave_mask = (z_prev != z_curr).astype(jnp.float32)
    
    counts = jnp.zeros((num_states, num_states))
    counts = counts.at[z_prev, z_curr].add(leave_mask)
    
    posterior_alpha = counts + alpha
    diag_idx = jnp.arange(num_states)
    posterior_alpha = posterior_alpha.at[diag_idx, diag_idx].set(0.0) 
    
    gammas = jr.gamma(seed, posterior_alpha + 1e-8)
    gammas = gammas.at[diag_idx, diag_idx].set(0.0)
    pi_base = gammas / gammas.sum(axis=-1, keepdims=True)
    
    # ANTI-DEATH FLOOR: Guarantee a 0.5% chance of jumping to any state
    floor = 0.005
    pi_safe = pi_base + floor
    pi_safe = pi_safe.at[diag_idx, diag_idx].set(0.0)
    
    return pi_safe / pi_safe.sum(axis=-1, keepdims=True)

def compute_sticky_rslds_transition_step(x_t, W_stay, b_stay, pi_other):
    logits = jnp.einsum("...l, kl -> ...k", x_t, W_stay) + b_stay
    p_stay = jax.nn.sigmoid(logits)
    
    p_stay_expanded = p_stay[..., jnp.newaxis]
    pi_t = (1.0 - p_stay_expanded) * pi_other
    
    diag_indices = jnp.arange(W_stay.shape[0])
    pi_t = pi_t.at[..., diag_indices, diag_indices].set(p_stay)
    return pi_t

@partial(jax.jit, static_argnames=("num_states",))
def resample_time_varying_discrete_stateseqs(
    seed, x_aligned, W_stay, b_stay, pi_other, log_lkhds, mask, num_states
):
    K = num_states

    def forward_step(log_alpha_prev, inputs):
        x_t, log_obs = inputs
        pi_t = compute_sticky_rslds_transition_step(x_t, W_stay, b_stay, pi_other)
        log_alpha_t = log_obs + jax.scipy.special.logsumexp(
            log_alpha_prev[..., :, None] + jnp.log(pi_t + 1e-12), axis=-2
        )
        return log_alpha_t, log_alpha_t

    log_alpha_0 = log_lkhds[..., 0, :]
    x_scan = jnp.moveaxis(x_aligned[..., :-1, :], -2, 0)
    log_lkhds_scan = jnp.moveaxis(log_lkhds[..., 1:, :], -2, 0)

    _, log_alphas_rest = jax.lax.scan(forward_step, log_alpha_0, (x_scan, log_lkhds_scan))
    log_alphas_rest = jnp.moveaxis(log_alphas_rest, 0, -2)
    log_alphas = jnp.concatenate([log_alpha_0[..., None, :], log_alphas_rest], axis=-2)

    def backward_step(carry, inputs):
        z_next, key = carry
        log_alpha, x_t = inputs

        pi_t = compute_sticky_rslds_transition_step(x_t, W_stay, b_stay, pi_other)
        z_next_oh = jax.nn.one_hot(z_next, K)
        log_trans_to_next = jnp.log(jnp.einsum("...ij, ...j -> ...i", pi_t, z_next_oh) + 1e-12)
        logits = log_alpha + log_trans_to_next
        
        key, subkey = jr.split(key)
        z_t = jr.categorical(subkey, logits, axis=-1)
        return (z_t, key), z_t

    seed, init_key = jr.split(seed)
    z_T = jr.categorical(init_key, log_alphas[..., -1, :], axis=-1)

    log_alphas_bwd = jnp.flip(jnp.moveaxis(log_alphas[..., :-1, :], -2, 0), axis=0)
    x_bwd = jnp.flip(x_scan, axis=0)

    seed, bwd_key = jr.split(seed)
    _, z_rev = jax.lax.scan(backward_step, (z_T, bwd_key), (log_alphas_bwd, x_bwd))
    z_fwd = jnp.flip(z_rev, axis=0)
    z_fwd = jnp.moveaxis(z_fwd, 0, -1)
    z = jnp.concatenate([z_fwd, z_T[..., None]], axis=-1)
    return z

def resample_rslds_model(
    data,
    seed,
    states,
    params,
    hypparams,
    noise_prior,
    ar_only=False,
    states_only=False,
    jitter=1e-2,
    parallel_message_passing=False,
    verbose=False,
    **kwargs
):
    # -----------------------------------------------------------------
    # AR-HMM WARMUP PHASE
    # -----------------------------------------------------------------
    if ar_only:
        # Construct a targeted data dictionary. The AR-HMM's "observations" 
        # are the continuous PCA latents (x) of the rSLDS.
        ar_data = {
            "Y": states["x"],
            "mask": data["mask"]
        }
        
        ar_model = arhmm.resample_model(
            ar_data, seed, states, params, hypparams, states_only, verbose=verbose
        )
        
        # Safely merge AR-HMM updates into the full parameter/state dicts.
        states = states.copy()
        params = params.copy()
        states.update(ar_model["states"])
        params.update(ar_model["params"])
        
        return {
            "seed": ar_model["seed"],
            "states": states,
            "params": params,
            "hypparams": hypparams,
            "noise_prior": noise_prior,
        }
    # -----------------------------------------------------------------
    # RECURRENT SLDS PHASE
    # -----------------------------------------------------------------
    seed, seed_w, seed_z = jr.split(seed, 3)

    states = states.copy()
    params = params.copy()

    T_z = states["z"].shape[-1]
    n_lags = states["x"].shape[-2] - T_z
    x_aligned = states["x"][..., n_lags - 1 : n_lags - 1 + T_z, :]
    num_states = hypparams["trans_hypparams"]["num_states"]

    if not states_only:
        if verbose:
            print("Updating W_stay, b_stay (MAP) and pi_other")
            
        kappa = hypparams["trans_hypparams"].get("kappa", 1e4)
        alpha = hypparams["trans_hypparams"].get("alpha", 10.0)
        
        x_prev = x_aligned[..., :-1, :].reshape(-1, params["W_stay"].shape[-1])
        z_prev = states["z"][..., :-1].reshape(-1)
        z_curr = states["z"][..., 1:].reshape(-1)

        params["W_stay"], params["b_stay"] = update_sticky_weights_map(
            x_prev, z_prev, z_curr, params["W_stay"], params["b_stay"], kappa
        )
        
        params["pi_other"] = resample_pi_other(seed_w, states["z"], num_states, alpha)

    log_lkhds_states_first = jax.lax.map(
        partial(ar_log_likelihood, states["x"]),
        (params["Ab"], params["Q"])
    )
    log_lkhds = jnp.moveaxis(log_lkhds_states_first, 0, -1)

    if verbose:
        print("Resampling z (discrete latent states)")
        
    states["z"] = resample_time_varying_discrete_stateseqs(
        seed_z, x_aligned, params["W_stay"], params["b_stay"], params["pi_other"], log_lkhds, data["mask"], num_states
    )

    if not states_only:
        if verbose:
            print("Resampling Ab, Q (AR parameters)")
        params["Ab"], params["Q"] = arhmm.resample_ar_params(
            seed,
            x=states["x"],
            z=states["z"],
            mask=data["mask"],
            **hypparams["ar_hypparams"],
        )

    if verbose:
        print("Resampling x (continuous latent states)")
    states["x"] = resample_continuous_stateseqs(
        seed,
        **data,
        **states,
        **params,
        jitter=jitter,
        parallel_message_passing=parallel_message_passing,
    )

    if not kwargs.get("fix_heading", False):
        if verbose:
            print("Resampling h (heading)")
        states["h"] = resample_heading(seed, **data, **states, **params)

    if verbose:
        print("Resampling v (location)")
    states["v"] = resample_location(
        seed, **data, **states, **params, **hypparams["cen_hypparams"],
        parallel_message_passing=parallel_message_passing,
    )

    if kwargs.get("resample_local_noise_scale", True):
        if verbose:
            print("Resampling s (local noise scales)")
        states["s"] = resample_scales(
            seed,
            **data,
            **states,
            **params,
            s_0=noise_prior,
            **hypparams["obs_hypparams"],
        )

    return {
        "seed": seed,
        "states": states,
        "params": params,
        "hypparams": hypparams,
        "noise_prior": noise_prior,
    }
