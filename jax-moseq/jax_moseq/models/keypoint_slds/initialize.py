import jax
import jax.numpy as jnp
import jax.random as jr

from jax_moseq import utils
from jax_moseq.utils import jax_io, device_put_as_scalar, check_precision

from jax_moseq.models import arhmm, slds
from jax_moseq.models.keypoint_slds.gibbs import resample_scales
from jax_moseq.models.keypoint_slds.alignment import preprocess_for_pca


def init_states(
    seed,
    Y,
    mask,
    params,
    noise_prior,
    obs_hypparams,
    Y_flat=None,
    v=None,
    h=None,
    fix_heading=False,
    **kwargs,
):
    """
    Initialize the latent states of the keypoint rSLDS.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    Y : jax array of shape (N, T, k, d)
        Keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    params : dict
        Values for each model parameter (must contain Cd, Ab, Q, W, b, sigmasq).
    noise_prior : array or scalar
        Prior on noise scale.
    obs_hypparams : dict
        Observation hyperparameters (nu_s, s_0, etc.).
    Y_flat : jax array of shape (N, T, (k-1)*d), optional
        Pre-computed aligned/flattened keypoint observations.
    v : jax array of shape (N, T, d), optional
        Pre-computed centroid positions.
    h : jax array of shape (N, T), optional
        Pre-computed heading angles.
    fix_heading : bool, default=False
        Whether to keep heading fixed at 0.
    **kwargs : dict
        Passed to preprocess_for_pca (e.g. anterior_idxs, posterior_idxs).

    Returns
    -------
    states : dict
        Initialized latent state dictionary.
    """
    if Y_flat is None:
        Y_flat, v, h = preprocess_for_pca(Y, fix_heading=fix_heading, **kwargs)

    x = slds.init_continuous_stateseqs(Y_flat, params["Cd"])

    # BUG FIX 6: arhmm.init_states requires params["pi"] to be present because
    # it calls resample_discrete_stateseqs which takes **params and expects "pi".
    # The rSLDS removes "pi" from params, so we inject a temporary uniform one.
    num_states = params["Ab"].shape[0]
    temp_params = dict(params)
    temp_params["pi"] = jnp.ones((num_states, num_states)) / num_states

    states = arhmm.init_states(seed, x, mask, temp_params)

    states["x"] = x
    states["v"] = v
    states["h"] = h
    states["s"] = resample_scales(
        seed, Y, **states, **params, s_0=noise_prior, **obs_hypparams
    )
    return states


def init_params(
    seed, pca, Y_flat, mask, trans_hypparams, ar_hypparams, whiten, k, **kwargs
):
    """
    Initialize the parameters of the keypoint SLDS from the
    data and hyperparameters.

    Parameters
    ----------
    seed : jr.PRNGKey
        JAX random seed.
    pca : sklearn.decomposition._pca.PCA
        PCA object fit to observations.
    Y_flat : jax array of shape (N, T, (k - 1) * d)
        Aligned and embedded keypoint observations.
    mask : jax array of shape (N, T)
        Binary indicator for valid frames.
    trans_hypparams : dict
        HDP transition hyperparameters.
    ar_hypparams : dict
        Autoregression hyperparameters.
    whiten : bool
        Whether to whiten PC's to initialize continuous latents.
    k : int
        Number of keypoints.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    params : dict
        Values for each model parameter.
    """
    params = arhmm.init_params(seed, trans_hypparams, ar_hypparams)
    params["Cd"] = slds.init_obs_params(pca, Y_flat, mask, whiten, **ar_hypparams)
    params["sigmasq"] = jnp.ones(k)
    return params


def init_hyperparams(
    trans_hypparams, ar_hypparams, obs_hypparams, cen_hypparams, **kwargs
):
    """
    Formats the hyperparameter dictionary of the keypoint SLDS.

    Parameters
    ----------
    trans_hypparams : dict
        HDP transition hyperparameters.
    ar_hypparams : dict
        Autoregression hyperparameters.
    obs_hypparams : dict
        Observation hyperparameters.
    cen_hypparams : dict
        Centroid movement hyperparameters.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    hypparams : dict
        Values for each group of hyperparameters.
    """
    hyperparams = slds.init_hyperparams(trans_hypparams, ar_hypparams, obs_hypparams)
    hyperparams["cen_hypparams"] = cen_hypparams.copy()
    return hyperparams


def init_model(
    data=None,
    states=None,
    params=None,
    hypparams=None,
    noise_prior=None,
    seed=jr.PRNGKey(0),
    pca=None,
    whiten=True,
    PCA_fitting_num_frames=1000000,
    anterior_idxs=None,
    posterior_idxs=None,
    conf_threshold=0.5,
    error_estimator=None,
    trans_hypparams=None,
    ar_hypparams=None,
    obs_hypparams=None,
    cen_hypparams=None,
    verbose=False,
    exclude_outliers_for_pca=True,
    fix_heading=False,
    **kwargs,
):
    """
    Initialize a keypoint rSLDS model dict containing the
    hyperparameters, noise prior, and initial seed, states,
    and parameters.

    Parameters
    ----------
    data : dict, optional
        Data dictionary containing the observations, mask,
        and (optionally) confidences.
    states : dict, optional
        State values for each latent variable, if precomputed.
    params : dict, optional
        Values for each model parameter, if precomputed.
    hypparams : dict, optional
        Values for each group of hyperparameters.
    noise_prior : array or scalar, optional
        Prior on the noise for each keypoint observation.
    seed : int or jr.PRNGKey, default=jr.PRNGKey(0)
        Initial random seed value.
    pca : sklearn.decomposition.PCA, optional
        PCA object, if precomputed.
    whiten : bool, default=True
        Whether to whiten PCs to initialize continuous latents.
    PCA_fitting_num_frames : int, default=1000000
        Maximum number of frames for PCA fitting.
    anterior_idxs : iterable of ints, optional
        Anterior keypoint indices for heading initialization.
    posterior_idxs : iterable of ints, optional
        Posterior keypoint indices for heading initialization.
    conf_threshold : float, default=0.5
        Confidence threshold for interpolation.
    error_estimator : dict, optional
        Parameters used to initialize noise_prior from confidences.
    trans_hypparams : dict, optional
        HDP transition hyperparameters.
    ar_hypparams : dict, optional
        Autoregression hyperparameters.
    obs_hypparams : dict, optional
        Observation hyperparameters.
    cen_hypparams : dict, optional
        Centroid movement hyperparameters.
    verbose : bool, default=False
        Whether to print progress info during initialization.
    exclude_outliers_for_pca : bool, default=True
        Whether to exclude low-confidence frames from PCA fitting.
    fix_heading : bool, default=False
        Whether to keep heading fixed at 0.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    model : dict
        Dictionary containing the hyperparameters, noise prior,
        and initial seed, states, and parameters of the model.
    """
    has_conf = data and ("conf" in data)
    _check_init_args(
        data,
        states,
        params,
        hypparams,
        trans_hypparams,
        ar_hypparams,
        obs_hypparams,
        cen_hypparams,
        has_conf,
        noise_prior,
        error_estimator,
        anterior_idxs,
        posterior_idxs,
    )

    model = {}

    conf = data["conf"] if has_conf else None

    if not (states and params):
        Y, mask = data["Y"], data["mask"]
        Y_flat, v, h = preprocess_for_pca(
            Y,
            anterior_idxs,
            posterior_idxs,
            conf,
            conf_threshold,
            fix_heading,
            verbose,
        )

    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)
    model["seed"] = seed

    if hypparams is None:
        if verbose:
            print("Keypoint rSLDS: Initializing hyperparameters")
        hypparams = init_hyperparams(
            trans_hypparams, ar_hypparams, obs_hypparams, cen_hypparams
        )
    else:
        hypparams = device_put_as_scalar(hypparams)
    model["hypparams"] = hypparams

    if noise_prior is None:
        if verbose:
            print("Keypoint rSLDS: Initializing noise prior")
        if has_conf:
            noise_prior = estimate_error(conf, **error_estimator)
        else:
            noise_prior = 1.0
    else:
        noise_prior = jax.device_put(noise_prior)
    model["noise_prior"] = noise_prior

    if params is None:
        if verbose:
            print("Keypoint rSLDS: Initializing parameters")

        if pca is None:
            if not exclude_outliers_for_pca or conf is None:
                pca_mask = mask
            else:
                pca_mask = jnp.logical_and(mask, (conf > conf_threshold).all(-1))
            pca = utils.fit_pca(Y_flat, pca_mask, PCA_fitting_num_frames, verbose)

        # BUG FIX 7: use the resolved hypparams dict (not the raw arg variables)
        # so num_states is always correctly sourced even when hypparams was
        # passed in pre-built by the caller.
        params = init_rslds_params(
            seed=seed,
            pca=pca,
            Y_flat=Y_flat,
            mask=mask,
            trans_hypparams=hypparams["trans_hypparams"],
            ar_hypparams=hypparams["ar_hypparams"],
            whiten=whiten,
            k=Y.shape[-2],
            num_states=hypparams["trans_hypparams"]["num_states"],
        )
    else:
        params = jax.device_put(params)

    model["params"] = params

    if states is None:
        if verbose:
            print("Keypoint rSLDS: Initializing states")
        obs_hypparams_resolved = hypparams["obs_hypparams"]
        states = init_states(
            seed,
            Y,
            mask,
            params,
            noise_prior,
            obs_hypparams_resolved,
            Y_flat,
            v,
            h,
            fix_heading,
            anterior_idxs=anterior_idxs,
            posterior_idxs=posterior_idxs,
        )
    else:
        states = jax.device_put(states)
    model["states"] = states

    return model


def estimate_error(conf, slope, intercept):
    """
    Using the provided keypoint confidences and parameters
    learned from the noise calibration, returns prior on
    the noise for each datapoint.

    Parameters
    ----------
    conf : jax array of shape (..., k)
        Confidence for each keypoint observation.
    slope : float
        Slope learned by noise calibration.
    intercept : float
        Intercept learned by noise calibration.

    Returns
    -------
    noise_prior : jax array of shape (..., k)
        Prior on the noise for each observation.
    """
    return 10 ** (2 * (jnp.log10(conf + 1e-6) * slope + intercept))


def init_rslds_params(
    seed, pca, Y_flat, mask, trans_hypparams, ar_hypparams, whiten, k, num_states, **kwargs
):
    seed_w, seed_b, seed_arhmm = jr.split(seed, 3)

    params = arhmm.init_params(seed_arhmm, trans_hypparams, ar_hypparams)
    params["Cd"] = slds.init_obs_params(pca, Y_flat, mask, whiten, **ar_hypparams)
    params["sigmasq"] = jnp.ones(k)

    latent_dim = ar_hypparams["latent_dim"]
    
    params["W_stay"] = jr.normal(seed_w, (num_states, latent_dim)) * 0.01
    
    # Initialize bias at 2.0 to match the loss prior
    params["b_stay"] = jnp.ones(num_states) * 2.0 + jr.normal(seed_b, (num_states,)) * 0.1
    
    pi_other = jnp.ones((num_states, num_states))
    diag_idx = jnp.arange(num_states)
    pi_other = pi_other.at[diag_idx, diag_idx].set(0.0)
    params["pi_other"] = pi_other / pi_other.sum(axis=-1, keepdims=True)

    return params


@check_precision
def _check_init_args(
    data,
    states,
    params,
    hypparams,
    trans_hypparams,
    ar_hypparams,
    obs_hypparams,
    cen_hypparams,
    has_conf,
    noise_prior,
    error_estimator,
    anterior_idxs,
    posterior_idxs,
):
    """
    Validates initialization arguments.  Raises ValueError if a required
    subset is missing.
    """
    if not (data or (states and params)):
        raise ValueError("Must provide either `data` or both `states` and `params`.")

    if not (
        hypparams
        or (trans_hypparams and ar_hypparams and obs_hypparams and cen_hypparams)
    ):
        raise ValueError(
            "Must provide either `hypparams` or all of `trans_hypparams`, "
            "`ar_hypparams`, `obs_hypparams`, and `cen_hypparams`."
        )

    if has_conf and (noise_prior is None) and (error_estimator is None):
        raise ValueError(
            "If confidences are provided, must also provide "
            "either `error_estimator` or `noise_prior`."
        )

    if not (states and params) and (anterior_idxs is None or posterior_idxs is None):
        raise ValueError(
            "If `states` and `params` not provided, must "
            "provide `anterior_idxs` and `posterior_idxs`."
        )

    if data:
        if ar_hypparams:
            latent_dim = ar_hypparams["latent_dim"]
        else:
            latent_dim = hypparams["ar_hypparams"]["latent_dim"]
        max_dim = (data["Y"].shape[-2] - 1) * data["Y"].shape[-1]
        if latent_dim > max_dim:
            raise ValueError(
                "`latent_dim` must be <= `(num_keypoints - 1) * keypoint_dim`. "
                f"Got latent_dim={latent_dim}, max={max_dim}."
            )
