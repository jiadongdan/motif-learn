import numpy as np


def _validate_image(image):
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array.")
    return image


def estimate_characteristic_spacing(
    image,
    window_size=None,
    n_samples=16,
    random_state=0,
    **kwargs,
):
    """
    Estimate a characteristic foreground spacing in pixels.

    This helper uses the patch-size estimator from ``mtflearn.features`` as a
    reference scale for choosing background-removal parameters.
    """
    image = _validate_image(image)

    from mtflearn.features._patch_size import estimate_patch_size

    if random_state is None:
        return estimate_patch_size(
            image,
            window_size=window_size,
            n_samples=n_samples,
            **kwargs,
        )

    state = np.random.get_state()
    np.random.seed(random_state)
    try:
        return estimate_patch_size(
            image,
            window_size=window_size,
            n_samples=n_samples,
            **kwargs,
        )
    finally:
        np.random.set_state(state)


def suggest_background_parameters(
    image,
    spacing=None,
    window_size=None,
    n_samples=16,
    random_state=0,
    opening_factor=2.0,
    rolling_ball_factor=3.0,
    baseline_factor=1.5,
):
    """
    Suggest background-removal parameters from a characteristic spacing.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    spacing : float, optional
        Characteristic spacing in pixels. If omitted, estimate it from the
        image using the patch-size estimator.
    window_size : int, optional
        Window size forwarded to the spacing estimator.
    n_samples : int, default=16
        Number of patches used by the spacing estimator.
    random_state : int or None, default=0
        Random seed for deterministic spacing estimation.
    opening_factor : float, default=2.0
        Multiplier from spacing to morphology opening size.
    rolling_ball_factor : float, default=3.0
        Multiplier from spacing to rolling-ball radius.
    baseline_factor : float, default=1.5
        Multiplier from spacing to baseline smoothing sigma.

    Returns
    -------
    dict
        Dictionary containing the spacing and suggested parameters for all
        three background-removal methods.
    """
    image = _validate_image(image)

    if spacing is None:
        spacing = estimate_characteristic_spacing(
            image,
            window_size=window_size,
            n_samples=n_samples,
            random_state=random_state,
        )
    if spacing is None or spacing <= 0:
        raise ValueError("spacing must be positive or estimable from the image.")

    spacing = float(spacing)

    opening_size = max(3, int(round(opening_factor * spacing)))
    if opening_size % 2 == 0:
        opening_size += 1

    rolling_ball_radius = max(3, int(round(rolling_ball_factor * spacing)))
    baseline_sigma = max(1.0, float(baseline_factor * spacing))

    return {
        "spacing": spacing,
        "opening_size": opening_size,
        "rolling_ball_radius": rolling_ball_radius,
        "baseline_sigma": baseline_sigma,
    }


def select_background_parameter(method, image, spacing=None, **kwargs):
    """
    Select a recommended parameter for one background-removal method.

    Parameters
    ----------
    method : {"opening", "rolling_ball", "baseline"}
        Background-removal method name.
    image : ndarray
        Input 2D image.
    spacing : float, optional
        Characteristic spacing in pixels.
    **kwargs
        Additional keyword arguments forwarded to
        ``suggest_background_parameters``.

    Returns
    -------
    int or float
        Recommended parameter value for the requested method.
    """
    method = str(method).lower()
    params = suggest_background_parameters(image, spacing=spacing, **kwargs)

    if method == "opening":
        return params["opening_size"]
    if method == "rolling_ball":
        return params["rolling_ball_radius"]
    if method == "baseline":
        return params["baseline_sigma"]

    raise ValueError(
        "method must be one of {'opening', 'rolling_ball', 'baseline'}."
    )
