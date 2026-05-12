import numpy as np
from scipy.ndimage import uniform_filter


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


def local_variance(image, window_size):
    """
    Compute a local variance map using a square moving window.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    window_size : int
        Side length of the local window.

    Returns
    -------
    ndarray
        Local variance map.
    """
    image = _validate_image(image).astype(float)
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    mean = uniform_filter(image, size=window_size, mode="reflect")
    mean_sq = uniform_filter(image**2, size=window_size, mode="reflect")
    return np.maximum(mean_sq - mean**2, 0.0)


def score_opening_background_local_variance(
    image,
    opening_size,
    spacing=None,
    shape="disk",
    window_size=None,
    n_samples=16,
    random_state=0,
):
    """
    Score the texture remaining in an opening-estimated background using
    the mean local variance.

    Lower scores indicate a smoother background with less atomic-scale
    texture remaining.
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

    from mtflearn.background._morphology import estimate_background_opening

    background = estimate_background_opening(image, size=opening_size, shape=shape)
    lv_window = max(1, int(round(spacing)))
    lv_map = local_variance(background, window_size=lv_window)
    return float(lv_map.mean())


def select_opening_size_local_variance(
    image,
    sizes,
    spacing=None,
    shape="disk",
    window_size=None,
    n_samples=16,
    random_state=0,
    relative_threshold=0.1,
):
    """
    Select the smallest opening size whose background local-variance score
    is sufficiently low relative to the first candidate size.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    sizes : sequence of int
        Candidate opening sizes to evaluate, typically in increasing order.
    spacing : float, optional
        Characteristic spacing in pixels. If omitted, estimate it.
    shape : {"disk", "square"}, default="disk"
        Structuring shape used for opening-size evaluation.
    window_size : int, optional
        Window size forwarded to the spacing estimator.
    n_samples : int, default=16
        Number of patches used by the spacing estimator.
    random_state : int or None, default=0
        Random seed for deterministic spacing estimation.
    relative_threshold : float, default=0.1
        Select the first size whose score is less than or equal to
        ``relative_threshold * scores[0]``.

    Returns
    -------
    dict
        Selection summary containing the chosen size, scores, spacing, and
        evaluated candidate sizes.
    """
    image = _validate_image(image)
    sizes = [int(s) for s in sizes]
    if len(sizes) == 0:
        raise ValueError("sizes must contain at least one candidate.")
    if any(s <= 0 for s in sizes):
        raise ValueError("sizes must contain only positive integers.")
    if relative_threshold < 0:
        raise ValueError("relative_threshold must be nonnegative.")

    if spacing is None:
        spacing = estimate_characteristic_spacing(
            image,
            window_size=window_size,
            n_samples=n_samples,
            random_state=random_state,
        )
    if spacing is None or spacing <= 0:
        raise ValueError("spacing must be positive or estimable from the image.")

    scores = [
        score_opening_background_local_variance(
            image,
            opening_size=size,
            spacing=spacing,
            shape=shape,
        )
        for size in sizes
    ]

    baseline_score = scores[0]
    cutoff = relative_threshold * baseline_score
    chosen_index = next(
        (i for i, score in enumerate(scores) if score <= cutoff),
        len(sizes) - 1,
    )

    return {
        "spacing": float(spacing),
        "sizes": sizes,
        "scores": scores,
        "cutoff": float(cutoff),
        "selected_size": sizes[chosen_index],
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
