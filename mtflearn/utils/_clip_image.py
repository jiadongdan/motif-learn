import numpy as np

def _mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))

def percentile_clip(
        img,
        low=1.0,
        high=99.0,
        method="auto",           # "ratio", "mad", "iqr", "auto"
        # ratio method
        high_ratio_thresh=5.0,
        # MAD method
        mad_k=8.0,               # larger = less sensitive
        # IQR method
        iqr_k=3.0,               # larger = less sensitive
        eps=1e-8,
        copy=True,
):
    """
    Conditionally apply percentile clipping using robust outlier detection.

    Parameters
    ----------
    img : np.ndarray
        Image of shape (H, W) or (H, W, C).
    low, high : float
        Percentiles used for clipping when triggered.
    method : str
        "ratio", "mad", "iqr", or "auto".
    high_ratio_thresh : float
        For method="ratio": clip if max / p_high > threshold.
    mad_k : float
        For method="mad": clip if max is beyond median + mad_k * 1.4826 * MAD.
    iqr_k : float
        For method="iqr": clip if max is beyond Q3 + iqr_k * IQR.
    eps : float
        Numerical safety.
    copy : bool
        If True, copy into float32.

    Returns
    -------
    out : np.ndarray
        float32 image (clipped or original).
    did_clip : bool
        Whether clipping was applied.
    info : dict
        Diagnostics and thresholds used.
    """
    x = np.array(img, dtype=np.float32, copy=copy)
    flat = x.reshape(-1)

    # Basic stats
    x_max = float(flat.max())
    x_min = float(flat.min())
    p_high = float(np.percentile(flat, high))
    p_low = float(np.percentile(flat, low))

    # ----- Method 1: ratio -----
    high_ratio = x_max / (p_high + eps) if abs(p_high) > eps else np.inf
    ratio_flag = high_ratio > high_ratio_thresh

    # ----- Method 2: MAD -----
    med = float(np.median(flat))
    mad = float(_mad(flat))
    robust_sigma = 1.4826 * mad
    mad_upper = med + mad_k * robust_sigma
    mad_flag = (robust_sigma > eps) and (x_max > mad_upper)

    # ----- Method 3: IQR -----
    q1 = float(np.percentile(flat, 25))
    q3 = float(np.percentile(flat, 75))
    iqr = q3 - q1
    iqr_upper = q3 + iqr_k * iqr
    iqr_flag = (iqr > eps) and (x_max > iqr_upper)

    # Decide
    method = method.lower()
    if method == "ratio":
        did_clip = ratio_flag
    elif method == "mad":
        did_clip = mad_flag
    elif method == "iqr":
        did_clip = iqr_flag
    elif method == "auto":
        did_clip = ratio_flag or mad_flag or iqr_flag
    else:
        raise ValueError("method must be 'ratio', 'mad', 'iqr', or 'auto'.")

    info = {
        "did_clip": did_clip,
        "method": method,
        "low": low,
        "high": high,
        "min": x_min,
        "p_low": p_low,
        "median": med,
        "p_high": p_high,
        "max": x_max,
        # ratio diagnostics
        "high_ratio": float(high_ratio),
        "high_ratio_thresh": high_ratio_thresh,
        "ratio_flag": bool(ratio_flag),
        # MAD diagnostics
        "mad": float(mad),
        "robust_sigma(1.4826*MAD)": float(robust_sigma),
        "mad_k": mad_k,
        "mad_upper": float(mad_upper),
        "mad_flag": bool(mad_flag),
        # IQR diagnostics
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "iqr_k": iqr_k,
        "iqr_upper": float(iqr_upper),
        "iqr_flag": bool(iqr_flag),
    }

    if not did_clip:
        return x, False, info

    vmin = float(np.percentile(flat, low))
    vmax = float(np.percentile(flat, high))
    out = np.clip(x, vmin, vmax)
    info.update({"vmin": vmin, "vmax": vmax})

    return out, True, info

def value_clip(img, vmin, vmax, copy=True):
    x = np.array(img, dtype=np.float32, copy=copy)
    return np.clip(x, float(vmin), float(vmax))

