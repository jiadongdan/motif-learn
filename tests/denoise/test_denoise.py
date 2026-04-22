import warnings

import numpy as np
import pytest

import mtflearn.denoise as denoise_pkg
import mtflearn.denoise._denoise_svd as svd_mod
import mtflearn.denoise._denoise_svd_memory_view as svd_memory_view_mod
from mtflearn.denoise._denoise_fft import denoise_fft


def test_denoise_subpackage_exports_explicit_memory_view_name():
    assert hasattr(denoise_pkg, "DenoiseSVD")
    assert hasattr(denoise_pkg, "denoise_svd_memory_view")
    assert not hasattr(denoise_pkg, "denoise_svd")


def test_denoise_fft_preserves_shape():
    image = np.arange(16, dtype=float).reshape(4, 4)
    denoised = denoise_fft(image, p=0.25)
    assert denoised.shape == image.shape


def test_denoise_svd_rejects_exact_patch_size():
    image = np.ones((8, 8), dtype=float)
    with pytest.raises(ValueError, match="patch_size must be strictly smaller than the image dimensions."):
        svd_mod.denoise_svd(image, patch_size=8, n_components=1, verbose=False)


def test_denoise_svd_small_patch_uses_safe_default_step():
    image = np.arange(64, dtype=float).reshape(8, 8)
    denoised = svd_mod.denoise_svd(image, patch_size=3, n_components=1, extraction_step=None, verbose=False)
    assert denoised.shape == image.shape


def test_denoise_svd_class_forwards_extraction_step(monkeypatch):
    image = np.ones((8, 8), dtype=float)
    captured = {}

    def fake_denoise_svd(img, patch_size, n_components, extraction_step=None, verbose=True, return_s=False):
        captured["extraction_step"] = extraction_step
        result = np.zeros_like(img)
        if return_s:
            return result, np.array([1.0])
        return result

    monkeypatch.setattr(svd_mod, "denoise_svd", fake_denoise_svd)

    model = svd_mod.DenoiseSVD(image, n_components=1, patch_size=4, extraction_step=2)
    output = model.run(verbose=False)

    assert captured["extraction_step"] == 2
    np.testing.assert_array_equal(output, np.zeros_like(image))


def test_memory_view_rejects_rectangular_patches():
    image = np.ones((8, 8), dtype=float)
    with pytest.raises(ValueError, match="supports only square patches"):
        svd_memory_view_mod.denoise_svd(image, patch_size=(2, 3), n_components=1, show_progress=False)


def test_memory_view_handles_single_patch_without_warnings():
    image = np.ones((8, 8), dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        recon, explained_variance_ratio, n_components = svd_memory_view_mod.denoise_svd(
            image,
            patch_size=8,
            n_components=1,
            show_progress=False,
        )

    assert caught == []
    np.testing.assert_allclose(recon, image)
    assert n_components == 1
    assert not np.isnan(explained_variance_ratio).any()


def test_memory_view_handles_zero_variance_with_auto_components():
    image = np.ones((8, 8), dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        recon, explained_variance_ratio, n_components = svd_memory_view_mod.denoise_svd(
            image,
            patch_size=3,
            n_components=None,
            show_progress=False,
        )

    assert caught == []
    np.testing.assert_allclose(recon, image)
    np.testing.assert_array_equal(explained_variance_ratio, np.zeros_like(explained_variance_ratio))
    assert n_components == 1
