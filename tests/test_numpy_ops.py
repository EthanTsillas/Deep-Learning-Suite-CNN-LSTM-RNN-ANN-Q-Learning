from __future__ import annotations

import numpy as np

from src.deep_learning.numpy_ops import conv2d_single_channel, max_pool2d, softmax


def test_conv2d_single_channel_identity_like_kernel() -> None:
    image = np.array([[1, 2], [3, 4]], dtype=np.float32)
    kernel = np.array([[1]], dtype=np.float32)
    result = conv2d_single_channel(image, kernel)
    assert np.allclose(result, image)


def test_max_pool2d() -> None:
    image = np.array([[1, 2], [3, 4]], dtype=np.float32)
    result = max_pool2d(image, pool_size=2, stride=2)
    assert result.shape == (1, 1)
    assert result[0, 0] == 4


def test_softmax_rows_sum_to_one() -> None:
    logits = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float32)
    probs = softmax(logits, axis=1)
    assert np.allclose(probs.sum(axis=1), np.ones(2))
