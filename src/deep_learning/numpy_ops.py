from __future__ import annotations

import numpy as np


def conv2d_single_channel(image: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1) -> np.ndarray:
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must both be 2D arrays")
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant")
    kh, kw = kernel.shape
    out_h = (image.shape[0] - kh) // stride + 1
    out_w = (image.shape[1] - kw) // stride + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)
    for r in range(out_h):
        for c in range(out_w):
            region = image[r * stride : r * stride + kh, c * stride : c * stride + kw]
            output[r, c] = np.sum(region * kernel)
    return output


def max_pool2d(image: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    out_h = (image.shape[0] - pool_size) // stride + 1
    out_w = (image.shape[1] - pool_size) // stride + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)
    for r in range(out_h):
        for c in range(out_w):
            region = image[r * stride : r * stride + pool_size, c * stride : c * stride + pool_size]
            output[r, c] = np.max(region)
    return output


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def laplacian_edge_detect(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    return conv2d_single_channel(image, kernel, padding=1)
