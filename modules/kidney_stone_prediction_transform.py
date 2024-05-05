"""
Author: Rama Astra
Date: 5/5/2024
Usage:
- Preprocess raw data
"""

import tensorflow as tf
import tensorflow_transform as tft

FEATURE_KEYS = ["gravity", "ph", "osmo", "cond", "urea", "calc"]
LABEL_KEY = "target"


def transformed_name(key):
    """Defines transformed key name

    Args:
        key (string): Key name

    Returns:
        Transformed key name
    """
    return key + "_xf"


def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    """

    outputs = {}

    for feature in FEATURE_KEYS:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
