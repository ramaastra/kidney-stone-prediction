"""
Author: Rama Astra
Date: 5/5/2024
Usage:
- Defines models' hyperparameter tuning process of the pipeline
"""

from typing import NamedTuple, Dict, Text, Any
import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from keras_tuner.engine import base_tuner
from keras import layers
from modules.kidney_stone_prediction_transform import (
    LABEL_KEY,
    FEATURE_KEYS,
    transformed_name,
)


def gzip_reader_fn(filenames):
    """Loads compressed data

    Args:
        filenames (tf.data.Dataset): Filenames to read

    Returns:
        TFRecordDataset reader
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(
    file_pattern, tf_transform_output, num_epochs, batch_size=64
) -> tf.data.Dataset:
    """Get post_tranform feature & create batches of data
    Args:
        file_pattern: Input tfrecord file pattern
        tf_transform_output: A TFTransformOutput
        num_epochs: (int): Number of times to read through the dataset
        batch_size (int): Representing the number of consecutive elements of
        returned dataset to combine in a single batch
    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices
    """

    # Get post_transform feature spec
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    # Create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def get_tuner_model(hyperparameters, show_summary=True):
    """Creates the model architecture

    Args:
        hyperparameters (dict): Tuner's best hyperparameter values
        show_summary (bool): True if shows summary after the model structured

    Returns:
        Keras model
    """

    input_features = []

    for feature in FEATURE_KEYS:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    dense_units = hyperparameters.Int(
        "dense_units", min_value=16, max_value=256, step=16
    )
    num_hidden_layers = hyperparameters.Choice("num_hidden_layers", values=[1, 2, 3])
    dropout_rate = hyperparameters.Float(
        "dropout_rate", min_value=0.1, max_value=0.4, step=0.1
    )
    learning_rate = hyperparameters.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    concatenate = layers.concatenate(input_features)
    deep = layers.Dense(dense_units, activation="relu")(concatenate)
    for _ in range(num_hidden_layers):
        deep = layers.Dense(dense_units, activation="relu")(deep)
    deep = layers.Dropout(dropout_rate)(deep)
    outputs = layers.Dense(1, activation="sigmoid")(deep)

    tuner_model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    tuner_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    if show_summary:
        tuner_model.summary()

    return tuner_model


def tuner_fn(fn_args):
    """Build the tuner using the KerasTuner API.
    Args:
      fn_args: Holds args used to tune models as name/value pairs.

    Returns:
      A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                      model , e.g., the training and validation dataset. Required
                      args depend on the above tuner"s implementation.
    """
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, num_epochs=10)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, num_epochs=10)

    # Define the tuner
    tuner = kt.Hyperband(
        lambda hp: get_tuner_model(hp, show_summary=False),
        objective="val_binary_accuracy",
        max_epochs=10,
        factor=3,
        directory=fn_args.working_dir,
        project_name="kidney_stone_prediction_tuner",
    )

    # Define early stopping callback
    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1)

    TunerFnResult = NamedTuple(
        "TunerFnResult",
        [("tuner", base_tuner.BaseTuner), ("fit_kwargs", Dict[Text, Any])],
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [stop_early],
            "x": train_set,
            "validation_data": val_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )
