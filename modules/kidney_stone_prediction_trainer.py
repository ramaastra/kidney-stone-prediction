"""
Author: Rama Astra
Date: 5/5/2024
Usage:
- Defines training process of the pipeline
"""

import os
import tensorflow as tf
import tensorflow_transform as tft
from keras import layers
from keras.utils import plot_model
from modules.kidney_stone_prediction_transform import (
    LABEL_KEY,
    FEATURE_KEYS,
    transformed_name,
)
from modules.kidney_stone_prediction_tuner import gzip_reader_fn


def get_model(hyperparameters, show_summary=True):
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

    concatenate = layers.concatenate(input_features)
    deep = layers.Dense(hyperparameters["dense_units"], activation="relu")(concatenate)
    for _ in range(hyperparameters["num_hidden_layers"]):
        deep = layers.Dense(hyperparameters["dense_units"], activation="relu")(deep)
    deep = layers.Dropout(hyperparameters["dropout_rate"])(deep)
    outputs = layers.Dense(1, activation="sigmoid")(deep)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hyperparameters["learning_rate"]
        ),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    if show_summary:
        model.summary()

    return model


def get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example

    Args:
        model (tf.keras.models.Model): Keras model

    Returns:
        Serve TF examples
    """

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Generates features and labels for training
    Args:
        file_pattern: input tfrecord file pattern
        tf_transform_output: A TFTransformOutput
        batch_size (int): representing the number of consecutive elements of
        returned dataset to combine in a single batch
    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def run_fn(fn_args):
    """Train the model based on given args.

    Args:
        fn_args (dict): Holds args used to train the model as name/value pairs.
    """
    hp = fn_args.hyperparameters["values"]
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 64)

    model = get_model(hp)

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy", mode="max", patience=5
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor="val_binary_accuracy",
        mode="max",
        save_best_only=True,
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, stop_early, model_checkpoint],
        epochs=hp["tuner/epochs"],
    )

    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)

    plot_model(
        model, to_file="images/model_plot.png", show_shapes=True, show_layer_names=True
    )
