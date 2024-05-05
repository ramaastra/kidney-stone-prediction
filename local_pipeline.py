"""
Author: Rama Astra
Date: 5/5/2024
Usage:
- Initiate Machine Learning pipeline based on the components defined
- Runs pipeline with BeamDagRunner
"""

import os
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = "ramaastra-pipeline"

# Pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/kidney_stone_prediction_transform.py"
TUNER_MODULE_FILE = "modules/kidney_stone_prediction_tuner.py"
TRAINER_MODULE_FILE = "modules/kidney_stone_prediction_trainer.py"

# Pipeline outputs
OUTPUT_BASE = "output"
serving_model_dir = os.path.join(OUTPUT_BASE, "serving_model")
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


def init_local_pipeline(components) -> pipeline.Pipeline:
    """
    Preprocess input features into transformed features

    Args:
        components (tuple): A tuple containing TFX components

    Return:
        Initiated pipeline based on the components defined
    """

    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = ["--direct_running_mode multi_processing", "--direct_num_workers 0"]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args,
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    initiated_components = init_components(
        {
            "data_dir": DATA_ROOT,
            "training_module": TRAINER_MODULE_FILE,
            "tuner_module": TUNER_MODULE_FILE,
            "transform_module": TRANSFORM_MODULE_FILE,
            "training_steps": 5000,
            "eval_steps": 1000,
            "serving_model_dir": serving_model_dir,
        }
    )

    pipeline = init_local_pipeline(initiated_components)
    BeamDagRunner().run(pipeline=pipeline)
