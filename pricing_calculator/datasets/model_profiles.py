import json
from pathlib import Path

import pandas as pd

from pricing_calculator.datasets.common import resource_path

resource_path: Path


def get_model_names():
    return {path.name for path in (resource_path / "profiles").iterdir()}


def load(model_name: str) -> list[dict]:
    """Load tabular data from csv."""
    model_profile_path = resource_path / "profiles" / model_name
    gpu_profiles = []
    if not model_profile_path.exists():
        return gpu_profiles
    platform_names = {path.name for path in model_profile_path.iterdir()}
    for platform_name in platform_names:
        platform_path = model_profile_path / platform_name
        instance_names = {path.stem.split("-")[0] for path in platform_path.iterdir()}

        for instance_name in instance_names:
            input_profile_path = platform_path / f"{instance_name}-input_profile.csv"
            output_profile_path = platform_path / f"{instance_name}-output_profile.json"
            input_profile = pd.read_csv(input_profile_path)
            input_profile = input_profile.set_index("batch_size")
            with output_profile_path.open("r") as _f:
                output_profile = json.load(_f)

            gpu_profiles.append(
                {
                    "instance_name": instance_name,
                    "platform": platform_name,
                    "input_profile": input_profile,
                    "output_profile": output_profile,
                }
            )

    return gpu_profiles
