import pandas as pd

from pricing_calculator.datasets.common import resource_path


def get_model_names():
    return {path.name for path in (resource_path / "profiles").iterdir()}


def load(model_name: str):
    """Load tabular data from csv."""
    root_path = resource_path / "profiles" / model_name
    gpu_profiles = {}
    if not root_path.exists():
        return gpu_profiles
    gpu_names = {path.stem.split("-")[0] for path in root_path.iterdir()}
    for gpu_name in gpu_names:
        input_profile_path = root_path / f"{gpu_name}-input_profile.csv"
        output_profile_path = root_path / f"{gpu_name}-output_profile.csv"
        input_profile = pd.read_csv(input_profile_path)
        input_profile = input_profile.set_index("batch_size")
        gpu_profiles[gpu_name] = {
            "input": input_profile,
            "output": {"time-per-output-token": 0.045},
        }
    return gpu_profiles
