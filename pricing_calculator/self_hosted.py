from typing import Dict, Union
import pandas as pd
from pricing_calculator.config import Config


def closest_input_size_in_the_profile(
    gpu_profile: pd.DataFrame, input_size: int
) -> int:
    previous_input_size = 0
    matching_input_size = None
    for column in gpu_profile.columns:
        if previous_input_size < input_size <= int(column):
            matching_input_size = column
        previous_input_size = int(column)

    if matching_input_size is None:
        matching_input_size = gpu_profile.columns[-1]
    return matching_input_size


def closest_batch_size(
    gpu_profile: pd.DataFrame, batch_size: int, direction: str
) -> int:
    if direction == "lower":
        return max(gpu_profile.index[gpu_profile.index <= batch_size])
    elif direction == "upper":
        return min(gpu_profile.index[gpu_profile.index >= batch_size])
    else:
        raise ValueError("Direction must be 'lower' or 'upper'.")


def ideal_batch_size(gpu_profile: pd.DataFrame, input_size: int) -> int:
    if input_size == 0:
        raise ValueError("Input size cannot be zero.")

    profile_input_size = closest_input_size_in_the_profile(gpu_profile, input_size)
    available_batch_sizes = gpu_profile[profile_input_size].dropna()
    return max(available_batch_sizes.index)


def time_per_input_token(
    io_profiles: Dict[str, Union[pd.DataFrame, Dict[str, float]]],
    config: Config,
    batch_size: int,
) -> float:
    profile_input_size = closest_input_size_in_the_profile(
        io_profiles["input_profile"], config.input_tokens
    )
    lower_batch_size = closest_batch_size(
        io_profiles["input_profile"][profile_input_size], batch_size, "lower"
    )
    upper_batch_size = closest_batch_size(
        io_profiles["input_profile"][profile_input_size], batch_size, "upper"
    )
    lower_batched_time = io_profiles["input_profile"][profile_input_size][
        lower_batch_size
    ]
    upper_batched_time = io_profiles["input_profile"][profile_input_size][
        upper_batch_size
    ]
    interpolated_time = (
        lower_batched_time + upper_batched_time / 2
    )  # Linear interpolation
    time_per_input_token = interpolated_time / (batch_size * int(profile_input_size))
    return time_per_input_token


def batch_latency(
    io_profiles: Dict[str, pd.DataFrame], config: Config, batch_size: int
) -> float:
    time_per_output_token = 1 / io_profiles["output_profile"]["throughput"]

    generation_time = time_per_output_token * config.output_tokens
    ingestion_time = (
        time_per_input_token(io_profiles, config, batch_size)
        * batch_size
        * config.input_tokens
    )
    return generation_time + ingestion_time


# def data_limited_batch_size(gpu_profile: pd.DataFrame, input_size: int, output_size, requests_per_min: int, time_per_input_token: float, time_per_output_token: float) -> int:
def data_limited_batch_size(
    io_profiles: Dict[str, Union[pd.DataFrame, Dict[str, float]]], config: Config
) -> int:
    profile_input_size = closest_input_size_in_the_profile(
        io_profiles["input_profile"], config.input_tokens
    )
    available_batch_sizes = io_profiles["input_profile"][profile_input_size].dropna()
    ideal_batch_size = max(available_batch_sizes.index)

    # Partial Differential Equation
    previous_batch_size = 0
    batch_size = max(
        2, int(config.requests_per_sec)
    )  # max batch size based on data throughput not including build-up from run-time latency

    iterations = 0
    while previous_batch_size != batch_size:
        if batch_size >= ideal_batch_size:
            return ideal_batch_size
        # TODO: Make this more accurate by using lookups on real world time per output token
        latency = batch_latency(io_profiles, config, batch_size)

        previous_batch_size = batch_size
        batch_size = max(2, int(config.requests_per_sec * latency))

        iterations += 1
        if iterations > 100:
            raise RuntimeError("Unable to converge on a batch size.")

    return batch_size


def input_server_utilization(
    io_profiles: Dict[str, Union[pd.DataFrame, Dict[str, float]]], config: Config
) -> int:
    """The number of servers required to satisfy the input throughput load."""
    # Input Throughput Load / server capacity
    batch_size = data_limited_batch_size(io_profiles, config)
    input_throughput_load = config.requests_per_sec * config.input_tokens
    server_input_capacity = 1 / time_per_input_token(io_profiles, config, batch_size)
    return input_throughput_load / server_input_capacity


def output_server_utilization(
    io_profiles: Dict[str, Union[pd.DataFrame, Dict[str, float]]], config: Config
) -> int:
    """The number of servers required to satisfy the output throughput load."""
    # Output Throughput Load / server capacity
    # Use the ideal batch size to calculate the maximum throughput capacity
    batch_size = ideal_batch_size(io_profiles["input_profile"], config.input_tokens)

    output_throughput_load = config.requests_per_sec * config.output_tokens
    server_output_capacity = io_profiles["output_profile"]["throughput"] * batch_size
    print(io_profiles["output_profile"]["throughput"])
    print("load", output_throughput_load)
    print("capacity", server_output_capacity)
    return output_throughput_load / server_output_capacity
