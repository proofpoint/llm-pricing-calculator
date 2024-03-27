from typing import Dict, List
import locale
import pandas as pd
import math
from pathlib import Path

from pricing_calculator.datasets.llm_as_a_service import SAASModelInfo
from pricing_calculator.config import Config
from pricing_calculator.datasets.hosting_costs import HostedGPUInfo
from pricing_calculator.self_hosted import (
    data_limited_batch_size,
    input_server_utilization,
    output_server_utilization,
)


def _req_per_min_to_req_per_sec(req_per_min: float) -> float:
    return req_per_min / 60


def _tokens_per_sec(requests_per_min: float, tokens_per_request: float) -> float:
    return _req_per_min_to_req_per_sec(requests_per_min) * tokens_per_request


def total_tokens_per_month(requests_per_min: float, tokens_per_request: float) -> float:
    """Return the total number of tokens per month."""
    return requests_per_min * 60 * 24 * 30 * tokens_per_request


def get_type1_latency(model_profile_path: Path, config: Config) -> float:
    input_profile = pd.read_csv(model_profile_path / "input_profile.csv")
    # nearest input_tokens in input_profile to config.input_tokens
    nearest_matching_row = input_profile.iloc[
        (input_profile["input_tokens"] - config.input_tokens).abs().argsort()[:1]
    ]
    nearest_matching_input_size = nearest_matching_row["input_tokens"].values[0]
    input_latency = nearest_matching_row["latency"].values[0]
    throughput = nearest_matching_input_size / input_latency
    input_time = config.input_tokens / throughput
    output_profile = pd.read_csv(model_profile_path / "output_profile.csv")
    avg_tokens = sum(output_profile["output_tokens"]) / len(
        output_profile["output_tokens"]
    )
    avg_time = sum(output_profile["latency"]) / len(output_profile["latency"])
    avg_throughput = avg_tokens / avg_time
    output_time = config.output_tokens / avg_throughput
    return input_time + output_time


def monthly_cost_for_llm_as_a_service(
    config: Config, model_pricing: List[SAASModelInfo]
):
    """Return the monthly cost for a LLM as a Service."""
    input_tokens_per_month = total_tokens_per_month(
        config.requests_per_min, config.input_tokens
    )
    output_tokens_per_month = total_tokens_per_month(
        config.requests_per_min, config.output_tokens
    )
    costs = {}
    for model_info in model_pricing:
        if model_info.model in config.models:
            input_cost = (
                input_tokens_per_month * model_info.input_cost_per_1k_tokens / 1000
            )
            output_cost = (
                output_tokens_per_month * model_info.output_cost_per_1k_tokens / 1000
            )
            costs[model_info.model] = {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost,
            }
            model_profile_path = Path("resources/type1_profiles") / model_info.model_id
            if model_info.model_id != "" and model_profile_path.exists():
                costs[model_info.model][
                    "latency"
                ] = f"{get_type1_latency(model_profile_path, config):.3f}s"
            else:
                costs[model_info.model]["latency"] = "unknown"
    return costs


def format_currency(value: float) -> str:
    return "${:,.2f}".format(value)


def sprint_price(price: float) -> str:
    """Return the price as a string."""
    return format_currency(price)


def pricing_print(costs: Dict[str, float]):
    """Print the pricing."""
    longest_key = max(costs.keys(), key=len)
    longest_value = max(costs.values(), key=lambda x: len(format_currency(x)))
    for model, cost in costs.items():
        print(f"{model:<21}{format_currency(cost):>20}")


def self_hosted_price(
    instance_type: str,
    gpu_profiles: Dict[str, pd.DataFrame],
    config: Config,
    model_hosting_costs: List[HostedGPUInfo],
):
    input_utilization = input_server_utilization(gpu_profiles, config)
    output_utilization = output_server_utilization(gpu_profiles, config)
    total_utilization = input_utilization + output_utilization
    servers_required = math.ceil(total_utilization)

    result = {}
    for gpu_info in model_hosting_costs:
        if gpu_info.instance_type == instance_type:
            gpu_cost = gpu_info.cost_per_hour * 24 * 30 * servers_required

            result[f"{gpu_info.instance_type}-{gpu_info.contract_type}"] = gpu_cost
    return result
