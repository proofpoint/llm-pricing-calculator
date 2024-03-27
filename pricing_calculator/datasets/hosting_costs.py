from pydantic import BaseModel, Field

from pricing_calculator.datasets.common import load_resource, StrippedString, CostNumber


class HostedGPUInfo(BaseModel):
    instance_type: StrippedString = Field(alias="Instance")
    provider: StrippedString = Field(alias="Provider")
    contract_type: StrippedString = Field(alias="Constract")
    cost_per_hour: CostNumber = Field(alias="Cost Per Hour")
    gpu: StrippedString = Field(alias="GPU")
    gpu_memory: int = Field(alias="GPU Memory")
    gpu_count: int = Field(alias="# GPU")


def load():
    """Load data from a csv."""
    return load_resource("hosting-costs", HostedGPUInfo)
