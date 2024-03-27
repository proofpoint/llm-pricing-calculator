import os

from pydantic import (
    BaseModel,
    Field,
)

from pricing_calculator.datasets.common import (
    load_resource,
    StrippedString,
    CostNumber,
    YesNoBool,
    Region,
    OptionalInt,
)

API_DISCOUNT_RATE = os.getenv("API_DISCOUNT_RATE", 0.0)


class SAASModelInfo(BaseModel):
    model: StrippedString = Field(alias="Model")
    provider: StrippedString = Field(alias="Provider")
    producer: StrippedString = Field(alias="Producer")
    model_id: StrippedString = Field(alias="Model ID")
    input_cost_per_1k_tokens: CostNumber = Field(alias="Cost/1k Input Token")
    output_cost_per_1k_tokens: CostNumber = Field(alias="Cost/1k Output Token")
    is_fine_tuneable: YesNoBool = Field(alias="Fine Tuneable?")
    region: Region = Field(alias="Region")
    max_tokens: int = Field(alias="Max Tokens")
    output_token_limit: OptionalInt = Field(alias="Max Output Tokens")


def load():
    """Load data from a csv."""
    all_model_info = load_resource("llm-as-a-service", SAASModelInfo)
    for model_info in all_model_info:
        if model_info.provider == "AWS Bedrock":
            model_info.input_cost_per_1k_tokens = (
                model_info.input_cost_per_1k_tokens * (1.0 - API_DISCOUNT_RATE)
            )
            model_info.output_cost_per_1k_tokens = (
                model_info.output_cost_per_1k_tokens * (1.0 - API_DISCOUNT_RATE)
            )
    return all_model_info
