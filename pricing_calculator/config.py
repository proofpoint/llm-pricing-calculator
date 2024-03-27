from typing import List
import yaml
from pydantic import BaseModel, Field

class Config(BaseModel):
    """Config for the data profile of the model's use-case."""

    input_tokens: int = Field(alias="input-tokens")
    output_tokens: int = Field(alias="output-tokens")
    requests_per_min: float = Field(alias="requests-per-minute")
    is_fine_tuned: bool = Field(alias="fine-tuned")
    models: List[str] = Field(alias="models")
    data_residence: str = Field(alias="data-residence")

    @property
    def requests_per_sec(self):
        return self.requests_per_min / 60
    
    @classmethod
    def from_yaml_file(cls, file_path: str):
        with open(file_path, "r") as _f:
            config = yaml.safe_load(_f)
        return cls(**config)