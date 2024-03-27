from typing import Union, Optional
from pathlib import Path
import csv

from pydantic import (
    BaseModel,
    ValidatorFunctionWrapHandler,
    ValidationInfo,
)
from pydantic.functional_validators import WrapValidator
from typing_extensions import Annotated

resource_path = Path("resources")


def get_resource(name):
    resources = []
    for path in resource_path.glob("*"):
        if name in path.name:
            resources.append(path)

    try:
        return sorted(resources, key=lambda x: x.name, reverse=True)[0]
    except IndexError:
        raise FileNotFoundError(
            f"Could not find any resources with {name} in {resource_path}"
        )

def load_resource(resource_name: str, resource_type: BaseModel):
    """Load data from a csv."""
    resource_path = get_resource(resource_name)
    # Read the file and parse as csv
    with resource_path.open("r", encoding="utf-8-sig") as _f:
        reader = csv.DictReader(_f)
        data = []
        for row in reader:
            data.append(resource_type(**row))
        return data


def cost_str_transform(
    cost: str, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
) -> float:
    """Convert a string denoting a cost to a float."""
    cost = cost.strip().replace("$", "").replace(",", "")
    return handler(float(cost))


def parse_yes_no_bool(
    value: Union[str, bool], handler: ValidatorFunctionWrapHandler, info: ValidationInfo
) -> bool:
    """Convert a string denoting a cost to a float."""
    if isinstance(value, bool):
        return handler(value)

    elif not isinstance(value, str):
        raise ValueError(f"Could not parse {type(value)}({value}) as a boolean")

    value = value.strip().lower()
    if value == "yes":
        return handler(True)
    elif value == "no":
        return handler(False)
    else:
        raise ValueError(f"Could not parse {value} as a boolean")


def stripped_str(
    value: str, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
) -> str:
    return handler(value.strip())


def restrict_to_region(
    value: str, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
) -> str:
    valid_regions = ["us", "eu", "ap"]
    value = value.strip().lower()
    if value in valid_regions:
        return handler(value)
    else:
        return handler("unknown")

def optional_int(
        value: str, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
) -> int:
    if len(value) == 0:
        return handler(None)
    else:
        return handler(int(value))

CostNumber = Annotated[float, WrapValidator(cost_str_transform)]
YesNoBool = Annotated[bool, WrapValidator(parse_yes_no_bool)]
StrippedString = Annotated[str, WrapValidator(stripped_str)]
Region = Annotated[str, WrapValidator(restrict_to_region)]
OptionalInt = Annotated[Optional[int], WrapValidator(optional_int)]
