# import logging

# default_logger = logging.getLogger()
# default_logger.error("test")

import locale
from typing import List
import math

import streamlit as st
import pandas as pd

from pricing_calculator.datasets import llm_as_a_service
from pricing_calculator.datasets import model_profiles as model_profile_utils
from pricing_calculator.datasets import hosting_costs

from pricing_calculator.config import Config
from pricing_calculator.pricing import (
    monthly_cost_for_llm_as_a_service,
    self_hosted_price,
    sprint_price,
)
from pricing_calculator.self_hosted import (
    ideal_batch_size,
    data_limited_batch_size,
    batch_latency,
    output_server_utilization,
    input_server_utilization,
)


# st._logger.init_tornado_logs()
# st._logger.get_logger(f"tornado.access").setLevel(logging.DEBUG)
# st._logger.get_logger(f"tornado.application").setLevel(logging.DEBUG)
# st._logger.get_logger(f"tornado.general").setLevel(logging.DEBUG)

# st._logger.get_logger(f"tornado.access").error("starting process")


def normalize_name(name: str) -> str:
    # return name.replace("-", " ").replace("_", " ").capitalize()
    return name


def type_1_models(fine_tuned: bool) -> set[str]:
    try:
        type_1_models_info = llm_as_a_service.load()
    except Exception as e:
        breakpoint()

    return {
        type_1_model.model
        for type_1_model in type_1_models_info
        if not fine_tuned or type_1_model.is_fine_tuneable
    }


def self_hosted_models() -> set[str]:
    return model_profile_utils.get_model_names()


def available_models(fine_tuned: bool) -> List[str]:
    return sorted(
        list(
            {
                normalize_name(name)
                for name in type_1_models(fine_tuned).union(self_hosted_models())
            }
        )
    )


def estimate_costs(
    data_residence, input_tokens, output_tokens, requests_per_minute, fine_tuned, models
):
    config = Config(
        **{
            "data-residence": data_residence,
            "input-tokens": input_tokens,
            "output-tokens": output_tokens,
            "requests-per-minute": requests_per_minute,
            "fine-tuned": fine_tuned,
            "models": models,
        }
    )
    type1_pricing = llm_as_a_service.load()
    model_hosting_costs = hosting_costs.load()
    monthly_costs = monthly_cost_for_llm_as_a_service(config, type1_pricing)

    monthly_costs_columnar = [
        {
            "Model": key,
            "Instance": "n/a",
            "Contract": "On Demand",
            "Server Count": "n/a",
            "Batch Size": "n/a",
            "Model Latency": value["latency"],
            "Surge Capacity": "unknown",
            "Monthly Cost": sprint_price(value["total_cost"]),
            "Input Processing Cost": sprint_price(value["input_cost"]),
            "Output Processing Cost": sprint_price(value["output_cost"]),
        }
        for key, value in monthly_costs.items()
    ]
    costs = pd.DataFrame(monthly_costs_columnar)

    for model in models:
        model_profiles = model_profile_utils.load(model)
        for instance_type, gpu_profiles in model_profiles.items():
            print(f"Instance: {instance_type}")
            batch_size = ideal_batch_size(gpu_profiles["input"], config.input_tokens)
            real_batch_size = data_limited_batch_size(gpu_profiles, config)
            print(f"Ideal batch size: {batch_size}")
            print(f"Data limited batch: {real_batch_size}")
            latency = batch_latency(gpu_profiles, config, real_batch_size)
            # print model latency to 3 decimal places
            print(f"Model Latency: {latency:.3f}s")
            input_utilization = input_server_utilization(gpu_profiles, config)
            output_utilization = output_server_utilization(gpu_profiles, config)
            total_utilization = input_utilization + output_utilization
            print(f"Input Server Utilization: {input_utilization}")
            print(f"Output Server Utilization: {output_utilization}")
            print(f"Total Server Utilization: {total_utilization}")
            servers_required = math.ceil(total_utilization)
            reserve_compute = (
                (servers_required - total_utilization) * 100 / total_utilization
            )
            print(f"Servers Required: {servers_required}")
            print(f"Surge Capacity: {reserve_compute:.2f}% increase in traffic")
            pricing = self_hosted_price(
                instance_type, gpu_profiles, config, model_hosting_costs
            )
            columnar_pricing = [
                {
                    "Model": model,
                    "Instance": key.split("-")[0],
                    "Contract": key.split("-")[1],
                    "Server Count": servers_required,
                    "Batch Size": real_batch_size,
                    "Model Latency": f"{latency:.3f}s",
                    "Surge Capacity": f"{reserve_compute:.2f}%",
                    "Monthly Cost": sprint_price(value),
                    "Input Processing Cost": sprint_price(
                        value * input_utilization / total_utilization
                    ),
                    "Output Processing Cost": sprint_price(
                        value * output_utilization / total_utilization
                    ),
                }
                for key, value in pricing.items()
            ]
            df_pricing = pd.DataFrame(columnar_pricing)
            costs = pd.concat([costs, df_pricing])

    min_c = None
    min_idx = None

    for idx, c in enumerate(costs["Monthly Cost"]):
        c_float = locale.atof(c.strip("$").replace(",", ""))

        if min_c is None or c_float < min_c:
            min_c = c_float
            min_idx = idx

    return costs.reset_index().style.map(
        lambda _: "color: black; background-color: yellow",
        subset=pd.IndexSlice[min_idx, "Monthly Cost"],
    )


def calculate_request_per_minute(request_rate, request_rate_timescale):
    if request_rate_timescale == "Minute":
        return request_rate
    elif request_rate_timescale == "Second":
        return request_rate * 60
    elif request_rate_timescale == "Hour":
        return request_rate / 60
    elif request_rate_timescale == "Day":
        return request_rate / (24 * 60)
    elif request_rate_timescale == "Week":
        return request_rate / (7 * 24 * 60)
    elif request_rate_timescale == "Month":
        return request_rate / (30 * 24 * 60)
    else:
        raise Exception("Invalid request rate timescale {request_rate_timescale}}")


def get_output_token_limit(model, input_token_size):
    if model.output_token_limit is None:
        return model.max_tokens - input_token_size
    else:
        return min(model.output_token_limit, model.max_tokens - input_token_size)


def state_from_query_param(key: str, multiple=False, default=None, transformation=None):
    if not transformation:
        transformation = lambda x: x

    if key not in st.session_state:
        if multiple:
            st.session_state[key] = transformation(
                st.query_params.get_all(key) or default if default is not None else []
            )
        else:
            st.session_state[key] = transformation(st.query_params.get(key, default))

    def set_query_param(value):
        st.query_params[key] = value

    return key, set_query_param


st.set_page_config(
    page_title="LLM Pricing Calculator",
    page_icon=":moneybag:",
    layout="wide",
)


# max-width: Content will fill width on small screens up to this maximum; this also seems to fix an issue with tables scrolling horizontally when they don't need to
# padding-top: Just a little bit closer to the top
# padding-bottom: No need for so much extra space along the bottom; this usually leads to scrolling when we don't need it
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1260px;
        padding-top: 3rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
calc_tab, data_tab, info_tab = st.tabs(["Cost Calculator", "Underlying Data", "Info"])

with calc_tab:
    with st.container():
        st.write("##### Select Models")

        fine_tuned_key, set_fine_tuned_query_param = state_from_query_param(
            "fine_tuned", default=False, transformation=lambda x: x == "True"
        )
        fine_tuned = st.checkbox("Fine Tuned", key=fine_tuned_key)
        set_fine_tuned_query_param(fine_tuned)

        model_key, set_model_query_param = state_from_query_param(
            "model", multiple=True, default=available_models(fine_tuned)[:1]
        )
        models = st.multiselect(
            "Models",
            available_models(fine_tuned),
            available_models(fine_tuned)[:1],
            key=model_key,
        )
        set_model_query_param(models)

    if len(models) > 0:
        with st.container():
            st.write("##### Set Tokens")
            max_input_tokens = min(
                [
                    model.max_tokens
                    for model in llm_as_a_service.load()
                    if model.model in models
                ]
            )
            input_tokens_key, set_input_tokens_query_param = state_from_query_param(
                "input_tokens", default=5, transformation=int
            )
            input_tokens = st.slider(
                "Input Tokens",
                min_value=5,
                max_value=max_input_tokens,
                key=input_tokens_key,
            )
            set_input_tokens_query_param(input_tokens)

            max_output_tokens = min(
                [
                    (get_output_token_limit(model, input_tokens), model.model)
                    for model in llm_as_a_service.load()
                    if model.model in models
                ]
            )
            output_tokens_key, set_output_tokens_query_param = state_from_query_param(
                "output_tokens", default=5, transformation=int
            )
            limiting_msg = ""
            if len(models) > 1:
                limiting_msg = f" (Limited by the smallest output token limit: {max_output_tokens[1]})"
            output_tokens = st.slider(
                f"Output Tokens{limiting_msg}",
                min_value=5,
                max_value=max_output_tokens[0],
                key=output_tokens_key,
            )
            set_output_tokens_query_param(output_tokens)

        with st.container():
            st.write("##### Set Request Rate")
            with st.expander("Advanced Request-Rate Slider"):
                (
                    lower_bound_request_rate_key,
                    set_lower_bound_request_rate_query_param,
                ) = state_from_query_param(
                    "lower_bound_request_rate", default=1, transformation=int
                )
                lower_bound_request_rate = st.number_input(
                    "Lower Bound Request Rate",
                    min_value=1,
                    step=1,
                    key=lower_bound_request_rate_key,
                )
                set_lower_bound_request_rate_query_param(lower_bound_request_rate)

                (
                    upper_bound_request_rate_key,
                    set_upper_bound_request_rate_query_param,
                ) = state_from_query_param(
                    "upper_bound_request_rate",
                    default=max(100, lower_bound_request_rate + 1),
                    transformation=int,
                )
                st.session_state[upper_bound_request_rate_key] = max(
                    st.session_state[upper_bound_request_rate_key],
                    lower_bound_request_rate + 1,
                )
                upper_bound_request_rate = st.number_input(
                    "Upper Bound Request Rate",
                    min_value=lower_bound_request_rate,
                    step=1,
                    key=upper_bound_request_rate_key,
                )
                set_upper_bound_request_rate_query_param(upper_bound_request_rate)

                request_rate_timescale_key, set_request_rate_timescale_query_param = (
                    state_from_query_param("request_rate_timescale", default="Minute")
                )
                request_rate_timescale = st.selectbox(
                    "Request Rate Timescale",
                    ["Minute", "Second", "Hour", "Day", "Week", "Month"],
                    key=request_rate_timescale_key,
                )
                set_request_rate_timescale_query_param(request_rate_timescale)

            request_rate_key, set_request_rate_query_param = state_from_query_param(
                "request_rate", default=lower_bound_request_rate, transformation=int
            )
            st.session_state[request_rate_key] = min(
                st.session_state[request_rate_key], upper_bound_request_rate
            )
            st.session_state[request_rate_key] = max(
                st.session_state[request_rate_key], lower_bound_request_rate
            )
            request_rate = st.slider(
                f"Requests Per {request_rate_timescale}",
                min_value=lower_bound_request_rate,
                max_value=upper_bound_request_rate,
                step=1,
                key=request_rate_key,
            )
            set_request_rate_query_param(request_rate)

        with st.container():
            st.write("##### Results")
            requests_per_minute = calculate_request_per_minute(
                request_rate, request_rate_timescale
            )
            st.dataframe(
                estimate_costs(
                    "us",
                    input_tokens,
                    output_tokens,
                    requests_per_minute,
                    fine_tuned,
                    models,
                ),
                use_container_width=True,
                column_config={"index": None},
            )

with data_tab:
    st.write("### LLM-as-a-Service Costs")
    st.dataframe(pd.DataFrame([llm.model_dump() for llm in llm_as_a_service.load()]))

    st.write("### Self Hosting Costs")
    st.dataframe(
        pd.DataFrame([hosting.model_dump() for hosting in hosting_costs.load()])
    )

    st.write("### Model Profiles")
    for model in model_profile_utils.get_model_names():
        for gpu_name, profile in model_profile_utils.load(model).items():
            st.write(f"#### {model} - {gpu_name}")
            st.dataframe(profile["input"])

with info_tab:
    st.write("### Limitations")
    st.write(
        "API Pricing does not currently take into account API Rate limits.  You will be able to enter rates that are not currently supported by the LLM-as-a-Service APIs."
    )
    st.write("### API Pricing Formula")
    st.write(
        r"""
            $$Monthly Price = [(\frac{InputToken}{min} \frac{USD}{InputToken}) + (\frac{OutputToken}{min} \frac{USD}{OutputToken})] * \frac{Minutes}{Month} \frac{Requests}{Min} = \frac{USD}{Month}$$
            """
    )
    st.write("### Self Hosted Algorithm")
    st.write(
        "1. Find the ideal batch size for the given model and input size. (Sampled from real profiling of the model)"
    )
    st.write(
        "Note: If there is not enough data througphut to meet the ideal batch-size, use the data-limited batch size instead."
    )
    st.write(
        "2. Calculate the throughput capacity of a single server for the given model."
    )
    st.write("  a. Determine the input throughput.")
    st.write(
        r"""
            $$Input Capacity = \frac{TokensInNearestKnownBatch}{LatencyOfNearestKnownBatch} / Server = \frac{Token}{Sec}/Server$$ 
            """
    )
    st.write(
        "  b. Determine the output throughput. Each generation cycle takes the same amount of time regardless of batch size, so we only need to calculate this per-request."
    )
    st.write(
        r"""
            $$Output Capacity = \frac{1}{\frac{Sec}{Token_{gen}}} / Server = \frac{Token}{Sec}/Server$$
            """
    )
    st.write(
        "3. Calculate throughput requirement for the data load for input and output tokens."
    )
    st.write(
        r"""
            $$Input Requirement = \frac{InputToken}{request} * \frac{request}{min} * \frac{min}{sec} = \frac{Token}{Sec}$$
            """
    )
    st.write(
        r"""
            $$Output Requirement = \frac{OuputToken}{request} * \frac{request}{min} * \frac{min}{sec} = \frac{Token}{Sec}$$
            """
    )
    st.write(
        "4. Calculate the number of servers required to satisfy the throughput requirement."
    )
    st.write(
        r"""
            $$Servers Required = Ceil(\frac{InputRequirement}{InputCapacity} + \frac{OutputRequirement}{OutputCapacity})$$
            """
    )

# st._logger.get_logger(f"tornado.access").error("ending process")
