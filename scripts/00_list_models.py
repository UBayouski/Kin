"""List the models Token Factory serves, flagging NVIDIA ones."""

from kin import llm

for model_id in llm.list_models():
    marker = "*" if "nvidia" in model_id.lower() or "nemotron" in model_id.lower() else " "
    print(marker, model_id)
