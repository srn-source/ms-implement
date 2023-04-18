from data import (
    BaseProcessor,
    SST2Processor
)

from model import (
    GPT2Wrapper
)

PROCESSORS = {
    "sst2": SST2Processor,
}

MODELS = {"gpt2": GPT2Wrapper}
