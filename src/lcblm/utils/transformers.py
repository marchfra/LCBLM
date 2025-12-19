import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def get_generic_llm(model_id: str, *, eval_mode: bool = True) -> PreTrainedModel:
    """Instantiate AutoModel from model_id.

    Args:
        model_id: The model id of a pretrained model hosted inside a model repo on
            huggingface.co.
        eval_mode: If True, call `model.eval()` before returning the model.

    """
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if eval_mode:
        model.eval()
    return model


def get_causal_llm(model_id: str, *, eval_mode: bool = True) -> PreTrainedModel:
    """Instantiate AutoModelForCausalLM from model_id.

    Args:
        model_id: The model id of a pretrained model hosted inside a model repo on
            huggingface.co.
        eval_mode: If True, call `model.eval()` before returning the model.

    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if eval_mode:
        model.eval()
    return model


def get_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """Instantiate AutoTokenizer from model_id.

    Add '<s>' as `bos_token` if a `bos_token` isn't already defined by the tokenizer.
    Use a generic special token already defined by the tokenizer to use as `pad_token`,
    if `pad_token` isn't already defined by the tokenizer.

    Args:
        model_id: The model id of a predefined tokenizer hosted inside a model repo on
            huggingface.co.
        eval_mode: If True, call `model.eval()` before returning the model.

    """
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<s>"})

    if tokenizer.pad_token is None:
        existing_special_tokens = list(
            tokenizer.special_tokens_map_extended.values(),
        )
        # check that the model already has at least one special token defined
        if len(existing_special_tokens) == 0:
            msg = (
                "The tokenizer must have at least one special token defined to use "
                "for padding. Please use a different tokenizer."
            )
            raise ValueError(msg)
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    return tokenizer
