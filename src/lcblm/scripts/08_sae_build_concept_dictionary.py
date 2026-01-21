# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SAE concept dictionary
#
# <!-- Starting from sst2 I need to:
# 1. tokenize the sentences with Mistral's tokenizer
# 2. pass the tokenized sentences through Mistral to get the embeddings at the last layer
#     - in the future, get the embeddings from various layers
# 3. train a SAE to map from the embedding of the last layer to concept space
#     - in the future, train a SAE to map from various layers to concept space
# 4. extract the concept representations for each token in the sentence
# 5. train a linear classifier to predict the *next* token id from the concept representation of the *current* token
#
# The focus of this notebook is step 4.
#
# Since the train set tensor is too large to fit in memory, we will extract the concept representations in chunks and save them to disk.
#
# This notebook will output `n_chunks` files for each dataset (train, validation, test), which must be manually downloaded after each chunk is processed, as per the instructions in the last cell.
#
# The output files will be named as follows:
# - `f"{extractor.output_dir}/latent_chunks_{chunk_idx:4d}.pt"`
#
# to distinguish the various splits, the files must be manually renamed. -->

# %% [markdown]
# ## 0 - Environment Setup

# %% [markdown]
# #### Check that the notebook is running in Kaggle

# %%
from pathlib import Path

IN_KAGGLE = Path("/kaggle").exists()
if IN_KAGGLE:
    print("Running on Kaggle")
else:
    print("Not running on Kaggle")
    msg = "This notebook is intended to run on Kaggle only."
    raise SystemExit(msg)

# %% [markdown]
# ### 0.1 - Import libraries

# %%
import gc
import json
import os
from itertools import zip_longest
from pathlib import Path
from typing import NamedTuple

import regex
import torch
from sae_utils import SparseAE, TopK
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

torch.manual_seed(3742)

# %% [markdown]
# ## 1 - Load Data and Pretrained SAE

# %%
PRETRAINED_SAE_PATH = (
    Path("/kaggle/input")
    / "sae-on-mistral-embeddings"
    / "pytorch/default/1"
    / "sae_on_mistral_sst2-200epochs.pt"
)

if not PRETRAINED_SAE_PATH.exists():
    msg = "Pretrained SAE model file is missing."
    raise FileNotFoundError(msg)
else:
    print("Found SAE at specified location")

# %%
DATA_PATH = Path("/kaggle/input/sst2-mistral-embeddings")
TRAIN_PATH = DATA_PATH / "extracted_features_train.pt"
VAL_PATH = DATA_PATH / "extracted_features_validation.pt"

if not DATA_PATH.exists():
    msg = f"Data path {DATA_PATH} does not exist."
    raise FileNotFoundError(msg)


if TRAIN_PATH.exists() and VAL_PATH.exists():
    train_data = torch.load(TRAIN_PATH)
    val_data = torch.load(VAL_PATH)
else:
    msg = "Training or validation data files are missing."
    raise FileNotFoundError(msg)


# %%
class Item(NamedTuple):  # noqa: D101
    input_ids: torch.Tensor
    embeddings: torch.Tensor


class EmbeddingsDataset(Dataset[Item]):  # noqa: D101
    def __init__(self, input_ids: torch.Tensor, embeddings: torch.Tensor) -> None:  # noqa: D107
        self.input_ids = input_ids
        self.embeddings = embeddings

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> Item:
        return Item(self.input_ids[idx], self.embeddings[idx])



# %%
train_dataset = EmbeddingsDataset(train_data["input_ids"], train_data["embeddings"])
val_dataset = EmbeddingsDataset(val_data["input_ids"], val_data["embeddings"])

del train_data, val_data
gc.collect()

# %%
BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
LATENT_DIM_FACTOR = 4
TOP_K = 128

_sae = SparseAE(
    input_dim=train_dataset.embeddings.shape[-1],
    latent_dim_factor=LATENT_DIM_FACTOR,
    activation=TopK(k=TOP_K),
)
_sae.load_state_dict(torch.load(PRETRAINED_SAE_PATH))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sae = torch.nn.DataParallel(_sae).to(device)

del _sae
gc.collect()

# %% [markdown]
# ## 2 - Create input concept dictionary
#
# This concept dictionary is built by mapping each latent to the input tokens that activate it the most. If a token appears multiple times in the dataset, we take the maximum activation across all occurrences.

# %%
NUM_EXAMPLES_IN_DICT = 10

latent_dim = sae.module.latent_dim

latent_token_values = [{} for _ in range(latent_dim)]

sae.eval()

with torch.inference_mode():
    for batch in tqdm(
        train_loader,
        desc="Extracting data for concept dictionary",
        unit="batch",
    ):
        input_ids: torch.Tensor = batch.input_ids  # (B, T)
        embeddings: torch.Tensor = batch.embeddings.to(device)  # (B, T, embed_dim)

        # Forward pass -> (B, T, latent_dim)
        latents: torch.Tensor = sae(embeddings).latents.cpu()

        # Compute top-L latents per token: shapes (B*T, L)
        flat_ids = input_ids.reshape(-1)  # (B*T,)
        flat_latents = latents.reshape(-1, latent_dim)  # (B*T, D)

        top_vals, top_idxs = flat_latents.topk(TOP_K, dim=-1)

        # Now update heaps only for those top-L latents
        for token_idx in range(flat_ids.size(0)):
            token_id = flat_ids[token_idx].item()
            vals = top_vals[token_idx].tolist()
            idxs = top_idxs[token_idx].tolist()

            for value, latent_idx in zip(vals, idxs, strict=True):
                token_values = latent_token_values[latent_idx]
                if token_id not in token_values or value > token_values[token_id]:
                    token_values[token_id] = value

heaps = []
for token_values in latent_token_values:
    # Get top NUM_EXAMPLES_IN_DICT tokens by value
    sorted_items = sorted(token_values.items(), key=lambda x: x[1], reverse=True)
    heap = [(val, token_id) for token_id, val in sorted_items[:NUM_EXAMPLES_IN_DICT]]
    heaps.append(heap)

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

# %%
concept_dictionary_input: dict[int, list[tuple[float, int, str]]] = {}

for latent_idx, heap in enumerate(heaps):
    sorted_heap = sorted(heap, key=lambda x: x[0], reverse=True)
    concept_dictionary_input[latent_idx] = [
        (value, input_id, tokenizer.decode(input_id)) for value, input_id in sorted_heap
    ]

with Path("input_concept_dictionary.json").open("w") as f:
    json.dump(concept_dictionary_input, f, indent=4)

# %%
latent = 42

print(f"Top examples for latent {latent}")
print("=" * 40)
print(" Rank | Activation | Input ID | Token")
print("=" * 40)
for rank, (activation, input_id, token) in enumerate(
    concept_dictionary_input[latent],
    start=1,
):
    print(f"{rank:5d} | {activation:10.2f} | {input_id:8} | {token}")

# %% [markdown]
# ## 3 - Create output concept dictionary
#
# This concept dictionary is built by mapping each latent to the output tokens that it is most predictive of, based on the linear classifier weights.

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

# %%
weights = torch.load(
    "/kaggle/input/next-token-classifier-torch/next_token_classifier.pt",
)["1.weight"]
vocab_size: int = weights.size(0)
latent_dim: int = weights.size(1)

# %%
top_weights, top_token_ids = torch.topk(weights.T, k=NUM_EXAMPLES_IN_DICT, dim=-1)
bottom_weights, bottom_token_ids = torch.topk(
    -weights.T,
    k=NUM_EXAMPLES_IN_DICT,
    dim=-1,
)
top_weights.shape, top_token_ids.shape

# %%
concept_dictionary_output_top: dict[int, list[tuple[float, int, str]]] = {}
concept_dictionary_output_bottom: dict[int, list[tuple[float, int, str]]] = {}

for latent_idx in range(latent_dim):
    token_ids = top_token_ids[latent_idx].tolist()
    weights_vals = top_weights[latent_idx].tolist()
    tokens = [tokenizer.decode(token_id) for token_id in token_ids]

    concept_dictionary_output_top[latent_idx] = [
        (weight, token_id, token)
        for weight, token_id, token in zip(weights_vals, token_ids, tokens, strict=True)
    ]

    token_ids = bottom_token_ids[latent_idx].tolist()
    weights_vals = bottom_weights[latent_idx].tolist()
    tokens = [tokenizer.decode(token_id) for token_id in token_ids]

    concept_dictionary_output_bottom[latent_idx] = [
        (-weight, token_id, token)
        for weight, token_id, token in zip(weights_vals, token_ids, tokens, strict=True)
    ]


with Path("output_concept_dictionary_top.json").open("w") as f:
    json.dump(concept_dictionary_output_top, f, indent=4)
with Path("output_concept_dictionary_bottom.json").open("w") as f:
    json.dump(concept_dictionary_output_bottom, f, indent=4)

# %%
latent = 42

print(f"Top (+) examples for latent {latent}")
print("=" * 40)
print(" Rank | Weight | Token ID | Token")
print("=" * 40)
for rank, (weight, token_id, token) in enumerate(
    concept_dictionary_output_top[latent],
    start=1,
):
    print(f"{rank:5d} | {weight:6.3g} | {token_id:8} | {token}")

print(f"\nTop (-) examples for latent {latent}")
print("=" * 40)
print(" Rank | Weight | Token ID | Token")
print("=" * 40)
for rank, (weight, token_id, token) in enumerate(
    concept_dictionary_output_bottom[latent],
    start=1,
):
    print(f"{rank:5d} | {weight:6.3g} | {token_id:8} | {token}")


# %%
def visualize_latent(latent: int) -> None:  # noqa: D103
    if (
        not concept_dictionary_input[latent]
        or not concept_dictionary_output_top[latent]
        or not concept_dictionary_output_bottom[latent]
    ):
        print(f"Latent {latent} is dead.")
        return

    # print(f"Examples for latent {latent}")
    print(
        f" {latent:5d}  ||  {'INPUT':^35}  ||  {'TOP (+) OUTPUT':^35}  ||  {'TOP (-) OUTPUT':^35}",  # noqa: E501
    )
    print("=" * 131)
    print(
        "  Rank  ||  "
        "Activation | Input ID | Input Token  ||  "
        "Weight | Next Token ID | Next Token  ||  "
        "Weight | Next Token ID | Next Token",
    )
    print("=" * 131)
    for rank, (input_dict, output_dict_top, output_dict_bottom) in enumerate(
        zip_longest(
            concept_dictionary_input[latent],
            concept_dictionary_output_top[latent],
            concept_dictionary_output_bottom[latent],
            fillvalue=None,
        ),
        start=1,
    ):
        if input_dict is not None:
            activation, input_id, input_token = input_dict
            input_str = f"{activation:10.2f} | {input_id:8} | {input_token:<11}"
        else:
            input_str = " " * 10 + " | " + " " * 8 + " | " + " " * 11

        if output_dict_top is not None:
            weight, next_token_id, next_token = output_dict_top
            output_str_top = f"{weight:6.4f} | {next_token_id:13} | {next_token:<10}"
        else:
            output_str_top = " " * 6 + " | " + " " * 13 + " | " + " " * 10

        if output_dict_bottom is not None:
            weight, next_token_id, next_token = output_dict_bottom
            output_str_bottom = f"{weight:6.3f} | {next_token_id:13} | {next_token:<10}"
        else:
            output_str_bottom = " " * 6 + " | " + " " * 13 + " | " + " " * 10

        print(
            f"{rank:6d}  ||  {input_str}  ||  {output_str_top}  ||  {output_str_bottom}",  # noqa: E501
        )


visualize_latent(latent=11527)

# %%
longest_token = ""
threshold_length = 10
num_tokens_above_threshold = 0
for token_id in range(tokenizer.vocab_size):
    token = tokenizer.decode(token_id)
    if len(token) >= len(longest_token):
        longest_token = token
        print(
            f"New longest token (length {len(longest_token):2d}): "
            f"ID {token_id:5d} -> '{longest_token}'",
        )
    if len(token) > threshold_length:
        num_tokens_above_threshold += 1

print(f"\nLongest token overall (length {len(longest_token)}): '{longest_token}'")
print(
    f"Number of tokens with length > {threshold_length}: {num_tokens_above_threshold} "
    f"({num_tokens_above_threshold / tokenizer.vocab_size:.2%})",
)

# %%
print(f"Weights mean: {weights.mean().item():.4g}, std: {weights.std().item():.4g}")
print(f"Weights min: {weights.min().item():.4g}, max: {weights.max().item():.4g}")

# %%
formatted_concept_dictionary = {}
for latent in range(latent_dim):
    input_tokens = [token for *_, token in concept_dictionary_input[latent]]
    output_tokens_top = [token for *_, token in concept_dictionary_output_top[latent]]
    output_tokens_bottom = [
        token for *_, token in concept_dictionary_output_bottom[latent]
    ]
    if input_tokens and output_tokens_top and output_tokens_bottom:
        formatted_concept_dictionary[latent] = {
            "input": input_tokens,
            "top_output": output_tokens_top,
            "bottom_output": output_tokens_bottom,
        }

with Path("formatted_concept_dictionary.json").open("w") as f:
    json.dump(formatted_concept_dictionary, f, indent=4)

for latent, tokens in list(formatted_concept_dictionary.items()):
    if tokens["input"] and tokens["top_output"] and tokens["bottom_output"]:
        print(f"Latent {latent}:")
        print(f"  Input tokens: {tokens['input']}")
        print(f"  Top (+) output tokens: {tokens['top_output']}")
        print(f"  Top (-) output tokens: {tokens['bottom_output']}")

# %% [markdown]
# ## 4 - Interpret concept dictionary

# %%
interpret_model_name = "Qwen/Qwen3-8B"
interpret_model = AutoModelForCausalLM.from_pretrained(
    interpret_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
interpret_model.eval()
interpret_tokenizer = AutoTokenizer.from_pretrained(interpret_model.config.name_or_path)

# %%
Interpretation = dict[str, str | list[str]]
ConceptDictionary = dict[int, dict[str, list[str]]]
InterpretedDictionary = dict[int, dict[str, list[str] | Interpretation]]


def extract_last_json(text: str) -> str | None:
    """Extract the last valid JSON object from text using recursive regex."""
    # recursive pattern for matching balanced {...}
    pattern = r"\{(?:[^{}]|(?R))*\}"
    matches = regex.findall(pattern, text)
    return matches[-1] if matches else None


def interpret_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tokens: list[str],
    max_new_tokens: int = 256,
) -> Interpretation:
    """Interpret a list of tokens using a language model.

    Raises:
        ValueError: If no tokens are provided or if no JSON is found in the model
            output.

    """
    prompt = f"""
You are an expert at interpreting groups of related text tokens.

Below is a list of tokens associated with a sparse autoencoder latent, in decreasing order of importance.

Your task:
1. Infer the shared linguistic or semantic pattern.
2. Give a short concept description in English (1-5 words).
3. Optionally list alternative interpretations if relevant.

Tokens:
{", ".join(tokens)}

Provide a JSON dictionary with fields:
- "concept": a short (1-5 word) English description
- "alternatives": a list of 1-3 alternative plausible interpretations

Answer only with valid JSON, with no additional explanations. Use only English.
"""  # noqa: E501
    if len(tokens) == 0:
        msg = "No tokens provided for interpretation."
        raise ValueError(msg)

    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
        )

    raw = interpret_tokenizer.decode(output[0], skip_special_tokens=True)

    json_text = extract_last_json(raw.replace("\n", ""))
    if json_text is None:
        msg = "No JSON found in the model output."
        raise ValueError(msg)

    return json.loads(json_text)


def interpret_concept_dictionary(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    concept_dictionary: ConceptDictionary,
) -> InterpretedDictionary:
    """Interpret a concept dictionary using a language model."""
    interpreted_dictionary: InterpretedDictionary = {}

    for latent_idx, tokens in tqdm(
        concept_dictionary.items(),
        desc="Interpreting latents",
        unit="latent",
    ):
        input_tokens = tokens["input"]
        output_tokens_top = tokens["top_output"]
        output_tokens_bottom = tokens["bottom_output"]

        try:
            input_interpretation = interpret_tokens(model, tokenizer, input_tokens)
        except Exception as e:  # noqa: BLE001
            print(
                f"Skipping latent {latent_idx} due to error in input:  {e}",
            )
            continue

        try:
            top_output_interpretation = interpret_tokens(
                model,
                tokenizer,
                output_tokens_top,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"Skipping latent {latent_idx} due to error in top output: {e}",
            )
            continue

        try:
            bottom_output_interpretation = interpret_tokens(
                model,
                tokenizer,
                output_tokens_bottom,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"Skipping latent {latent_idx} due to error in bottom output: {e}",
            )
            continue

        interpreted_dictionary[latent_idx] = {
            "input_explanation": input_interpretation,
            "input_tokens": input_tokens,
            "top_output_explanation": top_output_interpretation,
            "top_output_tokens": output_tokens_top,
            "bottom_output_explanation": bottom_output_interpretation,
            "bottom_output_tokens": output_tokens_bottom,
        }

    return interpreted_dictionary


# %%
START_LATENT = 0
END_LATENT = 100

save_path = f"interpreted_concept_dictionary_sae_{START_LATENT}_{END_LATENT - 1}.json"

# %%
from itertools import islice

if "formatted_concept_dictionary" not in locals():
    with Path("formatted_concept_dictionary.json").open("r") as f:
        formatted_concept_dictionary = json.load(f)


interpreted_dict = interpret_concept_dictionary(
    interpret_model,
    interpret_tokenizer,
    dict(islice(formatted_concept_dictionary.items(), START_LATENT, END_LATENT)),  # pyright: ignore[reportPossiblyUnboundVariable]
)

with Path(save_path).open("w") as f:
    json.dump(interpreted_dict, f, indent=4)

# %% [markdown]
# ## 5 - Visualize interpreted concept dictionary
#
# This concept dictionary contains one entry for each latent associated with at least one token. Each entry contains an explanation, i.e., a short concept description and possibly a couple of alternative descriptions generated by an LLM, and a list of tokens that are associated with that latent. This is done for both the input and output concept dictionaries.
#
# If a latent index does not appear in the dictionary, it means that no token is associated with it (or possibly that an error occurred during interpretation).
#
# Only the first 100 non-dead latents are interpreted, to reduce the time required to generate the interpretations.

# %%
if "interpreted_dict" not in locals():
    with Path(save_path).open("r") as f:
        interpreted_dict = json.load(f)

for latent in list(interpreted_dict):
    print(f"Latent {latent}:")
    input_exp = interpreted_dict[latent]["input_explanation"]
    top_output_exp = interpreted_dict[latent]["top_output_explanation"]
    bottom_output_exp = interpreted_dict[latent]["bottom_output_explanation"]
    if isinstance(input_exp, dict):
        print(f"  Input concept: {input_exp.get('concept')}", end=" ")
        if alternatives := input_exp.get("alternatives"):
            print(f"({'/'.join(alternatives)})")
        print(f"  Input tokens: {interpreted_dict[latent]['input_tokens']}", end="\n\n")
    if isinstance(top_output_exp, dict):
        print(f"  Top (+) output concept: {top_output_exp.get('concept')}", end=" ")
        if alternatives := top_output_exp.get("alternatives"):
            print(f"({'/'.join(alternatives)})")
        print(
            f"  Top (+) output tokens: {interpreted_dict[latent]['top_output_tokens']}",
            end="\n\n",
        )
    if isinstance(bottom_output_exp, dict):
        print(f"  Top (-) output concept: {bottom_output_exp.get('concept')}", end=" ")
        if alternatives := bottom_output_exp.get("alternatives"):
            print(f"({'/'.join(alternatives)})")
        print(
            f"  Top (-) output tokens: {interpreted_dict[latent]['bottom_output_tokens']}",  # noqa: E501
            end="\n\n",
        )
    print()
