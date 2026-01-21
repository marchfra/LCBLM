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
# # Perplexity evaluation
#
# <!-- Starting from sst2 I need to:
# 1. tokenize the sentences with Mistral's tokenizer
# 2. pass the tokenized sentences through Mistral to get the embeddings at the last layer
#     - in the future, get the embeddings from various layers
#     - train a classifier to predict the token ids from the embeddings as a sanity check
# 3. train a SAE to map from the embedding of the last layer to concept space
#     - in the future, train a SAE to map from various layers to concept space
# 4. extract the concept representations for each token in the sentence
# 5. train a linear classifier to predict the *next* token id from the concept representation of the *current* token
#
# The focus of this notebook is step 2.2. -->

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
    raise SystemExit("This notebook is intended to run on Kaggle only.")

# %% [markdown]
# ### Import libraries

# %%
import gc
import json
import random
from pathlib import Path

import torch
from better_kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from sae_utils import SparseAE, TopK
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from tqdm.auto import trange
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

import evaluate

torch.manual_seed(3742)
random.seed(3742)


# %% [markdown]
# ### Free GPU memory function

# %%
def free_gpu_memory(models: list[str] | None = None) -> None:
    """Free GPU memory by deleting model references and emptying the cache."""
    if models is not None:
        for model in models:
            if model in globals():
                del globals()[model]

    if not torch.cuda.is_available():
        print("CUDA is not available. No GPU memory to free.")
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    if allocated == 0 and reserved == 0:
        print("GPU memory is already free.")
        return

    print(
        f"GPU memory before cleanup: "
        f"allocated: {allocated:.2f} GiB | reserved: {reserved:.2f} GiB",
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(
        f"GPU memory after cleanup:  "
        f"allocated: {allocated:.2f} GiB | reserved: {reserved:.2f} GiB",
    )


# %% [markdown]
# ## 1 - Load backbone LLM and tokenizer

# %% [markdown]
# ### 1.1 - Load tokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7b-v0.1")
tokenizer.pad_token = tokenizer.eos_token
EOS_TOKEN_ID = tokenizer.eos_token_id

# %% [markdown]
# ### 1.2 - Load backbone LLM

# %%
user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")

login(token=HF_TOKEN)

# float16 is due to memory constraints on Kaggle (not really, but with float32 it's a
# bit slower)
pre_lm = AutoModel.from_pretrained(
    "mistralai/Mistral-7b-v0.1",
    torch_dtype=torch.float16,
    device_map="auto",
)
pre_lm.eval()

# %%
EMBEDDING_SIZE = pre_lm.config.hidden_size
VOCAB_SIZE = pre_lm.config.vocab_size


# %% [markdown]
# ### 1.3 - Define generation functions

# %%
def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = float("-inf"),
) -> Tensor:
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][:, -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0][indices_to_remove] = filter_value
    return logits


def generate(  # noqa: PLR0913
    input_ids: Tensor,
    pre_lm: nn.Module,
    classifier_head: nn.Module,
    max_length: int = 100,
    temp: float = 0.7,
    topk: int = 100,
    topp: float = 0.9,
    repetition_penalty: float = 1.5,
    eos_token_id: int = EOS_TOKEN_ID,
) -> Tensor:
    past_key_values = None
    for _i in range(max_length):
        lm_embeddings = pre_lm(
            input_ids[:, -1:] if past_key_values is not None else input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = lm_embeddings.past_key_values
        features = lm_embeddings.last_hidden_state.float()
        logits = classifier_head(features)
        score = logits[:, -1, input_ids[0]]
        score = torch.where(
            score < 0,
            score * repetition_penalty,
            score / repetition_penalty,
        )  # ? What is this?
        logits[:, -1, input_ids[0]] = score
        next_token_logits = logits[:, -1, :] / temp
        filtered_logits = top_k_top_p_filtering(
            next_token_logits,
            top_k=topk,
            top_p=topp,
        )
        next_token = torch.multinomial(
            F.softmax(filtered_logits, dim=-1),
            num_samples=1,
        )
        input_ids = torch.cat((input_ids, next_token), dim=-1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return input_ids


def text_generation(  # noqa: PLR0913
    tokenizer,  # noqa: ANN001
    pre_lm: nn.Module,
    classifier_head: nn.Module,
    num_samples: int = 3,
    input_text: str = "",
    output_file: Path | str | None = None,
) -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    generated_texts = []
    with torch.inference_mode():
        for _ in trange(num_samples, desc="Generating texts", unit="sample"):
            output_ids = generate(
                input_ids=input_ids,
                pre_lm=pre_lm,
                classifier_head=classifier_head,
                eos_token_id=EOS_TOKEN_ID,
            )
            generated_text = tokenizer.decode(output_ids[0])
            generated_texts.append(generated_text)

    print("Generated Texts (sample):")
    for i, gen_text in enumerate(generated_texts[:3], 1):
        print(f"\n=== Sample {i} ===\n{gen_text}")
    print()

    if output_file is not None:
        output_file = Path(output_file)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(generated_texts, f, indent=4)
        print(f"Generated texts saved to: {output_file}")

    return generated_texts


# %% [markdown]
# ## 2 - Load Finetuned Classifier

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

finetuned_clf = nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE, bias=False).to(device)
finetuned_clf.load_state_dict(
    torch.load(
        "/kaggle/input/mistral-finetuned-head/pytorch/default/1/next_token_classifier.pt",
    ),
)
finetuned_clf.eval()
print(finetuned_clf)
print(f"Finetuned classifier device: {next(finetuned_clf.parameters()).device}")

# %% [markdown]
# ### 2.1 - Generate text open-endedly with the backbone LLM + Finetuned Classifier

# %%
generated_texts = text_generation(
    tokenizer=tokenizer,
    pre_lm=pre_lm,
    classifier_head=finetuned_clf,
    num_samples=100,
    input_text="",
    output_file="generated_texts_finetuned.json",
)
free_gpu_memory(models=["finetuned_clf"])

# %% [markdown]
# ## 3 - Load Baseline Classifier

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

baseline_clf = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE),
).to(device)
baseline_clf.load_state_dict(
    torch.load("/kaggle/input/next-token-clf-baseline/next_token_classifier.pt"),
)
baseline_clf.eval()
print(baseline_clf)
print(f"Baseline classifier device: {next(baseline_clf.parameters()).device}")

# %% [markdown]
# ### 3.1 - Generate text open-endedly with the backbone LLM + Baseline Classifier

# %%
generated_texts = text_generation(
    tokenizer=tokenizer,
    pre_lm=pre_lm,
    classifier_head=baseline_clf,
    num_samples=100,
    input_text="",
    output_file="generated_texts_baseline.json",
)
free_gpu_memory(models=["baseline_clf"])

# %% [markdown]
# ## 4 - Load Sparse Autoencoder + Classifier

# %%
NUM_CONCEPTS = 16384
SAE_PATH = Path("/kaggle/input/sae-training-on-mistral-embeddings")

sae = SparseAE(
    input_dim=EMBEDDING_SIZE,
    latent_dim_factor=4,
    activation=TopK(k=128),
)
sae.load_state_dict(torch.load(SAE_PATH / "sae_on_mistral_sst2-200epochs.pt"))

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

_sae_clf = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(NUM_CONCEPTS, VOCAB_SIZE),
).to(device)
_sae_clf.load_state_dict(
    torch.load("/kaggle/input/next-token-classifier-torch/next_token_classifier.pt"),
)
_sae_clf.eval()
print(_sae_clf)
print(f"SAE classifier device: {next(_sae_clf.parameters()).device}")


# %%
class ConceptClassifier(nn.Module):
    def __init__(self, sae: SparseAE, clf: nn.Module) -> None:
        super().__init__()
        self.sae = sae
        self.clf = clf

    def forward(self, x: Tensor) -> Tensor:
        concepts = self.sae(x).latents
        logits = self.clf(concepts)
        return logits


sae_clf = ConceptClassifier(sae, _sae_clf).to(device)
print(sae_clf)

# %% [markdown]
# ### 4.1 - Generate text open-endedly with the backbone LLM + SAE + Classifier

# %%
generated_texts = text_generation(
    tokenizer=tokenizer,
    pre_lm=pre_lm,
    classifier_head=sae_clf,
    num_samples=100,
    input_text="",
    output_file="generated_texts_sae.json",
)
free_gpu_memory(models=["sae", "_sae_clf", "sae_clf"])

# %% [markdown]
# ## 5 - Load original Mistral classifier head

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

original_clf = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
).lm_head.to(device)
original_clf.eval()
print(original_clf)
print(f"Original classifier device: {next(original_clf.parameters()).device}")

# %% [markdown]
# ### 5.1 - Generate text open-endedly with the unmodified backbone LLM

# %%
generated_texts = text_generation(
    tokenizer=tokenizer,
    pre_lm=pre_lm,
    classifier_head=original_clf,
    num_samples=100,
    input_text="",
    output_file="generated_texts_original.json",
)
free_gpu_memory(models=["original_clf"])

# %% [markdown]
# ## 6 - Load Sparse Autoencoder + Original head

# %%
NUM_CONCEPTS = 16384
SAE_PATH = Path("/kaggle/input/sae-training-on-mistral-embeddings")

sae = SparseAE(
    input_dim=EMBEDDING_SIZE,
    latent_dim_factor=4,
    activation=TopK(k=128),
)
sae.load_state_dict(torch.load(SAE_PATH / "sae_on_mistral_sst2-200epochs.pt"))

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

original_clf = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
).lm_head.to(device)
original_clf.eval()
print(original_clf)
print(f"Original classifier device: {next(original_clf.parameters()).device}")


# %%
class ReconClassifier(nn.Module):
    def __init__(self, sae: SparseAE, clf: nn.Module) -> None:
        super().__init__()
        self.sae = sae
        self.clf = clf

    def forward(self, x: Tensor) -> Tensor:
        concepts = self.sae(x).recon
        logits = self.clf(concepts)
        return logits


sae_clf = ReconClassifier(sae, original_clf).to(device)
print(sae_clf)

# %% [markdown]
# ### 6.1 - Generate text open-endedly with the backbone LLM + SAE + Original Classifier

# %%
generated_texts = text_generation(
    tokenizer=tokenizer,
    pre_lm=pre_lm,
    classifier_head=sae_clf,
    num_samples=100,
    input_text="",
    output_file="generated_texts_sae_og_head.json",
)
free_gpu_memory(models=["sae", "original_clf", "sae_clf"])

# %% [markdown]
# ## 7 - Evaluate perplexity using huggingface evaluate library

# %%
free_gpu_memory(models=["pre_lm", "tokenizer"])

ppl_lm_name = "Qwen/Qwen3-8B"
ppl_lm = AutoModelForCausalLM.from_pretrained(
    ppl_lm_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
ppl_lm.eval()

tokenizer = AutoTokenizer.from_pretrained(ppl_lm_name)
tokenizer.pad_token = tokenizer.eos_token

# %%
if not tokenizer.bos_token:
    tokenizer.add_special_tokens({"bos_token": "<s>"})

# %%
ppls: dict[str, float] = {}
for file in Path().glob("generated_texts_*.json"):
    print(f"Found generated texts file: {file}")

    with file.open("r") as f:
        preds: list[str] = json.load(f)

    perplexity = evaluate.load(
        "/kaggle/input/perplexity-with-depencendy-injection/perplexity.py",
        module_type="metric",
    )
    perplexity.add_batch(predictions=preds)
    ppl = perplexity.compute(
        model=ppl_lm,
        tokenizer=tokenizer,
        max_length=100,
    )
    if ppl is not None:
        ppls[file.name] = ppl["mean_perplexity"]

# %%
# Create a table with perplexity values
print("\n" + "=" * 43)
print(f"{'Method':<29} | {'Perplexity':>11}")
print("=" * 43)

# Map file names to method names
method_mapping = {
    "generated_texts_finetuned.json": "Finetuned head",
    "generated_texts_original.json": "Original head",
    "generated_texts_retrained_head.json": "Retrained head",
    "generated_texts_sae_new_head.json": "SAE + new head",
    "generated_texts_sae_og_head.json": "SAE + original head",
}

for file_name, ppl_value in sorted(ppls.items(), key=lambda x: x[1]):
    method_name = method_mapping.get(file_name, file_name)
    print(f"{method_name:<29} | {ppl_value:>11.2f}")

with Path(f"perplexities_{ppl_lm_name.split('/')[-1]}.json").open("w") as f:
    json.dump(ppls, f, indent=4)

# Add the reference values
print(f"{'Llama LoRA finetuned on SST2':<29} | {84.70:>11.2f}*")
print(f"{'CB-LLM on SST2':<29} | {116.22:>11.2f}*")
print("=" * 43)
print("* Reported values from CB-LLM paper")
