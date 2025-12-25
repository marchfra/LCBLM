# Learnable Concept-Base Language Model

## Project overview

1. [x] Embedding extraction
   1. [x] Tokenize dataset
   2. [x] Pass tokenized dataset through backbone LLM
   3. [x] Save to disk the embeddings after the last layer
2. [x] Baseline classifiers
   1. [x] Finetune original LLM's head on current dataset
   2. [x] Train linear classifier from scratch on current dataset
3. [x] Sparse AutoEncoder
   1. [x] Train SAE on the backbone LLM's embeddings
   2. [x] Train linear classifier from SAE's latent space to do next-token prediction
   3. [ ] (optional) Train linear classifier from SAE's reconstruction to do
          next-token prediction
4. [ ] Learnable Concept-Base Language Model
   1. [ ] Train LCBLM on the backbone LLM's embeddings
   2. [ ] Train linear classifier from LCBLM's latent space to do next-token prediction
   3. [ ] (optional) Train linear classifier from LCBLM's reconstruction to do
          next-token prediction
5. [ ] Evaluation metrics
   1. [ ] Perplexity: generate sentences open-endedly with all the methods listed
          below, then evaluate the perplexity of a third-party LLM on those
          sentences
      - [x] Backbone LLM + original head
      - [x] Backbone LLM + finetuned head
      - [x] Backbone LLM + new head
      - [x] Backbone LLM + SAE latents + new head
      - [x] Backbone LLM + SAE recon + original head
      - [ ] (optional) Backbone LLM + SAE recon + new head
      - [ ] Backbone LLM + LCBLM latents + new head
      - [ ] Backbone LLM + LCBLM recon + original head
      - [ ] (optional) Backbone LLM + LCBLM recon + new head
   2. [ ] Concept labelling: do SAE and LCBLM latents correspond to
          human-understandable concepts?
   3. [ ] Intervenability: does turning off or amplifying certain latents affect
          the generated sentences in the expected way?
   - [ ] Train all models on at least 3 different seeds

## Next steps

1. Create `config.toml` file where the user picks a dataset, a backbone LLM and
   all other project parameters
2. Use retry pattern on llm concept annotations
3. Generalize to different datasets
4. Generalize to different backbone LLMs
5. Use the embeddings of different layers of the backbone LLM

## Important notes

1. Saving to disk the SAE's latent space in dense format is a cumbersome and
   manual process. Saving it in sparse format makes all downstream tasks
   unfeasibly slow. In order to avoid these problems, don't save the latent space
   to disk, but pass the backbone LLM's embeddings through the SAE whenever
   needed. This adds a small computational overhead on all downstream tasks, but
   it's negligible compared to the other option.
