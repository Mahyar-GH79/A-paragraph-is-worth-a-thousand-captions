# Caption or Paragraph?

**Long synthetic descriptions, not more short captions, are what fix long-text image retrieval.**

This is the code and data release for our study of *what text you should train a
retrieval model on*. We take 500k CC3M images and, for each, synthesise five
positive captions, five hard negatives, and one descriptive paragraph. We then
fine-tune BLIP's text tower on ten different mixtures of that text and measure
what each mixture buys on four retrieval benchmarks.

The result is one-sided. **Paragraph supervision is the only ingredient that
matters.** Every configuration containing paragraphs reaches ~66% Recall@1 on
DOCCI; every configuration without them lands between 19% and 40%, i.e. *below*
the pretrained baseline. Adding more short positive captions does not help, and
adding synthetic hard negatives is actively destructive.

---

## Headline results

Image-to-text Recall@1. Full tables, both directions and R@5/R@10, are in
[`results/tables/`](results/tables/).

| Model | Training text | DOCCI | ShareGPT4V | Flickr30k | COCO |
|---|---|--:|--:|--:|--:|
| BLIP (pretrained) | — | 49.5 | 42.8 | **55.2** | **41.0** |
| CLIP (pretrained) | — | 43.4 | 31.8 | 40.4 | 19.0 |
| Long-CLIP-B | — | 48.2 | 48.5 | 48.7 | 23.6 |
| Long-CLIP-L | — | 51.6 | 52.6 | 54.2 | 28.3 |
| CFG1 | Original caption | 40.4 | 36.2 | 54.0 | 37.7 |
| CFG2 | Random positive | 18.6 | 27.7 | 53.0 | 37.5 |
| CFG8 | Original + all 5 positives | 30.1 | 36.3 | 53.7 | 36.6 |
| CFG7 | Original + positive + hard negative | 23.0 | 24.6 | 50.8 | 32.3 |
| **CFG5** | **Original caption + paragraph** | **66.1** | 51.7 | 56.6 | 38.4 |
| CFG4 | Paragraph only | 65.8 | 51.8 | 53.0 | 37.3 |
| CFG6 | Random positive + paragraph | 66.1 | **52.0** | 49.3 | 28.7 |

Three things to read off this table:

1. **Paragraphs win on long text by a wide margin.** CFG5 beats pretrained BLIP by
   +16.6 points on DOCCI and beats Long-CLIP-L, a model built specifically for long
   text, by +14.5.
2. **Short captions do not just fail to help — they hurt.** CFG2 (random positive
   captions only) *halves* DOCCI recall relative to the baseline.
3. **There is a real cost on short-caption retrieval.** Pretrained BLIP is still the
   best model on COCO, and no fine-tuned configuration recovers it. Fine-tuning on
   CC3M trades roughly 2–5 points of COCO recall for 16–20 points on DOCCI.

### Ablations

* **More positives make long-text retrieval worse.** Going from 1 to 5 positive
  captions moves Flickr30k I2T from 52.1 to 53.7 but drops DOCCI from 38.4 to 30.2.
* **Hard negatives are monotonically destructive.** 1 → 5 negatives sends ShareGPT4V
  I2T from 11.9 to 2.3. Our GPT-4o audit of the data explains why: the synthetic
  "hard negatives" score 7.98/10 for plausibility with a std of 1.86, meaning many of
  them are not actually wrong descriptions of the image.
* **The benefit saturates early.** 25% of the data already gives 64.9 DOCCI R@1
  against 66.1 for the full 500k.

---

## Installation

```bash
git clone https://github.com/Mahyar-GH79/A-paragraph-is-worth-a-thousand-captions.git
cd A-paragraph-is-worth-a-thousand-captions

python -m venv env && source env/bin/activate
pip install -e .            # add '.[quality]' for the GPT-4o judge, '.[dev]' for tests
```

All paths derive from a single root. Set `CAPARA_ROOT` if you keep datasets and
checkpoints outside the checkout (see [`src/capara/common/paths.py`](src/capara/common/paths.py)):

```bash
export CAPARA_ROOT=/mnt/data/caption_or_paragraph
```

## Data

### The generated CC3M annotations

The core artefact of this project is a 500k-row JSONL file: for every CC3M image,
five positive captions, five hard negatives, and one Qwen2-VL paragraph scored for
faithfulness by Llama-3.2-Vision.

```json
{
  "image_url": "http://...",
  "original_caption": "a very typical bus station",
  "positive_captions": ["A bustling bus station with numerous buses and people.", "..."],
  "hard_negative_captions": ["A bustling train station with numerous passengers.", "..."],
  "paragraph": "The image depicts a bustling bus station, filled with a variety of buses and people. ...",
  "llama_score": 8
}
```

It expands a 10.5-word average caption into a 92.9-word average paragraph.

A 200-row preview is committed at
[`data/sample_cc3m_annotations.jsonl`](data/sample_cc3m_annotations.jsonl). **The full
720 MB file is hosted on the Hugging Face Hub** (too large for git):

```bash
huggingface-cli download <your-hf-username>/cc3m-caption-or-paragraph \
    --repo-type dataset --local-dir cc3m_qwen_llama_outputs
```

To regenerate it from scratch instead (needs a GPU and many hours):

```bash
python -m capara.data.generate_annotations --max-samples 500000
```

### Benchmarks

```bash
python -m capara.data.download_benchmarks       # COCO 2017
python -m capara.data.download_llava_pretrain   # LLaVA-Pretrain images, for ShareGPT4V
```

Flickr30k and DOCCI must be obtained from their own sources and placed under
`datasets/flickr30k` and `datasets/docci`.

## Reproducing the paper

```bash
# 1. Embed the CC3M images with the frozen BLIP vision tower (768-d), then project
#    into BLIP's 256-d contrastive space. Training reads only the 256-d shards.
python -m capara.data.build_image_embeddings
python -m capara.data.project_embeddings

# 2. Train. The vision tower stays frozen; only text_encoder and text_proj move.
python -m capara.train --config cfg5

# 3. Evaluate on all four benchmarks.
python -m capara.evaluate --model blip-finetuned \
    --checkpoint blip_text_train/<run>/final_model.pt \
    --datasets flickr30k coco docci sharegpt4v

# 4. Baselines and figures.
python -m capara.evaluate --model blip --datasets docci
python -m capara.eval_longclip --datasets docci
python -m capara.analysis.results_tables
python -m capara.analysis.pareto
```

The ablations are standalone:

```bash
python -m capara.ablations.pos_neg_scaling
python -m capara.ablations.data_efficiency
```

## Repository layout

```
src/capara/
  configs.py          The ten training configurations (cfg1..cfg10)
  train.py            Text-tower fine-tuning, all configs
  evaluate.py         Retrieval eval: BLIP / CLIP / fine-tuned, 4 benchmarks
  eval_longclip.py    Long-CLIP-B/L baselines
  common/             Shared model, losses, metrics, shard streaming, paths
  data/               Annotation generation, image embedding, downloads
  ablations/          Positive/negative scaling, data efficiency
  analysis/           Every paper figure and LaTeX table
results/              Committed: eval JSON, LaTeX tables, figures
tests/                Numerical-equivalence tests for the losses and metrics
```

A config is defined purely by *which texts it pairs with each image*, so all ten
share one training loop:

```python
"cfg5": _config(
    "cfg5_original_plus_paragraph",
    "Original caption + paragraph",
    train_sources=[ORIGINAL, PARAGRAPH],
    val_sources=[ORIGINAL, PARAGRAPH],
    requires=[Requirement.ORIGINAL, Requirement.PARAGRAPH],
)
```

## Tests

```bash
pytest
```

The losses and metrics in this repository were merged from ten near-duplicate
training scripts. `tests/test_equivalence.py` re-implements the original
formulations verbatim and asserts the merged versions are numerically identical, so
the unified trainer provably reproduces the published runs.

To exercise the whole training loop without the real 3 GB shards:

```bash
python tests/make_fixture_shards.py --out-dir /tmp/fixture --shards 4 --rows-per-shard 32
python -m capara.train --config cfg5 --device cpu --shards-dir /tmp/fixture \
    --epochs 1 --batch-size 4 --max-steps-per-epoch 2 --num-workers 0 --val-shards 2
```

## Known issues and caveats

These are honest limitations of the study; they are documented here rather than
quietly fixed, because the published numbers depend on them.

* **ShareGPT4V is evaluated with a 77-token limit.** Its descriptions are far longer
  than that, so the ShareGPT4V column understates the paragraph models. Our own
  truncation study (`results/figures/fig_text_truncation_sharegpt4v.pdf`) shows CFG5
  reaching 87.7 I2T R@1 at 128 tokens versus 51.7 at 77. `--max-text-length` defaults
  reproduce the paper; raise it to 128 for the fairer comparison.
* **COCO is evaluated on `train2017`, not the Karpathy test split**, and
  `blip-itm-base-coco` was itself fine-tuned on COCO. The COCO column is internally
  consistent across our configs but is not comparable to published COCO numbers.
* **Per-epoch caption resampling never took effect in the published runs.** The
  dataloader used persistent workers, which hold a private copy of the dataset, so
  `set_epoch` never reached them and the randomly-sampled configs (cfg2, cfg3, cfg6,
  cfg7, cfg10) drew the *same* caption every epoch. This is preserved by default for
  reproducibility; pass `--no-persistent-workers` to make resampling actually happen.
* **Step counts are computed before filtering.** Configs that require a paragraph drop
  records that lack one, so they run slightly fewer steps than the cosine schedule
  plans and their learning rate does not fully anneal.
* **The original scripts encoded DOCCI in fp32 and every other benchmark in fp16.**
  That was an accident of six separate scripts, not a decision, so `evaluate.py`
  exposes a single `--fp16/--no-fp16` flag instead of reproducing the per-dataset
  precision matrix. `--no-fp16` reproduces the original DOCCI numbers exactly; the
  drift is far below the two decimals the paper reports.
* **In-domain validation recall saturates near 100%** and is not meaningful: a 93-word
  paragraph is almost a unique fingerprint for its image. Judge the models by the
  benchmark numbers, not the training curves.

## Citation

```bibtex
@article{ghazanfari2026paragraph,
  title   = {A Paragraph is Worth a Thousand Captions},
  author  = {Ghazanfari, Mahyar},
  year    = {2026}
}
```

## License

[MIT](LICENSE). The CC3M images are subject to their original terms; we release only
the generated text annotations.
