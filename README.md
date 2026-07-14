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
hf download Mahyar-79/cc3m-caption-or-paragraph \
    --repo-type dataset --local-dir cc3m_qwen_llama_outputs
```

To regenerate it from scratch instead:

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
    --datasets flickr coco docci sharegpt4v

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

To exercise the whole training loop without the real 1 GB of embedding shards:

```bash
python tests/make_fixture_shards.py --out-dir /tmp/fixture --shards 4 --rows-per-shard 32
python -m capara.train --config cfg5 --device cpu --shards-dir /tmp/fixture \
    --epochs 1 --batch-size 4 --max-steps-per-epoch 2 --num-workers 0 --val-shards 2
```

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
