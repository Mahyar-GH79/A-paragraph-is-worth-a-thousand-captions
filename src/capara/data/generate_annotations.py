"""Generate the CC3M caption/paragraph annotations with Qwen2-VL and Llama 3.2 Vision.

For each CC3M image-caption pair whose image still downloads:
  - Qwen2-VL writes five positive captions and five hard-negative captions.
  - Qwen2-VL writes one 3-6 sentence paragraph.
  - Llama 3.2 Vision scores the paragraph 0-10; the paragraph is regenerated up to
    ``--max-attempts`` times and the highest-scoring one is kept. The loop stops early
    once a paragraph scores above ``--llama-threshold``.

Output is JSONL, one record per line, written to
``<out-dir>/cc3m_qwen_llama_<max-samples>.jsonl``.
"""

import argparse
import json
import os
import re
from io import BytesIO
from typing import Any

import requests
import torch
from PIL import Image
from tqdm.auto import tqdm

from capara.common.paths import ANNOTATIONS_DIR

CC3M_DATASET_NAME = "google-research-datasets/conceptual_captions"
QWEN_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
LLAMA_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"

DEFAULT_MAX_SAMPLES = 500_000
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_LLAMA_THRESHOLD = 5
DEFAULT_POS_CAPTIONS = 5
DEFAULT_NEG_CAPTIONS = 5

POSITIVE_CAPTIONS_PROMPT = (
    "Using the given image and its original caption above, write five different short English captions.\n"
    "Each caption must be a correct description of the same image, but with different wording or focus.\n"
    "Each one should be a single sentence.\n"
    "Return them as a numbered list from 1 to 5."
)

HARD_NEGATIVE_CAPTIONS_PROMPT = (
    "The original caption above is a correct description of the image.\n"
    "Now you must create hard negative captions for image-text retrieval.\n"
    "Write five short English captions that are similar in style and length to the original caption, "
    "but that are NOT correct descriptions of this image.\n"
    "Each caption should change at least one important visual detail (objects, number of objects, colors, "
    "actions, relationships, or scene) so that a careful viewer can see it is wrong, but it should still "
    "sound plausible as a generic web image caption.\n"
    "Do NOT mention that the captions are negative or wrong.\n"
    "Return them as a numbered list from 1 to 5."
)

PARAGRAPH_PROMPT = (
    "Using both the image and the original caption above, write a single, fluent English paragraph of "
    "three to six sentences that thoroughly and concretely describes the image.\n"
    "You may refine and expand on the original caption, but the paragraph must remain faithful to what "
    "is actually visible in the image.\n"
    "Describe the main objects, their attributes, relationships, and the overall scene.\n"
    "Do not mention that you were given a caption or that you are describing an image.\n\n"
    "Paragraph:"
)


def download_image(url: str, timeout: float = 10.0) -> Image.Image | None:
    """Download an image and return it as RGB, or ``None`` on any failure."""
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        print(f"[skip] failed to download {url}: {exc}")
        return None


def parse_score(text: str) -> int:
    """Extract the last integer in ``text``, clamped to [0, 10]. Returns -1 if none is found."""
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return -1
    return max(0, min(10, int(matches[-1])))


def extract_numbered_list(text: str, expected_n: int) -> list[str]:
    """Parse a numbered list ("1. x", "2) y", "3 - z") into at most ``expected_n`` items.

    Falls back to the whole text as a single item when no numbered line is present.
    """
    items: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^\d+\s*[\.\)\-:]", line):
            item = re.sub(r"^\d+\s*[\.\)\-:]\s*", "", line).strip()
            if item:
                items.append(item)
    if not items and text:
        items = [text.strip()]
    return items[:expected_n]


def load_models(
    qwen_model_name: str,
    llama_model_name: str,
) -> tuple[Any, Any, Any, Any]:
    """Load Qwen2-VL and Llama 3.2 Vision in bfloat16, sharded across the visible devices."""
    from transformers import (
        AutoProcessor,
        MllamaForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
    )

    print("Loading Qwen2-VL and Llama 3.2 Vision models...")
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        qwen_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    qwen_processor = AutoProcessor.from_pretrained(qwen_model_name)

    llama_model = MllamaForConditionalGeneration.from_pretrained(
        llama_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    llama_processor = AutoProcessor.from_pretrained(llama_model_name)
    print("Models loaded.")

    return qwen_model, qwen_processor, llama_model, llama_processor


def qwen_chat_with_image_and_caption(
    qwen_model: Any,
    qwen_processor: Any,
    image: Image.Image,
    original_caption: str,
    user_text: str,
    max_new_tokens: int = 128,
) -> str:
    """Prompt Qwen2-VL with the image, its original CC3M caption, and task instructions."""
    full_prompt = (
        "You are an expert image-language assistant.\n"
        "You will be given an image and its original caption from a dataset.\n"
        "Use both the visual content of the image and the caption information when following the instructions.\n\n"
        f"Original caption:\n{original_caption}\n\n"
        f"{user_text}"
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_prompt},
            ],
        }
    ]

    chat_text = qwen_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = qwen_processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    ).to(qwen_model.device)

    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

    gen_ids = output_ids[:, input_ids.shape[1] :]
    return qwen_processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()


def qwen_generate_positive_captions(
    qwen_model: Any,
    qwen_processor: Any,
    image: Image.Image,
    original_caption: str,
    n: int = DEFAULT_POS_CAPTIONS,
) -> list[str]:
    """Generate ``n`` correct captions that vary the wording or focus."""
    raw = qwen_chat_with_image_and_caption(
        qwen_model,
        qwen_processor,
        image,
        original_caption,
        POSITIVE_CAPTIONS_PROMPT,
        max_new_tokens=160,
    )
    return extract_numbered_list(raw, n)


def qwen_generate_hard_negative_captions(
    qwen_model: Any,
    qwen_processor: Any,
    image: Image.Image,
    original_caption: str,
    n: int = DEFAULT_NEG_CAPTIONS,
) -> list[str]:
    """Generate ``n`` captions in the same style as the original but factually wrong for this image."""
    raw = qwen_chat_with_image_and_caption(
        qwen_model,
        qwen_processor,
        image,
        original_caption,
        HARD_NEGATIVE_CAPTIONS_PROMPT,
        max_new_tokens=200,
    )
    return extract_numbered_list(raw, n)


def qwen_generate_paragraph(
    qwen_model: Any,
    qwen_processor: Any,
    image: Image.Image,
    original_caption: str,
) -> str:
    """Generate one 3-6 sentence paragraph describing the image."""
    paragraph = qwen_chat_with_image_and_caption(
        qwen_model,
        qwen_processor,
        image,
        original_caption,
        PARAGRAPH_PROMPT,
        max_new_tokens=200,
    )
    return paragraph.strip()


def llama_score_image_paragraph(
    llama_model: Any,
    llama_processor: Any,
    image: Image.Image,
    paragraph: str,
) -> int:
    """Score how faithfully ``paragraph`` describes ``image``, in [0, 10]; -1 if unparseable.

    Decoding is greedy so the judge is deterministic across the feedback attempts.
    """
    judge_instructions = (
        "You are a strict evaluator of image descriptions.\n"
        "You will see an image and a paragraph that claims to describe it.\n"
        "Give a score from 0 to 10 based only on how accurate and faithful "
        "the paragraph is to the image.\n"
        "0 means completely wrong, 5 means partially correct with important errors or omissions, "
        "and 10 means very accurate and detailed.\n"
        "Respond with a single integer number only, no extra words.\n\n"
        "Paragraph:\n"
        f"{paragraph}\n\n"
        "Score (0 to 10):"
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": judge_instructions},
            ],
        }
    ]

    chat_text = llama_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = llama_processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    ).to(llama_model.device)

    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output_ids = llama_model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    gen_ids = output_ids[:, input_ids.shape[1] :]
    out_text = llama_processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    return parse_score(out_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ANNOTATIONS_DIR),
        help="Directory for the output JSONL.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default=None,
        help="Output JSONL path (default: <out-dir>/cc3m_qwen_llama_<max-samples>.jsonl).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Target number of records to write.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Paragraph generation attempts per image.",
    )
    parser.add_argument(
        "--llama-threshold",
        type=int,
        default=DEFAULT_LLAMA_THRESHOLD,
        help="Stop regenerating a paragraph once the judge scores above this value.",
    )
    parser.add_argument("--pos-captions", type=int, default=DEFAULT_POS_CAPTIONS)
    parser.add_argument("--neg-captions", type=int, default=DEFAULT_NEG_CAPTIONS)
    parser.add_argument("--dataset-name", type=str, default=CC3M_DATASET_NAME)
    parser.add_argument("--qwen-model", type=str, default=QWEN_MODEL)
    parser.add_argument("--llama-model", type=str, default=LLAMA_MODEL)
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-image download timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import load_dataset

    out_dir = args.out_dir
    out_jsonl = args.out_jsonl or os.path.join(
        out_dir, f"cc3m_qwen_llama_{args.max_samples}.jsonl"
    )
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    qwen_model, qwen_processor, llama_model, llama_processor = load_models(
        args.qwen_model, args.llama_model
    )

    print(f"Loading CC3M dataset (streaming): {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    iterator = iter(dataset)

    processed = 0  # raw CC3M examples pulled from the stream
    written = 0  # records written to the JSONL

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        pbar = tqdm(total=args.max_samples, desc="Generating CC3M samples")

        while written < args.max_samples:
            try:
                try:
                    example = next(iterator)
                except StopIteration:
                    print("CC3M stream ended; restarting iterator.")
                    dataset = load_dataset(
                        args.dataset_name, split="train", streaming=True
                    )
                    iterator = iter(dataset)
                    example = next(iterator)

                processed += 1

                caption = example.get("caption", None)
                url = example.get("image_url", None)

                if not caption or not isinstance(caption, str):
                    continue
                if not url or not isinstance(url, str):
                    continue

                image = download_image(url, timeout=args.timeout)
                if image is None:
                    continue

                try:
                    pos_caps = qwen_generate_positive_captions(
                        qwen_model, qwen_processor, image, caption, args.pos_captions
                    )
                except Exception as exc:
                    print(f"[warn] positive caption gen failed at processed={processed}: {exc}")
                    pos_caps = []

                try:
                    neg_caps = qwen_generate_hard_negative_captions(
                        qwen_model, qwen_processor, image, caption, args.neg_captions
                    )
                except Exception as exc:
                    print(f"[warn] negative caption gen failed at processed={processed}: {exc}")
                    neg_caps = []

                best_paragraph: str | None = None
                best_score = -1

                for attempt in range(args.max_attempts):
                    try:
                        paragraph = qwen_generate_paragraph(
                            qwen_model, qwen_processor, image, caption
                        )
                    except Exception as exc:
                        print(
                            f"[warn] Qwen paragraph failed at processed={processed}, attempt={attempt}: {exc}"
                        )
                        continue

                    if not paragraph or not paragraph.strip():
                        continue

                    try:
                        score = llama_score_image_paragraph(
                            llama_model, llama_processor, image, paragraph
                        )
                    except Exception as exc:
                        print(
                            f"[warn] Llama scoring failed at processed={processed}, attempt={attempt}: {exc}"
                        )
                        continue

                    if score > best_score:
                        best_score = score
                        best_paragraph = paragraph

                    if score > args.llama_threshold:
                        break

                record: dict[str, Any] = {
                    "dataset": "CC3M",
                    "image_url": url,
                    "original_caption": caption,
                    "positive_captions": pos_caps,
                    "hard_negative_captions": neg_caps,
                    "paragraph": best_paragraph,
                    "llama_score": int(best_score),
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                pbar.update(1)

                if written % 10 == 0:
                    print(f"\n[info] processed={processed}, written={written}")
                    print("Original caption:", caption)
                    print("Positive[0]:", pos_caps[0] if pos_caps else None)
                    print("Negative[0]:", neg_caps[0] if neg_caps else None)
                    print("Best score:", best_score)
                    print("Paragraph snippet:", (best_paragraph or "")[:200], "...")
                    print("-" * 80)

            except KeyboardInterrupt:
                print("\n[info] Interrupted by user. Stopping early.")
                break
            except Exception as exc:
                print(f"[fatal-sample] unexpected error at processed={processed}: {exc}")
                continue

    print(
        f"\nDone. Processed {processed} raw CC3M samples, wrote {written} JSONL lines to {out_jsonl}"
    )


if __name__ == "__main__":
    main()
