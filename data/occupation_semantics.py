#!/usr/bin/env python3
"""Attach fixed multilingual semantic vectors to a prepared occupation graph.

Only occupation tuples visible on training people are encoded.  Validation and
test people retain a shared ``occupation_semantic_id=0`` so the resulting
artifact preserves the masked-label transductive protocol.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from data.extended import MISSING_VALUE_TOKENS


FEATURE_NAME = "occupation_semantic"
DEFAULT_MODEL = "intfloat/multilingual-e5-base"
PROMPT_TEMPLATE = (
    "passage: A person's occupation hierarchy is Level 1: {level1}. "
    "Level 2: {level2}. Level 3: {level3}."
)
INVALID_OCCUPATION_LABELS = frozenset({"__unknown__", "__missing__", *MISSING_VALUE_TOKENS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared graph_data.pt with hierarchical occupation masks")
    parser.add_argument("--output", required=True, help="New semantic graph_data.pt; input is never overwritten")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None, help="Optional Hugging Face revision to pin")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but torch.cuda.is_available() is False")
    return requested


def invert_vocabulary(vocabulary: Dict[str, int], name: str) -> Dict[int, str]:
    inverse = {int(index): str(value) for value, index in vocabulary.items()}
    if len(inverse) != len(vocabulary) or sorted(inverse) != list(range(len(vocabulary))):
        raise ValueError(f"Occupation vocabulary '{name}' must be a dense ID mapping")
    return inverse


def is_valid_occupation_label(label: str) -> bool:
    return label.strip().casefold() not in INVALID_OCCUPATION_LABELS


def build_semantic_ids(data, metadata: Dict) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    """Map visible training occupation tuples to deterministic non-zero IDs."""
    required = [f"occupation_level{level}" for level in (1, 2, 3)]
    if not all(hasattr(data, name) for name in required):
        raise ValueError("The source artifact lacks hierarchical occupation tensors; rerun prepare first")
    if "occupation_unknown_ids" not in metadata or "occupation_vocabularies" not in metadata:
        raise ValueError("The source artifact lacks hierarchical occupation masking metadata; rerun prepare first")

    occupation_ids = torch.stack([getattr(data, name).cpu() for name in required], dim=1)
    inverse_vocabularies = {
        level: invert_vocabulary(
            metadata["occupation_vocabularies"][f"occupation_level{level}"], f"occupation_level{level}"
        )
        for level in (1, 2, 3)
    }
    visible = data.train_mask.cpu().clone()
    for name in required:
        visible &= getattr(data, name).cpu().ne(int(metadata["occupation_unknown_ids"][name]))
    for column, level in enumerate((1, 2, 3)):
        labels = [inverse_vocabularies[level][index] for index in occupation_ids[:, column].tolist()]
        visible &= torch.tensor([is_valid_occupation_label(label) for label in labels], dtype=torch.bool)

    semantic_ids = torch.zeros(occupation_ids.size(0), dtype=torch.long)
    if not visible.any():
        raise ValueError("No training people have a complete visible L1/L2/L3 occupation tuple")

    tuples, inverse = torch.unique(occupation_ids[visible], dim=0, sorted=True, return_inverse=True)
    semantic_ids[visible] = inverse + 1
    if semantic_ids[~data.train_mask.cpu()].any():
        raise RuntimeError("Validation/test people must not receive a semantic occupation ID")

    entries: List[Dict[str, object]] = []
    for semantic_id, (level1_id, level2_id, level3_id) in enumerate(tuples.tolist(), start=1):
        level1 = inverse_vocabularies[1][level1_id]
        level2 = inverse_vocabularies[2][level2_id]
        level3 = inverse_vocabularies[3][level3_id]
        if not all(is_valid_occupation_label(label) for label in (level1, level2, level3)):
            raise RuntimeError("Missing/unknown occupation values must not produce a semantic prompt")
        entries.append({
            "semantic_id": semantic_id,
            "level1": level1,
            "level2": level2,
            "level3": level3,
            "prompt": PROMPT_TEMPLATE.format(level1=level1, level2=level2, level3=level3),
        })
    return semantic_ids, entries


def encode_prompts(
    prompts: List[str], model_name: str, model_revision: Optional[str], batch_size: int, device: str
) -> Tuple[torch.Tensor, Optional[str]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "occupation-embed requires sentence-transformers; install the project requirements first"
        ) from error

    model = SentenceTransformer(model_name, revision=model_revision, device=device)
    vectors = model.encode(
        prompts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).detach().cpu().float()
    if vectors.dim() != 2 or vectors.size(0) != len(prompts):
        raise RuntimeError("Sentence encoder returned an unexpected embedding shape")
    resolved_revision = getattr(getattr(model, "_first_module", lambda: None)(), "auto_model", None)
    resolved_revision = getattr(getattr(resolved_revision, "config", None), "_commit_hash", None)
    return vectors, resolved_revision


def main() -> None:
    args = parse_args()
    input_path, output_path = Path(args.data), Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--output must differ from --data so the source artifact remains unchanged")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    bundle = torch.load(input_path, map_location="cpu", weights_only=False)
    if not {"data", "metadata"}.issubset(bundle):
        raise ValueError("Expected a prepared graph artifact with data and metadata")
    data, metadata = bundle["data"], bundle["metadata"]
    if FEATURE_NAME in metadata.get("feature_schema", {}):
        raise ValueError(f"Source artifact already contains '{FEATURE_NAME}'; use the original categorical artifact")

    semantic_ids, entries = build_semantic_ids(data, metadata)
    prompts = [entry["prompt"] for entry in entries]
    # Hash the exact ordered prompts, not just the template.  This makes an
    # artifact self-describing even when the vocabulary or its labels change.
    prompt_fingerprint = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()
    vectors, resolved_revision = encode_prompts(
        prompts, args.model_name, args.model_revision, args.batch_size, resolve_device(args.device)
    )
    table = torch.zeros((vectors.size(0) + 1, vectors.size(1)), dtype=torch.float32)
    table[1:] = vectors
    if not torch.allclose(torch.linalg.vector_norm(table[1:], dim=1), torch.ones(table.size(0) - 1), atol=1e-4):
        raise RuntimeError("Occupation semantic vectors are not L2-normalised")

    data.occupation_semantic = semantic_ids
    metadata = dict(metadata)
    metadata["feature_schema"] = dict(metadata["feature_schema"])
    metadata["feature_schema"][FEATURE_NAME] = {
        "kind": "semantic_categorical",
        "cardinality": int(table.size(0)),
        "input_dim": int(table.size(1)),
        "unknown_id": 0,
        "semantic_table_key": FEATURE_NAME,
    }
    metadata["occupation_unknown_ids"] = dict(metadata["occupation_unknown_ids"])
    metadata["occupation_unknown_ids"][FEATURE_NAME] = 0
    semantic_tables = dict(metadata.get("semantic_feature_tables", {}))
    semantic_tables[FEATURE_NAME] = table
    metadata["semantic_feature_tables"] = semantic_tables
    metadata["semantic_features"] = dict(metadata.get("semantic_features", {}))
    metadata["semantic_features"][FEATURE_NAME] = {
        "model_name": args.model_name,
        "requested_revision": args.model_revision,
        "resolved_revision": resolved_revision,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_fingerprint": prompt_fingerprint,
        "normalised": True,
        "embedding_dim": int(table.size(1)),
        "entries": entries,
        "source_artifact": str(input_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": data, "metadata": metadata}, output_path)

    manifest_path = output_path.with_name(f"{output_path.stem}_occupation_semantic_prompts.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata["semantic_features"][FEATURE_NAME], handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "output": str(output_path),
        "prompt_manifest": str(manifest_path),
        "semantic_occupation_count": len(entries),
        "embedding_dim": int(table.size(1)),
        "model_name": args.model_name,
        "resolved_revision": resolved_revision,
        "prompt_fingerprint": prompt_fingerprint,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
