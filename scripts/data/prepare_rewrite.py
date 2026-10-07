"""Project Stage 2 approved final text into the active rewrite handoff."""

from pathlib import Path

from rcr.common.io import load_jsonl, write_jsonl
from rcr.dataset.rewrite import prepare_rewrite_inputs, validate_review_output

INPUT = Path("dataset/data/work/selection/selected.jsonl")
OUTPUT = Path("dataset/data/work/rewrite/rewrite_input.jsonl")
ACCEPTED = Path("dataset/data/work/rewrite/rewrite_output.jsonl")


def main() -> None:
    records = load_jsonl(INPUT)
    rewrite_inputs = prepare_rewrite_inputs(records)
    accepted = []
    for row in rewrite_inputs:
        desc, change = validate_review_output(
            row["case_type"],
            {"final_desc": row["final_desc"], "final_change": row["final_change"]},
        )
        accepted.append(
            {"sample_id": row["sample_id"], "final_desc": desc, "final_change": change}
        )

    write_jsonl(OUTPUT, rewrite_inputs)
    write_jsonl(ACCEPTED, accepted)

    print(f"Input: {len(records)}")
    print(f"Prepared: {len(rewrite_inputs)}")
    print(f"Wrote rewrite inputs to {OUTPUT}")
    print(f"Wrote accepted text to {ACCEPTED} (no Gemini request)")


if __name__ == "__main__":
    main()
