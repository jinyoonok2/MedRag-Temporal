import os
import re
from typing import List
from tqdm import tqdm


def _remove_underscores(line: str) -> str:
    return line.replace("_", "")


def _lowercase(line: str) -> str:
    return line.lower()


def _is_comparison_line(line: str) -> bool:
    """
    Treat any line that starts with 'comparison' (optionally followed by ':' or text)
    as part of the comparison section and drop it.
    """
    s = line.strip().lower()
    return s.startswith("comparison")  # catches 'comparison:', 'comparison is...', etc.


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _pipeline_refine(text: str) -> List[str]:
    """
    Refinement pipeline:
      - split into lines
      - drop comparison lines
      - remove underscores
      - lowercase
      - strip, drop empties
      - join into a single line (remove newlines)
      - collapse whitespace
      - return as a single-element list to keep caller interface stable
    """
    lines = text.split("\n")

    # 1) drop comparison lines
    lines = [ln for ln in lines if not _is_comparison_line(ln)]

    # 2) remove underscores, lowercase, strip
    lines = [_lowercase(_remove_underscores(ln)).strip() for ln in lines]

    # 3) drop empties
    lines = [ln for ln in lines if ln]

    # 4) join into single line (remove newlines) + collapse spaces
    merged = _collapse_spaces(" ".join(lines))

    # Return as [single_string] so generate_refined_reports can still "\n".join(...)
    return [merged]


def clean_report_text(text: str, mode: str = "refine") -> List[str]:
    """
    Cleans report text according to mode.

    Args:
        text (str): Original report text.
        mode (str): Cleaning mode. Options:
            - "none": leave text as-is (split into original lines)
            - "refine": remove underscores, lowercase, drop comparison lines,
                        remove newlines (merge into one line), collapse spaces
    Returns:
        List[str]: Lines to write (for "refine" this is a single-element list).
    """
    if mode == "none":
        return text.split("\n")
    elif mode == "refine":
        return _pipeline_refine(text)
    else:
        raise ValueError(f"Unsupported cleaning mode: {mode}")


def generate_refined_reports(data_loader, mode: str = "refine"):
    """
    Generate refined reports with chosen cleaning mode.

    Expects each batch to contain:
      - 'report_txt' (List[str])
      - 'refined_report_txt_path' (List[str])
    """
    print(f"\n🧼 Generating refined reports (mode = '{mode}')...")

    for batch in tqdm(data_loader, desc=f"Processing (mode='{mode}')"):
        report_texts = batch["report_txt"]
        refined_txt_paths = batch["refined_report_txt_path"]

        for i in range(len(report_texts)):
            lines = clean_report_text(report_texts[i], mode=mode)
            content = "\n".join(lines)  # for 'refine', this is just the single merged line

            os.makedirs(os.path.dirname(refined_txt_paths[i]), exist_ok=True)
            with open(refined_txt_paths[i], "w", encoding="utf-8") as f:
                f.write(content.strip())

    print(f"✅ Done. Reports generated with mode = '{mode}'")
