"""
Batch preprocessing of AFM images.

Usage:
    python preprocess_batch.py <input_dir> <output_dir>

Recursively finds all files with numeric extensions (.001, .002, …) in
input_dir, applies the standard preprocessing pipeline
(flatten_plane → flatten_lines → build_substrate_map) and saves z_result
as a .jpg into output_dir, mirroring the subdirectory structure so that
files with the same name from different folders never collide.

Example:
    input/exp1/img.001  →  output/exp1/img.001.jpg
    input/exp2/img.001  →  output/exp2/img.001.jpg
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from src.afm_io import load_afm
from src.preprocess import flatten_plane, flatten_lines, build_substrate_map


def process_file(src: Path, dst: Path) -> None:
    scan_size_nm, pixel_size_nm, z = load_afm(str(src), fmt="spm")

    z_plane = flatten_plane(z)
    z_flat = flatten_lines(z_plane, poly_order=1)
    _, z_result, _, _ = build_substrate_map(z_flat, pixel_size_nm, min_size_nm=5)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(z_result, cmap="afmhot", origin="lower")
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_spm_files(root: Path) -> list[Path]:
    return sorted(
        f for f in root.rglob("*")
        if f.is_file() and f.suffix.lstrip(".").isdigit()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch AFM preprocessor")
    parser.add_argument("input_dir", type=Path, help="Root folder with SPM files")
    parser.add_argument("output_dir", type=Path, help="Root folder for output .jpg files")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = find_spm_files(input_dir)
    if not files:
        print(f"No files with numeric extensions found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    ok = failed = 0
    for src in files:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel.parent / (src.name + ".jpg")
        dst.parent.mkdir(parents=True, exist_ok=True)

        print(f"  {rel}  →  {dst.relative_to(output_dir)}", end="", flush=True)
        try:
            process_file(src, dst)
            print(" ✓")
            ok += 1
        except Exception as e:
            print(f" FAILED: {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone. {ok} converted, {failed} failed  →  {output_dir}")


if __name__ == "__main__":
    main()
