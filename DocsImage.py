from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.afm_io import load_afm
from src.detection import detect_particles
from src.measure import measure_all_baseline
from src.preprocess import build_substrate_map, flatten_lines, flatten_plane

def save_raw_flat_figure(z_raw: np.ndarray, z_flat: np.ndarray, scan_size_nm: float, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    extent = [0, scan_size_nm, 0, scan_size_nm]
    for ax, data, title in zip(
        axes,
        [z_raw, z_flat],
        ["Исходная карта высот z_raw", "После выравнивания z_flat"],
        strict=True,
    ):
        im = ax.imshow(data, cmap="afmhot", origin="lower", extent=extent)
        plt.colorbar(im, ax=ax, label="Высота, нм", fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("X, нм")
        ax.set_ylabel("Y, нм")
    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_substrate_figure(substrate: np.ndarray, z_above: np.ndarray, scan_size_nm: float, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    extent = [0, scan_size_nm, 0, scan_size_nm]
    for ax, data, title in zip(
        axes,
        [substrate, z_above],
        ["Оценка поверхности подложки", "Частицы над подложкой z_above"],
        strict=True,
    ):
        im = ax.imshow(data, cmap="afmhot", origin="lower", extent=extent)
        plt.colorbar(im, ax=ax, label="Высота, нм", fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("X, нм")
        ax.set_ylabel("Y, нм")
    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_detections_figure(z_above: np.ndarray, blobs: np.ndarray, pixel_size_nm: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(z_above, cmap="afmhot", origin="lower")
    plt.colorbar(im, ax=ax, label="Высота, нм", fraction=0.046, pad=0.04)
    for y, x, sigma, _radius_nm in blobs:
        radius_px = sigma * np.sqrt(2.0)
        circle = plt.Circle((x, y), radius_px, fill=False, color="cyan", linewidth=1.2, alpha=0.9)
        ax.add_patch(circle)
        ax.plot(x, y, "+", color="white", markersize=4, markeredgewidth=0.9)
    ticks = np.linspace(0, z_above.shape[1] - 1, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t * pixel_size_nm:.0f}" for t in ticks])
    ax.set_yticklabels([f"{t * pixel_size_nm:.0f}" for t in ticks])
    ax.set_xlabel("X, нм")
    ax.set_ylabel("Y, нм")
    ax.set_title(f"LoG-детекция частиц, N={len(blobs)}")
    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_hist_figure(measurements, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.hist(measurements["height_nm"], bins=18, color="goldenrod", edgecolor="white", linewidth=0.8)
    ax.axvline(
        measurements["height_nm"].median(),
        color="firebrick",
        linestyle="--",
        linewidth=1.5,
        label=f"Медиана: {measurements['height_nm'].median():.2f} нм",
    )
    ax.axvline(
        measurements["height_nm"].mean(),
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        label=f"Среднее: {measurements['height_nm'].mean():.2f} нм",
    )
    ax.set_xlabel("Высота, нм")
    ax.set_ylabel("Количество частиц")
    ax.set_title("Распределение измеренных высот")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path("docs/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_size_nm, pixel_size_nm, z_raw = load_afm("data/pvp8k/2-5-dmfa-pvp-temp.030", fmt="spm")
    z_plane = flatten_plane(z_raw)
    z_flat = flatten_lines(z_plane, poly_order=1)
    substrate, z_above, opening_radius, sizes = build_substrate_map(
        z=z_flat,
        pixel_size_nm=pixel_size_nm,
        min_size_nm=5,
    )
    blobs = detect_particles(z_above, pixel_size_nm, sizes, overlap=0.4, percentile=30.0)
    measurements = measure_all_baseline(
        z_flat,
        z_above,
        blobs,
        outer_px=5,
        inner_erode_px=3,
    )

    save_raw_flat_figure(z_raw, z_flat, scan_size_nm, out_dir / "afm_raw_vs_flat.png")
    save_substrate_figure(substrate, z_above, scan_size_nm, out_dir / "substrate_and_z_above.png")
    save_detections_figure(z_above, blobs, pixel_size_nm, out_dir / "log_detection_overlay.png")
    save_hist_figure(measurements, out_dir / "height_histogram.png")

    print(f"opening_radius_px={opening_radius}")
    print(f"n_blobs={len(blobs)}")
    print(f"n_measurements={len(measurements)}")
    print(measurements.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
