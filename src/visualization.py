import numpy as np
import matplotlib.pyplot as plt

from src.types import PipelineResult, Detection


def plot_afm(
    ax: plt.Axes,
    z: np.ndarray,
    scan_size_nm: float,
    cmap: str = "afmhot",
    colorbar: bool = True
):
    """
    Отображает AFM карту высот на заданной оси.

    Args:
        ax: matplotlib ось
        z: карта высот (nm)
        scan_size_nm: размер скана (nm)
        cmap: colormap
        colorbar: добавить ли colorbar

    Returns:
        im: matplotlib image object
    """

    extent = [0, scan_size_nm, 0, scan_size_nm]

    im = ax.imshow(
        z,
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="nearest"
    )

    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_aspect("equal")

    if colorbar:
        plt.colorbar(im, ax=ax, label="Height (nm)")

    return im


def afm_viewer(
    z: np.ndarray,
    scan_size_nm: float | None = None,
    cmap: str = "afmhot"
):
    """
    Интерактивный viewer для AFM изображения.

    Возможности:
    - отображение координат и высоты
    - crosshair
    - профиль высоты по двум точкам

    Args:
        z: карта высот (nm)
        scan_size_nm: размер скана (nm)
        cmap: colormap

    Returns:
        fig, ax, ax_profile, im
    """

    h, w = z.shape
    pixel_size = scan_size_nm / w if scan_size_nm else None

    fig, (ax, ax_profile) = plt.subplots(
        1, 2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1, 1]}
    )

    if scan_size_nm:
        im = plot_afm(ax, z, scan_size_nm, cmap=cmap)
    else:
        im = ax.imshow(z, cmap=cmap, origin="lower")
        plt.colorbar(im, ax=ax, label="Height (nm)")

    ax.set_title("AFM height map")

    # crosshair
    hline = ax.axhline(0, color="white", lw=0.5)
    vline = ax.axvline(0, color="white", lw=0.5)

    # точки
    points = []
    markers = ax.scatter([], [], c="red", s=50)

    # текст
    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        color="white",
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6)
    )

    def on_move(event):

        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        x_nm = event.xdata
        y_nm = event.ydata

        if pixel_size:
            x_px = int(x_nm / pixel_size)
            y_px = int(y_nm / pixel_size)
        else:
            x_px = int(x_nm)
            y_px = int(y_nm)

        if not (0 <= x_px < w and 0 <= y_px < h):
            return

        hline.set_ydata([y_nm, y_nm])
        vline.set_xdata([x_nm, x_nm])

        height = z[y_px, x_px]

        if pixel_size:
            text.set_text(
                f"x = {x_nm:.1f} nm\n"
                f"y = {y_nm:.1f} nm\n"
                f"h = {height:.2f} nm"
            )
        else:
            text.set_text(
                f"x = {x_px}\n"
                f"y = {y_px}\n"
                f"h = {height:.2f} nm"
            )

        fig.canvas.draw_idle()

    def on_click(event):

        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        x_nm = event.xdata
        y_nm = event.ydata

        if pixel_size:
            x_px = int(x_nm / pixel_size)
            y_px = int(y_nm / pixel_size)
        else:
            x_px = int(x_nm)
            y_px = int(y_nm)

        if not (0 <= x_px < w and 0 <= y_px < h):
            return

        points.append((x_px, y_px))

        xs = [p[0] * pixel_size if pixel_size else p[0] for p in points]
        ys = [p[1] * pixel_size if pixel_size else p[1] for p in points]

        markers.set_offsets(np.column_stack([xs, ys]))

        if len(points) == 2:
            draw_profile(points[0], points[1])
            points.clear()
            markers.set_offsets(np.empty((0, 2)))

        fig.canvas.draw_idle()

    def draw_profile(p1, p2):

        x0, y0 = p1
        x1, y1 = p2

        n = int(np.hypot(x1 - x0, y1 - y0))
        n = max(n, 2)

        xs = np.linspace(x0, x1, n).astype(int)
        ys = np.linspace(y0, y1, n).astype(int)

        profile = z[ys, xs]

        ax_profile.clear()

        if pixel_size:
            dist = np.linspace(0, n * pixel_size, n)
            ax_profile.set_xlabel("Distance (nm)")
        else:
            dist = np.arange(n)
            ax_profile.set_xlabel("Pixels")

        ax_profile.plot(dist, profile)

        ax_profile.set_ylabel("Height (nm)")
        ax_profile.set_title("Height profile")

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)

    return fig, ax, ax_profile, im

def plot_detections(
    z_above: np.ndarray,
    blobs: np.ndarray,
    pixel_size_nm: float,
    axes: plt.Axes
):
    """
    Визуализация результатов LoG детекции.

    Панель 1: z_above с кружками вокруг каждой частицы
    Панель 2: гистограмма радиусов в нм

    Returns:
        axes
    """

    axes.imshow(z_above, cmap="afmhot", origin="lower")

    for blob in blobs:
        y, x, sigma, _ = blob
        radius_px = sigma * np.sqrt(2)
        circle = plt.Circle(
            (x, y), radius_px,
            color="cyan", fill=False, linewidth=1.2, alpha=0.8
        )
        axes.add_patch(circle)
        axes.plot(x, y, "+", color="cyan", markersize=4, markeredgewidth=0.8)
    h, w = z_above.shape
    ticks_px = np.linspace(0, w - 1, 5)
    ticks_py = np.linspace(0, h - 1, 5)
    axes.set_xticks(ticks_px)
    axes.set_yticks(ticks_py)
    axes.set_xticklabels([f"{v * pixel_size_nm:.0f}" for v in ticks_px])
    axes.set_yticklabels([f"{v * pixel_size_nm:.0f}" for v in ticks_py])
    axes.set_xlabel("X, nm")
    axes.set_ylabel("Y, nm")
    axes.set_title(f"LoG detection: {len(blobs)} particles", fontsize=11)

    return axes

def plot_detections_histogram(
    blobs: np.ndarray,
    axes: plt.Axes,
):
    if len(blobs) > 0:
        radius_nm = blobs[:, 3]
        axes.hist(
            radius_nm, bins=20,
            color="steelblue", edgecolor="white", linewidth=0.7
        )
        axes.axvline(
            np.median(radius_nm), color="gold",
            linestyle="--", linewidth=1.5,
            label=f"Медиана: {np.median(radius_nm):.1f} нм"
        )
        axes.axvline(
            np.mean(radius_nm), color="tomato",
            linestyle="--", linewidth=1.5,
            label=f"Среднее: {np.mean(radius_nm):.1f} нм"
        )
        axes.set_xlabel("Радиус, нм")
        axes.set_ylabel("Количество частиц")
        axes.set_title("Распределение радиусов (LoG)", fontsize=11)
        axes.legend(fontsize=9)
        axes.grid(alpha=0.3)
    else:
        axes.text(
            0.5, 0.5, "Частицы не найдены",
            ha="center", va="center", transform=axes.transAxes,
            fontsize=12
        )
    return axes


def plot_pipeline_result(
    result: PipelineResult,
    z_result: np.ndarray,
    scan_size_nm: float,
) -> plt.Figure:
    """
    Visualise a PipelineResult in a single figure.

    Layout depends on mode:

    mode="detect"  → two panels:
        [0] z_result with detection circles
        [1] radius distribution histogram

    mode="segment" → three panels:
        [0] z_result with SAM2 mask overlay + detection centres
        [1] radius distribution histogram
        [2] height distribution histogram

    Args:
        result:       PipelineResult from run_pipeline()
        z_result:     z_flat - substrate array (same one passed to run_pipeline)
        scan_size_nm: full scan size in nm (from PreprocessingResult.scan_size_nm)

    Returns:
        matplotlib Figure
    """
    from src.segmentation import overlay_masks, afm_to_rgb

    is_segment = result.mode == "segment" and len(result.masks) > 0
    n_cols     = 3 if is_segment else 2
    fig, axes  = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    pixel_size_nm = result.pixel_size_nm

    # ── Panel 0: image + detections or masks ─────────────────────────────────
    ax = axes[0]

    if is_segment:
        rgb     = afm_to_rgb(z_result)
        overlay = overlay_masks(rgb, result.masks, alpha=0.45)
        ax.imshow(overlay, origin="lower")
        ax.set_title(f"SAM2 masks — {len(result.masks)} particles")
    else:
        ax.imshow(z_result, cmap="afmhot", origin="lower")
        ax.set_title(f"Detections — {len(result.detections)} particles")

    for det in result.detections:
        circle = plt.Circle(
            (det.x_px, det.y_px), det.radius_px,
            color="cyan", fill=False, linewidth=1.0, alpha=0.7,
        )
        ax.add_patch(circle)
        ax.plot(det.x_px, det.y_px, "+", color="cyan", markersize=4, markeredgewidth=0.8)

    h, w     = z_result.shape
    ticks_px = np.linspace(0, w - 1, 5)
    ticks_py = np.linspace(0, h - 1, 5)
    ax.set_xticks(ticks_px)
    ax.set_yticks(ticks_py)
    ax.set_xticklabels([f"{v * pixel_size_nm:.0f}" for v in ticks_px])
    ax.set_yticklabels([f"{v * pixel_size_nm:.0f}" for v in ticks_py])
    ax.set_xlabel("X, nm")
    ax.set_ylabel("Y, nm")

    # ── Panel 1: radius histogram ─────────────────────────────────────────────
    ax       = axes[1]
    radii_nm = np.array([d.radius_nm for d in result.detections])
    if len(radii_nm) > 0:
        ax.hist(radii_nm, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(np.median(radii_nm), color="gold", linestyle="--", linewidth=1.5,
                   label=f"Median: {np.median(radii_nm):.1f} nm")
        ax.axvline(np.mean(radii_nm), color="tomato", linestyle="--", linewidth=1.5,
                   label=f"Mean: {np.mean(radii_nm):.1f} nm")
        ax.legend(fontsize=9)
    ax.set_xlabel("Radius, nm")
    ax.set_ylabel("Count")
    ax.set_title(f"Radius distribution  (n={len(radii_nm)})")
    ax.grid(alpha=0.3)

    # ── Panel 2: height histogram (segment mode only) ─────────────────────────
    if is_segment:
        ax        = axes[2]
        height_nm = result.measurements["height_nm"].dropna()
        if len(height_nm) > 0:
            ax.hist(height_nm, bins=30, color="darkorange", edgecolor="white", alpha=0.85)
            ax.axvline(height_nm.median(), color="gold", linestyle="--", linewidth=1.5,
                       label=f"Median: {height_nm.median():.1f} nm")
            ax.axvline(height_nm.mean(), color="tomato", linestyle="--", linewidth=1.5,
                       label=f"Mean: {height_nm.mean():.1f} nm")
            ax.legend(fontsize=9)
        ax.set_xlabel("Height, nm")
        ax.set_ylabel("Count")
        ax.set_title(f"Height distribution  (n={len(height_nm)})")
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"{result.detector_name.upper()} + {result.mode}  |  "
        f"scan {scan_size_nm:.0f} nm  |  {pixel_size_nm:.2f} nm/px",
        fontsize=12,
    )
    plt.tight_layout()
    return fig