import numpy as np
import matplotlib.pyplot as plt


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