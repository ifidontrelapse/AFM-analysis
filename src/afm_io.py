"""
Утилиты для загрузки AFM данных из различных форматов и генерации топологических карт
(в нанометрах) для тестирования и демонстрации.

- load_afm: поддержка форматов .spm, .ibw, .gwy и .npy.
- make_synthetic_afm: планируется
"""

from __future__ import annotations

import numpy as np
import os
import re


# =========================
# ВНУТРЕННИЙ кастомный парсер Bruker/Nanoscope
# =========================

def _read_nanoscope_z(file_path: str) -> np.ndarray:
    """
    Кастомное чтение Bruker/Nanoscope (.spm, .000, .001).

    Возвращает:
        2D массив высот в нанометрах (float32)
    """

    HEADER_READ_BYTES = 65536

    # --- читаем ASCII header ---
    with open(file_path, "rb") as f:
        raw = f.read(HEADER_READ_BYTES)

    header = raw.split(b"\x1A")[0].decode("latin-1", errors="ignore")

    # --- ищем блоки *Ciao image list ---
    blocks = header.split("\\*Ciao image list")
    if len(blocks) < 2:
        raise ValueError("Ciao image list blocks not found")

    # выбираем первый блок (обычно Height)
    blk = blocks[1]

    def find_int(pattern: str):
        m = re.search(pattern, blk)
        return int(m.group(1)) if m else None

    data_offset = find_int(r"Data offset\s*:\s*(\d+)")
    data_length = find_int(r"Data length\s*:\s*(\d+)")
    samps = find_int(r"Samps/line\s*:\s*(\d+)")
    lines = find_int(r"Number of lines\s*:\s*(\d+)")
    bpp = find_int(r"Bytes/pixel\s*:\s*(\d+)")

    if None in (data_offset, data_length, samps, lines, bpp):
        raise ValueError("Header fields missing in SPM file")

    # --- парсим Z-scale ---
    # V/LSB
    v_match = re.search(r"\(([^)]+)\s*V/LSB\)", blk)
    if not v_match:
        raise ValueError("V/LSB not found")

    v_per_lsb = float(v_match.group(1))

    # Zsens nm/V
    zsens_match = re.search(r"Zsens[^\n]*?([0-9.]+)\s*nm/V", header)
    if not zsens_match:
        raise ValueError("Zsens nm/V not found")

    nm_per_v = float(zsens_match.group(1))
    z_scale = v_per_lsb * nm_per_v

    # --- читаем бинарные данные ---
    dtype = np.int16 if bpp == 2 else np.int32

    with open(file_path, "rb") as f:
        f.seek(data_offset)
        raw = np.frombuffer(f.read(data_length), dtype=dtype)

    z = raw[: lines * samps].reshape((lines, samps)).astype(np.float32)
    z *= z_scale  # в нм

    return z


# =========================
# Публичный интерфейс
# =========================

def load_afm(file_path: str, fmt: str) -> np.ndarray:
    """
    Загрузка AFM данных из различных форматов
    и генерация топологических карт (в нанометрах).

    Supported formats:
    - "spm": Bruker .spm / .000 / .001 (custom parser)
    - "ibw": Asylum .ibw via igor
    - "gwy": Gwyddion .gwy via pygwyfile
    - "npy": raw NumPy array

    Args:
        file_path: путь к файлу AFM
        fmt: формат ("spm", "ibw", "gwy", "npy")

    Returns:
        2D numpy array (float32) — Z-map в нанометрах.
    """

    if fmt == "npy":
        z = np.load(file_path).astype(np.float32)

    elif fmt == "spm":
        z = _read_nanoscope_z(file_path)

    elif fmt == "ibw":
        import igor.binarywave as bw
        data = bw.load(file_path)
        z = (data["wave"]["wData"][:, :, 0] * 1e9).astype(np.float32)

    elif fmt == "gwy":
        import pygwyfile
        gwy = pygwyfile.load(file_path)
        z = (pygwyfile.util.get_datafield(gwy, "/0/data") * 1e9).astype(np.float32)

    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return z


def make_synthetic_afm(size: int = 256, n_particles: int = 40, seed: int = 42) -> np.ndarray:
    """
    Генерация синтетической AFM Z-карты с заданным количеством частиц и размером.
    Планируется.
    """
    pass
