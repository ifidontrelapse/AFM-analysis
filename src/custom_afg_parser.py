"""
custom_afm_parser.py

Надежный кастомный парсер Bruker/Nanoscope SPM файлов (.000, .001, .spm).
Возвращает карту высот в нанометрах (nm), размер пикселя в nm и подробные метаданные.

Основные функции:
- parse_nanoscope_header(path) -> dict
- read_nanoscope_channel(path) -> (z_map_nm, pixel_size_nm, metadata)
- read_nanoscope_fallback(path) -> tries custom -> nanoscope -> pySPM
- load_afm(path, fmt=None) -> unified loader (returns z, pixel_size_nm, meta)
- make_synthetic_afm(...) -> Генерация тестовой карты высот

Автор: адаптировано для пользователя
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Опциональные внешние зависимости (используются только в fallback)
try:
    import nanoscope as ns_mod  # type: ignore
    HAS_NANOSCOPE = True
except Exception:
    HAS_NANOSCOPE = False

try:
    import pySPM as pyspm_mod  # type: ignore
    HAS_PYSPM = True
except Exception:
    HAS_PYSPM = False

# Логирование
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Если модуль используется как библиотека, не менять конфигурацию root-логгера глобально.
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Константы
HEADER_READ_BYTES = 65536  # сколько байт читать для ASCII-заголовка
HEADER_PREVIEW_CHARS = 1000
UM_TO_NM = 1000.0

# Параметры синтетики
DEFAULT_SYNTH_SHAPE = (256, 256)
DEFAULT_SYNTH_PARTICLES = 40
SYNTH_MIN_HEIGHT_NM = 2.0
SYNTH_MAX_HEIGHT_NM = 25.0
SYNTH_MIN_RADIUS_PX = 2.0
SYNTH_MAX_RADIUS_PX = 12.0
SYNTH_TILT_X = 0.02
SYNTH_TILT_Y = -0.015
SYNTH_NOISE_STD_NM = 0.3


# -------------------------
# Вспомогательные функции
# -------------------------
def nm_extent(z: np.ndarray, pixel_size_nm: float) -> List[float]:
    """
    Возвращает extent (x0, x1, y1, y0) в нм для matplotlib.imshow.
    """
    return [0, z.shape[1] * pixel_size_nm, z.shape[0] * pixel_size_nm, 0]


def _parse_scan_size_nm(line: str) -> Optional[float]:
    """
    Парсит строку с 'Scan Size' и возвращает значение в нм, если возможно.
    Поддерживает: nm, um, µm, micron.
    """
    m = re.search(r"Scan Size\s*:\s*([0-9.]+)", line)
    if not m:
        return None
    val = float(m.group(1))
    lower = line.lower()
    if "nm" in lower:
        return val
    if any(u in lower for u in ("um", "µm", "~m", "micron")):
        return val * UM_TO_NM
    return val


def _parse_z_scale(header: str, block: str) -> float:
    """
    Вычисляет Z-scale в nm/LSB.
    Попытки:
        1) найти выражение "(X V/LSB) * Y nm/V" в полном заголовке
        2) найти "(X V/LSB)" в блоке и Zsens ... nm/V в заголовке
    Бросает ValueError, если не найдено.
    """
    # Вариант 1: (V/LSB) * nm/V прямо
    direct = re.search(r"\(\s*([0-9.]+)\s*V/LSB\s*\)\s*\*\s*([0-9.]+)\s*nm/V", header)
    if direct:
        v_per_lsb = float(direct.group(1))
        nm_per_v = float(direct.group(2))
        return v_per_lsb * nm_per_v

    # Вариант 2: найти V/LSB в блоке и Zsens nm/V в header
    v_match = re.search(r"\(([^)]+)\s*V/LSB\)", block)
    if not v_match:
        # иногда встречается "Z scale : V [..]" — попытаемся найти более общий паттерн
        v_match = re.search(r"Z scale\s*:\s*V[^\(]*\(([^)]+)\s*V/LSB\)", block, flags=re.IGNORECASE)
    if not v_match:
        raise ValueError("Z scale (V/LSB) не найден в блоке заголовка")

    v_per_lsb = float(v_match.group(1))
    zsens_match = re.search(r"Zsens[^\n]*?([0-9.]+)\s*nm/V", header, flags=re.IGNORECASE)
    if not zsens_match:
        # иногда пишут "Zsens: 11.58821 nm/V" или "@Sens. Zsens: 11.58821 nm/V"
        zsens_match = re.search(r"([Zz]sens)[^\n]*?([0-9.]+)\s*nm/V", header)
    if not zsens_match:
        raise ValueError("Zsens (nm/V) не найден в заголовке")

    nm_per_v = float(zsens_match.group(2) if len(zsens_match.groups()) >= 2 else zsens_match.group(1))
    return v_per_lsb * nm_per_v


def _parse_ciao_blocks(header: str) -> List[Dict[str, Any]]:
    """
    Разбирает блоки '*Ciao image list' и извлекает ключевые метрики.
    Возвращает список словарей с ключами:
      image_label, data_offset, data_length, samps, lines, bytes_per_pixel, scan_size_nm, block
    """
    # Разбиваем по метке (в файле она обычно записана как "\*Ciao image list")
    parts = header.split("\\*Ciao image list")
    out: List[Dict[str, Any]] = []
    for blk in parts[1:]:
        def _find_int(pat: str) -> Optional[int]:
            m = re.search(pat, blk)
            return int(m.group(1)) if m else None

        label_m = re.search(r'Image Data:.*?"([^"]+)"', blk)
        image_label = label_m.group(1) if label_m else None

        scan_size_nm: Optional[float] = None
        for line in blk.splitlines():
            if "Scan Size" in line:
                scan_size_nm = _parse_scan_size_nm(line)
                if scan_size_nm is not None:
                    break

        out.append({
            "image_label": image_label,
            "data_offset": _find_int(r"Data offset\s*:\s*(\d+)"),
            "data_length": _find_int(r"Data length\s*:\s*(\d+)"),
            "samps": _find_int(r"Samps/line\s*:\s*(\d+)"),
            "lines": _find_int(r"Number of lines\s*:\s*(\d+)"),
            "bytes_per_pixel": _find_int(r"Bytes/pixel\s*:\s*(\d+)"),
            "scan_size_nm": scan_size_nm,
            "block": blk
        })
    return out


# -------------------------
# Основные функции парсера
# -------------------------
def parse_nanoscope_header(path: str) -> Dict[str, Any]:
    """
    Прочитать ASCII-заголовок Nanoscope/Bruker файла и вернуть структурированную информацию.

    Parameters
    ----------
    path : str
        Путь к .001/.000/.spm файлу

    Returns
    -------
    dict
        Словарь с метаданными (включая channels, выбранный height block и preview).
    """
    with open(path, "rb") as f:
        raw = f.read(HEADER_READ_BYTES)

    # Заголовчик отделён символом SUB (0x1A). Иногда его нет — тогда используем всё что прочитали.
    split = raw.split(b"\x1A", 1)
    head_bytes = split[0] if split else raw
    header = head_bytes.decode("latin-1", errors="ignore")

    channels = _parse_ciao_blocks(header)
    if not channels:
        # Некоторые файлы могут иметь другую структуру — возвращаем минимальный header
        raise ValueError("Не удалось найти блоки '*Ciao image list' в заголовке")

    # Ищем наиболее подходящий блок - 'height' (регистр не важен)
    height_block = None
    for c in channels:
        lbl = (c.get("image_label") or "").lower()
        if "height" in lbl or "height image" in lbl:
            height_block = c
            break
    if height_block is None:
        # если нет явно высотного канала - берем первый
        height_block = channels[0]

    # вычисляем Z-scale (nm/LSB)
    z_scale = _parse_z_scale(header, height_block["block"])

    meta = {
        "data_offset": height_block.get("data_offset"),
        "data_length": height_block.get("data_length"),
        "samps": height_block.get("samps"),
        "lines": height_block.get("lines"),
        "bytes_per_pixel": height_block.get("bytes_per_pixel"),
        "scan_size_nm": height_block.get("scan_size_nm"),
        "z_scale_nm_per_lsb": z_scale,
        "channels": [c.get("image_label") for c in channels],
        "header_preview": header[:HEADER_PREVIEW_CHARS],
        "raw_header": header
    }
    return meta


def read_nanoscope_channel(path: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Считывает карту высот из файла Bruker Nanoscope, возвращает z (nm), pixel_size_nm и metadata.

    Steps:
      - парсит header (parse_nanoscope_header)
      - читает бинарные данные на основе data_offset/data_length
      - reshape -> (lines, samps)
      - умножает на z_scale (nm/LSB)
      - вычисляет pixel_size_nm = scan_size_nm / samps

    Raises
    ------
    ValueError если недостающие поля в header или бинарные данные короче ожидаемого.
    """
    meta = parse_nanoscope_header(path)

    # Проверяем наличие обязательных полей
    required = ("data_offset", "data_length", "samps", "lines", "bytes_per_pixel", "scan_size_nm")
    for k in required:
        if meta.get(k) is None:
            raise ValueError(f"В заголовке не хватает поля: {k}")

    data_offset = int(meta["data_offset"])
    data_length = int(meta["data_length"])
    samps = int(meta["samps"])
    lines = int(meta["lines"])
    bpp = int(meta["bytes_per_pixel"])

    # dtype: Bruker обычно хранит signed int16 (2 байта) или int32
    if bpp == 2:
        dtype = np.int16
    elif bpp == 4:
        dtype = np.int32
    else:
        # Попробуем угадать
        raise ValueError(f"Неожиданный bytes_per_pixel: {bpp}")

    filesize = os.path.getsize(path)
    if data_offset + data_length > filesize:
        raise ValueError("Файл короче, чем ожидается по Data offset/length")

    with open(path, "rb") as f:
        f.seek(data_offset)
        raw = f.read(data_length)

    arr = np.frombuffer(raw, dtype=dtype)
    expected = lines * samps
    if arr.size < expected:
        raise ValueError(f"Бинарных значений меньше, чем ожидается: {arr.size} < {expected}")

    arr = arr[:expected].reshape((lines, samps)).astype(np.float32)

    # Преобразуем в нанометры
    z = arr * float(meta["z_scale_nm_per_lsb"])

    # вычисляем pixel size
    pixel_size_nm = float(meta["scan_size_nm"]) / float(samps)

    # Пополнить метаданные
    meta_out = dict(meta)
    meta_out.update({
        "reader": "custom",
        "file_path": path,
        "pixel_size_nm": pixel_size_nm,
        "data_offset": data_offset,
        "data_length": data_length,
    })

    return z, pixel_size_nm, meta_out


def read_nanoscope_fallback(path: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Попытки считать SPM-файл несколькими способами:
      1) custom parser (read_nanoscope_channel)
      2) nanoscope (если установлен)
      3) pySPM (если установлен)
    Возвращает (z, pixel_size_nm, meta)
    """
    # 1) custom parser
    try:
        z, px, meta = read_nanoscope_channel(path)
        meta["reader"] = "custom"
        return z, px, meta
    except Exception as e:
        logger.debug("custom parser failed: %s", e)

    # 2) nanoscope (jmarini)
    if HAS_NANOSCOPE:
        try:
            ns = ns_mod.Nanoscope(path)
            channels = ns.channels
            if not channels:
                raise RuntimeError("nanoscope: no channels")
            ch = next(iter(channels.values()))
            z = ch.data.astype(np.float32)
            px = float(ch.scan_size) / float(z.shape[1])
            meta = {"reader": "nanoscope", "scan_size_nm": float(ch.scan_size)}
            logger.info("read with nanoscope (jmarini)")
            return z, px, meta
        except Exception as e:
            logger.debug("nanoscope reader failed: %s", e)

    # 3) pySPM
    if HAS_PYSPM:
        try:
            spm = pyspm_mod.Bruker(path)
            channels = spm.list_channels()
            if not channels:
                raise RuntimeError("pySPM: no channels")
            ch0 = channels[0]
            z = spm.get_channel(ch0).data.astype(np.float32)
            scan_size_nm = float(spm.info.get("scan_size", [np.nan])[0])
            px = float(scan_size_nm) / float(z.shape[1]) if scan_size_nm and not np.isnan(scan_size_nm) else np.nan
            meta = {"reader": "pySPM", "scan_size_nm": scan_size_nm, "channel": ch0}
            logger.info("read with pySPM")
            return z * 1.0, px, meta  # pySPM обычно возвращает в метрах? уточняйте в вашей версии
        except Exception as e:
            logger.debug("pySPM reader failed: %s", e)

    raise RuntimeError("Все методы чтения файла не сработали.")


# -------------------------
# Удобный единый интерфейс
# -------------------------
def load_afm(file_path: str, fmt: Optional[str] = None) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Унифицированная загрузка AFM-данных.
    Возвращает (z_map_nm, pixel_size_nm, metadata).

    Parameters
    ----------
    file_path : str
        Путь к файлу (.000, .001, .spm, .npy)
    fmt : Optional[str]
        Явно указать формат: "spm", "npy", "synthetic". Если None — определяется по расширению.

    Notes
    -----
    - Для .npy ожидается уже готовая карта (возвращается как float32).
    - Для spm/.001/.000 — используется custom parser и fallback.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # synthetic (для тестирования)
    if fmt == "synthetic" or ext == ".synthetic":
        z, px, meta = make_synthetic_afm()
        return z, px, meta

    # numpy
    if fmt == "npy" or ext == ".npy":
        arr = np.load(file_path)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        meta = {"reader": "npy", "file_path": file_path, "pixel_size_nm": 1.0}
        return arr, meta.get("pixel_size_nm", 1.0), meta

    # Bruker/SPM-like
    if fmt == "spm" or ext in (".spm", ".001", ".000"):
        try:
            return read_nanoscope_fallback(file_path)
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить SPM-файл: {e}")

    raise ValueError(f"Unsupported AFM format or unknown extension: {ext}")


# -------------------------
# Синтетика (для тестов)
# -------------------------
def make_synthetic_afm(shape: Tuple[int, int] = DEFAULT_SYNTH_SHAPE,
                       n_particles: int = DEFAULT_SYNTH_PARTICLES,
                       seed: Optional[int] = 7) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Генерирует синтетическую карту высот (гауссовы бугры + наклон + шум).
    Возвращает (z_map_nm, pixel_size_nm, meta)
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape
    z = np.zeros(shape, dtype=np.float32)

    yy, xx = np.mgrid[0:rows, 0:cols]
    ys = rng.integers(0, rows, size=n_particles)
    xs = rng.integers(0, cols, size=n_particles)
    hs = rng.uniform(SYNTH_MIN_HEIGHT_NM, SYNTH_MAX_HEIGHT_NM, size=n_particles)
    rs = rng.uniform(SYNTH_MIN_RADIUS_PX, SYNTH_MAX_RADIUS_PX, size=n_particles)

    for y, x, h, r in zip(ys, xs, hs, rs):
        z += h * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * r ** 2))

    # наклон + шум
    z += SYNTH_TILT_X * xx + SYNTH_TILT_Y * yy
    z += rng.normal(0, SYNTH_NOISE_STD_NM, size=shape).astype(np.float32)

    meta = {"reader": "synthetic", "n_particles": n_particles}
    return z.astype(np.float32), 1.0, meta


# -------------------------
# При запуске как скрипт — короткая демонстрация
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Custom AFM parser demo")
    parser.add_argument("path", help="Path to .000/.001/.spm/.npy file")
    parser.add_argument("--plot", action="store_true", help="Show image with matplotlib")
    args = parser.parse_args()

    p = args.path
    try:
        z, px, meta = load_afm(p)
        logger.info("Loaded %s -> shape=%s, pixel_size_nm=%.4f", p, z.shape, px)
        logger.info("Metadata keys: %s", list(meta.keys()))
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(z, cmap="afmhot", extent=nm_extent(z, px))
                ax.set_title(os.path.basename(p))
                ax.set_xlabel("x (nm)")
                ax.set_ylabel("y (nm)")
                plt.colorbar(im, ax=ax, label="height (nm)")
                plt.show()
            except Exception as e:
                logger.error("Не удалось построить график: %s", e)
    except Exception as exc:
        logger.exception("Ошибка при загрузке: %s", exc)