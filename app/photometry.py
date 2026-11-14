"""FITS 帧孔径测光与目标跟踪。

流程
----
1. 读取 ZIP 中 10～30 个同尺寸二维 FITS 帧,校验 MJD-OBS、EXPTIME;
2. 以 ``MJD-OBS + EXPTIME / 2 / 86400`` 为曝光中点,按升序排列;
3. 从首帧给定坐标出发,在「允许帧间位移」范围内寻找可信目标;
4. 每帧用圆孔径内像素总量减去背景环中位数 × 孔径有效像素数得到通量;
5. 目标丢失、出现两个可信位置、孔径/背景环越界时,该帧记坏并说明原因,
   下一帧仍从最后一个可信位置继续跟踪;
6. 有效帧不足 10 个或有效帧曝光中点重复时,不给出可下载序列。

坐标约定:x 为列号、y 为行号,均从 0 起;像素中心位于整数坐标。
"""
from __future__ import annotations

import gzip
import io
import math
import zipfile
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from astropy.time import Time

MIN_FRAMES = 10
MAX_FRAMES = 30
MIN_VALID_FRAMES = 10

FITS_SUFFIXES = (".fits", ".fit", ".fts",
                 ".fits.gz", ".fit.gz", ".fts.gz")

# ZIP 防御性上限(解压前/解压后)
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 400 * 1024 * 1024
MAX_TOTAL_PIXELS = 32_000_000   # 30 帧合计像素数上限(约 256 MB float64)

# 峰值检测与可信性参数
DETECT_SIGMA = 5.0          # 平滑图上的检测阈值(以平滑后噪声 sigma 计)
APERTURE_SIGMA = 5.0        # 孔径通量可信阈值
MERGE_DISTANCE = 2.0        # 小于该距离的局部极大值视为同一颗星
DUP_TIME_TOL_DAY = 1e-8     # 曝光中点重复判定容差(约 0.9 毫秒)
MIN_BACKGROUND_PIXELS = 5   # 背景环至少需要的像素数
FLUX_RATIO_LIMIT = 2.0      # 候选通量与目标近期中位数的最大容许比值(兜底)
MAX_TRACK_HISTORY = 5       # 用于判断「疑似邻星」的近期有效通量帧数


class PhotometryInputError(ValueError):
    """ZIP/FITS 或参数不合法,作为 HTTP 400 返回。"""


@dataclass
class Frame:
    """一帧的元数据、逐帧处理状态与测光结果。

    图像数据不常驻内存:处理时由 :class:`FrameSet` 按需读取单帧。
    """

    filename: str
    mjd_obs: float
    exptime: float
    shape: tuple[int, int]
    source_index: int
    midpoint: float = 0.0
    data: np.ndarray | None = field(default=None, repr=False)
    midpoint: float = 0.0
    order: int = -1
    status: str = "ok"                 # ok / bad
    reason: str | None = None          # 中文原因
    reason_code: str | None = None
    x: float | None = None
    y: float | None = None
    flux: float | None = None
    flux_err: float | None = None
    snr: float | None = None
    background: float | None = None
    background_sigma: float | None = None
    n_aperture: int = 0
    n_background: int = 0
    n_candidates: int = 0
    candidates: list[dict] = field(default_factory=list)

    @property
    def time_day(self) -> float:
        return self.midpoint

    def mark_bad(self, code: str, reason: str) -> None:
        self.status = "bad"
        self.reason_code = code
        self.reason = reason

    def as_dict(self) -> dict:
        d = {
            "index": self.order,
            "filename": self.filename,
            "mjd_obs": self.mjd_obs,
            "exptime": self.exptime,
            "time_day": self.midpoint,
            "status": self.status,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "x": self.x,
            "y": self.y,
            "flux": self.flux,
            "flux_err": self.flux_err,
            "snr": self.snr,
            "background": self.background,
            "background_sigma": self.background_sigma,
            "n_aperture": self.n_aperture,
            "n_background": self.n_background,
            "n_candidates": self.n_candidates,
            "candidates": self.candidates,
        }
        return d


# --------------------------------------------------------------------------- #
# ZIP / FITS 读取与校验
# --------------------------------------------------------------------------- #

def _is_fits_member(name: str) -> bool:
    if name.endswith("/") or "__MACOSX" in name:
        return False
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in FITS_SUFFIXES)


def _header_number(header, key: str, filename: str) -> float:
    """读取数值头;MJD-OBS 也允许 ISO 时间字符串。"""
    value = header.get(key)
    if value is None:
        raise PhotometryInputError(f"{filename}:缺少必需头 {key}")
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            try:
                number = float(Time(text).mjd)
            except Exception as exc:
                raise PhotometryInputError(
                    f"{filename}:头 {key} 无法解析为时间({text!r})") from exc
    else:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise PhotometryInputError(
                f"{filename}:头 {key} 不是数值") from exc
    if not math.isfinite(number):
        raise PhotometryInputError(f"{filename}:头 {key} 不是有限数值")
    return number


def _decompress_member(member_name: str, raw: bytes) -> bytes:
    """成员若为 .gz,在内存里解压一次;其余原样返回。"""
    if not member_name.lower().endswith(".gz"):
        return raw
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            raw = gz.read(MAX_MEMBER_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise PhotometryInputError(
            f"{member_name}:gzip 解压失败({exc})") from exc
    if len(raw) > MAX_MEMBER_BYTES:
        raise PhotometryInputError(f"{member_name}:解压后超过大小上限")
    return raw


def _image_hdu(hdul: fits.HDUList):
    """返回第一幅二维图像 HDU(没有则 None),优先按头判断以免解码无关 HDU。"""
    for hdu in hdul:
        if hdu.header.get("NAXIS", 0) == 2:
            return hdu
    for hdu in hdul:   # 压缩图像(CompImageHDU)头里 NAXIS 可能为 0
        data = hdu.data
        if data is not None and np.ndim(data) == 2:
            return hdu
    return None


def _read_fits(member_name: str, raw: bytes) -> tuple[np.ndarray, fits.Header]:
    """从内存字节读取第一幅二维图像,返回 (float64 数据, 对应头)。"""
    raw = _decompress_member(member_name, raw)
    try:
        with fits.open(io.BytesIO(raw), memmap=False) as hdul:
            hdu = _image_hdu(hdul)
            if hdu is None:
                raise PhotometryInputError(f"{member_name}:没有二维图像数据")
            arr = np.array(hdu.data, dtype=np.float64, copy=True)
            header = hdu.header.copy()
            if "MJD-OBS" not in header:
                header = hdul[0].header.copy()
    except PhotometryInputError:
        raise
    except Exception as exc:
        raise PhotometryInputError(
            f"{member_name}:FITS 解析失败({exc})") from exc
    # 非有限像素(如缺测、饱和标记)按图像中位数填补
    finite = np.isfinite(arr)
    if not finite.all():
        fill = float(np.median(arr[finite])) if finite.any() else 0.0
        arr[~finite] = fill
    return arr, header


def _read_header(member_name: str, raw: bytes) -> fits.Header:
    """只解析头(不加载像素),并检查存在二维图像。"""
    raw = _decompress_member(member_name, raw)
    try:
        with fits.open(io.BytesIO(raw), memmap=False) as hdul:
            hdu = _image_hdu(hdul)
            if hdu is None:
                raise PhotometryInputError(f"{member_name}:没有二维图像数据")
            header = hdu.header.copy()
            # 扩展 HDU 的头不含 NAXIS1/2 之外的观测信息时,回退到主头
            if "MJD-OBS" not in header:
                main = hdul[0].header
                for key in ("MJD-OBS", "EXPTIME"):
                    if key in main and key not in header:
                        header[key] = main[key]
    except PhotometryInputError:
        raise
    except Exception as exc:
        raise PhotometryInputError(
            f"{member_name}:FITS 解析失败({exc})") from exc
    return header


def _image_shape(header: fits.Header, member_name: str) -> tuple[int, int]:
    try:
        shape = header["NAXIS2"], header["NAXIS1"]
        ny, nx = int(shape[0]), int(shape[1])
    except (KeyError, TypeError, ValueError) as exc:
        raise PhotometryInputError(
            f"{member_name}:无法读取图像尺寸") from exc
    if ny <= 0 or nx <= 0:
        raise PhotometryInputError(f"{member_name}:图像尺寸非法")
    return ny, nx


def _open_zip(raw: bytes) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(io.BytesIO(raw))
    except (zipfile.BadZipFile, OSError) as exc:
        raise PhotometryInputError(f"ZIP 无法打开:{exc}") from exc


def _member_bytes(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    try:
        with zf.open(info, "r") as member:
            raw = member.read(MAX_MEMBER_BYTES + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise PhotometryInputError(
            f"{info.filename}:ZIP 成员读取失败({exc})") from exc
    if len(raw) > MAX_MEMBER_BYTES:
        raise PhotometryInputError(f"{info.filename}:文件超过大小上限")
    return raw


class FrameSet:
    """一批已校验的帧;图像数据按需从 ZIP 读取,避免整批常驻内存。

    调用顺序:先 ``scan()``(只读头与尺寸),再 ``process()``(逐帧读图)。
    """

    def __init__(self, raw: bytes) -> None:
        self._zip_bytes = raw
        self.frames: list[Frame] = []
        self.width = 0
        self.height = 0

    # -- 第一遍:只读头,校验数量、尺寸、时间 ------------------------------ #
    def scan(self) -> "FrameSet":
        with _open_zip(self._zip_bytes) as zf:
            infos = [info for info in zf.infolist()
                     if _is_fits_member(info.filename)]
            if len(infos) < MIN_FRAMES:
                raise PhotometryInputError(
                    f"至少需要 {MIN_FRAMES} 个 FITS 文件,当前 {len(infos)} 个")
            if len(infos) > MAX_FRAMES:
                raise PhotometryInputError(
                    f"最多允许 {MAX_FRAMES} 个 FITS 文件,当前 {len(infos)} 个")

            total = 0
            for info in infos:
                if info.file_size > MAX_MEMBER_BYTES:
                    raise PhotometryInputError(
                        f"{info.filename}:单个文件超过 "
                        f"{MAX_MEMBER_BYTES // 2**20} MB")
                total += info.file_size
            if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise PhotometryInputError("ZIP 解压后总体积超过上限")

            for source_index, info in enumerate(infos):
                raw_member = _member_bytes(zf, info)
                header = _read_header(info.filename, raw_member)
                del raw_member
                mjd_obs = _header_number(header, "MJD-OBS", info.filename)
                exptime = _header_number(header, "EXPTIME", info.filename)
                if exptime <= 0:
                    raise PhotometryInputError(
                        f"{info.filename}:EXPTIME 必须为正数(当前 {exptime})")
                shape = _image_shape(header, info.filename)
                self.frames.append(Frame(
                    filename=info.filename,
                    mjd_obs=mjd_obs,
                    exptime=exptime,
                    midpoint=mjd_obs + exptime / 2.0 / 86400.0,
                    shape=shape,
                    source_index=source_index,
                ))

        shapes = {frame.shape for frame in self.frames}
        if len(shapes) != 1:
            detail = ";".join(
                f"{frame.filename}:{frame.shape}" for frame in self.frames[:6])
            raise PhotometryInputError(
                f"所有 FITS 帧必须同尺寸,实际出现多种尺寸:{detail}")
        self.height, self.width = self.frames[0].shape
        if self.height < 3 or self.width < 3:
            raise PhotometryInputError("FITS 图像尺寸过小")
        total_pixels = sum(f.shape[0] * f.shape[1] for f in self.frames)
        if total_pixels > MAX_TOTAL_PIXELS:
            raise PhotometryInputError(
                "全部 FITS 帧的像素总数超过上限,请缩小图像或帧数")

        self.frames.sort(key=lambda f: (f.midpoint, f.source_index))
        for order, frame in enumerate(self.frames):
            frame.order = order
        return self

    # -- 第二遍:逐帧读图并处理 --------------------------------------------- #
    def load_frame_data(self, frame: Frame) -> np.ndarray:
        # ZIP 在本次调用内打开/关闭,数据只在这一帧处理期间存在
        with _open_zip(self._zip_bytes) as zf:
            try:
                info = zf.getinfo(frame.filename)
            except KeyError as exc:
                raise PhotometryInputError(
                    f"{frame.filename}:ZIP 中找不到该成员") from exc
            raw_member = _member_bytes(zf, info)
        data, _header = _read_fits(frame.filename, raw_member)
        if data.shape != frame.shape:
            raise PhotometryInputError(
                f"{frame.filename}:图像尺寸与扫描阶段不一致")
        frame.data = data
        return data

    def process(self, x0: float, y0: float, aperture_radius: float,
                bg_inner_radius: float, bg_outer_radius: float,
                max_shift: float) -> dict:
        return process_frames(self, x0, y0, aperture_radius,
                              bg_inner_radius, bg_outer_radius, max_shift)


def load_zip(raw: bytes) -> FrameSet:
    """扫描并校验 ZIP,返回可逐帧处理的 FrameSet。"""
    return FrameSet(raw).scan()


def validate_params(x0: float, y0: float, aperture_radius: float,
                    bg_inner_radius: float, bg_outer_radius: float,
                    max_shift: float) -> None:
    values = {
        "x": x0, "y": y0, "孔径半径": aperture_radius,
        "背景环内半径": bg_inner_radius, "背景环外半径": bg_outer_radius,
        "允许帧间位移": max_shift,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise PhotometryInputError(f"{name} 必须为有限数值")
    if aperture_radius <= 0:
        raise PhotometryInputError("孔径半径必须大于 0")
    if bg_inner_radius <= aperture_radius:
        raise PhotometryInputError("背景环内半径必须大于孔径半径")
    if bg_outer_radius <= bg_inner_radius:
        raise PhotometryInputError("背景环外半径必须大于内半径")
    if max_shift < 0:
        raise PhotometryInputError("允许帧间位移不能为负数")


# --------------------------------------------------------------------------- #
# 数值工具
# --------------------------------------------------------------------------- #

def _robust_stats(data: np.ndarray) -> tuple[float, float]:
    """抽样估计天空背景中位数与噪声 sigma(1.4826×MAD)。"""
    flat = np.asarray(data, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0, 1.0
    stride = max(1, flat.size // 200_000)
    sample = flat[::stride]
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    sigma = 1.4826 * mad
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(sample))
    if not math.isfinite(sigma) or sigma <= 0.0:
        scale = max(abs(med), 1.0)
        sigma = 1e-12 * scale
    return med, sigma


def _box3_mean(arr: np.ndarray) -> np.ndarray:
    """3×3 盒式平滑,边缘按最近像素延拓。"""
    p = np.pad(arr, 1, mode="edge")
    total = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
             + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
             + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])
    return total / 9.0


def _bounds_ok(x: float, y: float, r_out: float,
               width: int, height: int) -> bool:
    """圆心在 (x, y) 的外环是否完整落在图像内(像素中心坐标 0..N-1)。"""
    return (x - r_out >= -0.5 and x + r_out <= width - 0.5
            and y - r_out >= -0.5 and y + r_out <= height - 0.5)


def _measure(data: np.ndarray, x: float, y: float, r_ap: float,
             r_in: float, r_out: float) -> dict:
    """在 (x, y) 处做孔径测光。调用前应已通过 _bounds_ok。"""
    height, width = data.shape
    x_lo = max(0, int(math.floor(x - r_out)))
    x_hi = min(width - 1, int(math.ceil(x + r_out)))
    y_lo = max(0, int(math.floor(y - r_out)))
    y_hi = min(height - 1, int(math.ceil(y + r_out)))

    yy, xx = np.mgrid[y_lo:y_hi + 1, x_lo:x_hi + 1]
    rr2 = (xx - x) ** 2 + (yy - y) ** 2
    ap_mask = rr2 <= r_ap * r_ap
    ring_mask = (rr2 > r_in * r_in) & (rr2 <= r_out * r_out)

    n_ap = int(np.count_nonzero(ap_mask))
    n_ring = int(np.count_nonzero(ring_mask))
    if n_ap == 0 or n_ring < MIN_BACKGROUND_PIXELS:
        return {"error": "背景环或孔径可用像素不足",
                "n_aperture": n_ap, "n_background": n_ring}

    values = data[y_lo:y_hi + 1, x_lo:x_hi + 1]
    ap_values = values[ap_mask]
    ring_values = values[ring_mask]
    bg = float(np.median(ring_values))
    bg_mad = float(np.median(np.abs(ring_values - bg)))
    bg_sigma = 1.4826 * bg_mad
    if not math.isfinite(bg_sigma) or bg_sigma <= 0.0:
        bg_sigma = float(np.std(ring_values))
    if not math.isfinite(bg_sigma) or bg_sigma <= 0.0:
        bg_sigma = 0.0

    ap_sum = float(np.sum(ap_values))
    flux = ap_sum - bg * n_ap
    # 孔径内噪声 + 背景中位数估计的误差(1.5708 = (π/2) 的中位数方差因子)
    var = (n_ap * bg_sigma ** 2
           + 1.5708 * (n_ap ** 2) * (bg_sigma ** 2) / max(n_ring, 1))
    flux_err = math.sqrt(max(var, 0.0))
    # 环内完全无散度时(如极端仿真图),S/N 用一个大而有限的数表示,
    # 避免 None/inf 进入 JSON
    snr = flux / flux_err if flux_err > 0 else (
        1e6 if flux > 0 else 0.0)

    return {
        "x": float(x), "y": float(y),
        "flux": flux, "flux_err": flux_err, "snr": snr,
        "background": bg, "background_sigma": bg_sigma,
        "n_aperture": n_ap, "n_background": n_ring,
    }


# --------------------------------------------------------------------------- #
# 目标检测与跟踪
# --------------------------------------------------------------------------- #

def _detect_peaks(data: np.ndarray, anchor_x: float, anchor_y: float,
                  max_shift: float) -> list[dict]:
    """在最后可信位置附近检测显著局部极大值。返回亚像素质心列表。"""
    height, width = data.shape
    x_lo = max(0, int(math.floor(anchor_x - max_shift)) - 2)
    x_hi = min(width - 1, int(math.ceil(anchor_x + max_shift)) + 2)
    y_lo = max(0, int(math.floor(anchor_y - max_shift)) - 2)
    y_hi = min(height - 1, int(math.ceil(anchor_y + max_shift)) + 2)
    crop = data[y_lo:y_hi + 1, x_lo:x_hi + 1]
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return []

    smoothed = _box3_mean(crop)
    med, sigma = _robust_stats(data)
    sigma_sm = sigma / 3.0          # 3×3 盒式平滑对白噪声的抑制
    threshold = med + DETECT_SIGMA * sigma_sm

    # 局部极大值(含平台边),阈值以上
    s = smoothed
    core = s[1:-1, 1:-1]
    is_max = np.ones(core.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            is_max &= core >= s[1 + dy:s.shape[0] - 1 + dy,
                               1 + dx:s.shape[1] - 1 + dx]
    is_max &= core > threshold
    py, px = np.nonzero(is_max)
    py = py + 1
    px = px + 1

    peaks: list[dict] = []
    sm_med = float(np.median(s))
    for iy, ix in zip(py.tolist(), px.tolist()):
        # 3×3 平滑图上的亮权重质心,得到亚像素位置
        wy = slice(max(0, iy - 1), min(s.shape[0], iy + 2))
        wx = slice(max(0, ix - 1), min(s.shape[1], ix + 2))
        patch = s[wy, wx] - sm_med
        w = np.clip(patch, 0.0, None)
        total = float(w.sum())
        if total > 0.0:
            gy, gx = np.mgrid[wy.start:wy.stop, wx.start:wx.stop]
            cx = float((gx * w).sum() / total) + x_lo
            cy = float((gy * w).sum() / total) + y_lo
        else:
            cx, cy = float(ix) + x_lo, float(iy) + y_lo

        dist = math.hypot(cx - anchor_x, cy - anchor_y)
        if dist > max_shift:
            continue
        peaks.append({"x": cx, "y": cy, "dist": dist,
                      "peak": float(s[iy, ix])})

    # 由亮到暗去重:同一颗星在平台/噪声上可能产生多个相邻极大值
    peaks.sort(key=lambda p: p["peak"], reverse=True)
    kept: list[dict] = []
    for peak in peaks:
        if all(math.hypot(peak["x"] - q["x"], peak["y"] - q["y"])
               >= MERGE_DISTANCE for q in kept):
            kept.append(peak)
    return kept


def _recent_flux(history: list[float]) -> float | None:
    if not history:
        return None
    return float(np.median(np.asarray(history[-MAX_TRACK_HISTORY:],
                                      dtype=float)))


def _neighbor_like(flux: float, recent: float | None) -> bool:
    """候选通量与近期目标通量差异过大时,视为邻星而非目标。"""
    if recent is None or recent <= 0.0 or flux <= 0.0:
        return False
    ratio = flux / recent
    return ratio > FLUX_RATIO_LIMIT or ratio < 1.0 / FLUX_RATIO_LIMIT


def _process_frame(frame: Frame, anchor: tuple[float, float],
                   r_ap: float, r_in: float, r_out: float,
                   max_shift: float, history: list[float]) -> None:
    """处理单帧:更新 frame 状态;成功时由调用方更新 anchor 与 history。"""
    height, width = frame.data.shape
    anchor_x, anchor_y = anchor
    peaks = _detect_peaks(frame.data, anchor_x, anchor_y, max_shift)
    frame.n_candidates = len(peaks)
    candidates: list[dict] = []

    for peak in peaks:
        x, y = peak["x"], peak["y"]
        info = {"x": x, "y": y, "dist": peak["dist"],
                "in_bounds": _bounds_ok(x, y, r_out, width, height),
                "flux": None, "flux_err": None, "snr": None,
                "background": None, "credible": False,
                "n_aperture": 0, "n_background": 0}
        if info["in_bounds"]:
            measured = _measure(frame.data, x, y, r_ap, r_in, r_out)
            if "error" in measured:
                info["in_bounds"] = False
                info["error"] = measured["error"]
                info["n_aperture"] = measured["n_aperture"]
                info["n_background"] = measured["n_background"]
            else:
                for key, value in measured.items():
                    info[key] = value
                info["credible"] = bool(
                    measured["flux"] > 0.0
                    and measured["snr"] >= APERTURE_SIGMA)
        candidates.append(info)

    frame.candidates = candidates

    credible = [c for c in candidates if c["credible"]]
    oob = [c for c in candidates if not c["in_bounds"]]
    nearest_oob = min(oob, key=lambda c: c["dist"], default=None)
    nearest_credible = min(credible, key=lambda c: c["dist"], default=None)

    # 目标本身越界:最近候选(按到最后可信位置的距离)已无法完整放置背景环,
    # 比它更远的「可信」候选多半是邻星,不能拿来顶替目标。
    if (nearest_oob is not None
            and (nearest_credible is None
                 or nearest_oob["dist"] < nearest_credible["dist"])):
        c = nearest_oob
        detail = c.get("error") or "孔径或背景环超出图像范围"
        frame.mark_bad(
            "out_of_bounds",
            f"目标位置 ({c['x']:.2f}, {c['y']:.2f}) 处{detail};"
            f"孔径半径 {r_ap:g},背景环 {r_in:g}~{r_out:g},"
            f"图像 {width}×{height}(x 为列、y 为行)")
        return

    # 两个(或更多)可信位置:不能猜哪个是目标,也不把邻星当目标
    if len(credible) >= 2:
        parts = []
        for c in credible[:4]:
            parts.append(
                f"({c['x']:.2f}, {c['y']:.2f}) 通量 {c['flux']:.1f}"
                f"(S/N {c['snr']:.1f})")
        frame.mark_bad(
            "two_positions",
            "出现两个可信位置:" + ";".join(parts)
            + ",无法唯一确定目标,已跳过本帧")
        return

    if len(credible) == 1:
        chosen = credible[0]
        recent = _recent_flux(history)
        if _neighbor_like(chosen["flux"], recent):
            frame.mark_bad(
                "neighbor_like",
                f"候选位置 ({chosen['x']:.2f}, {chosen['y']:.2f}) 通量 "
                f"{chosen['flux']:.1f} 与目标近期中位数 {recent:.1f} 相差超过 "
                f"{FLUX_RATIO_LIMIT:g} 倍,疑似邻星,已跳过本帧")
            return
        frame.status = "ok"
        frame.reason = None
        frame.reason_code = None
        frame.x = chosen["x"]
        frame.y = chosen["y"]
        frame.flux = chosen["flux"]
        frame.flux_err = chosen["flux_err"]
        frame.snr = chosen["snr"]
        frame.background = chosen["background"]
        frame.background_sigma = chosen["background_sigma"]
        frame.n_aperture = chosen["n_aperture"]
        frame.n_background = chosen["n_background"]
        return

    best = max(candidates, key=lambda c: c["snr"] or -math.inf,
               default=None)
    if best is not None:
        snr = best["snr"]
        snr_text = f"{snr:.1f}" if snr is not None else "—"
        frame.mark_bad(
            "target_lost",
            f"目标丢失:距最后可信位置 ({anchor_x:.2f}, {anchor_y:.2f}) "
            f"±{max_shift:g} 像素内未找到可信目标"
            f"(最亮候选 S/N {snr_text} < {APERTURE_SIGMA:g})")
    elif not _bounds_ok(anchor_x, anchor_y, r_out, width, height):
        # 首帧给定坐标本身就放不下背景环(后续 anchor 一定在界内)
        frame.mark_bad(
            "out_of_bounds",
            f"首帧目标坐标 ({anchor_x:.2f}, {anchor_y:.2f}) 处"
            f"孔径或背景环超出图像范围;孔径半径 {r_ap:g},"
            f"背景环 {r_in:g}~{r_out:g},图像 {width}×{height}"
            f"(x 为列、y 为行)")
    else:
        frame.mark_bad(
            "target_lost",
            f"目标丢失:距最后可信位置 ({anchor_x:.2f}, {anchor_y:.2f}) "
            f"±{max_shift:g} 像素内未找到可信目标")


# --------------------------------------------------------------------------- #
# 入口:排序、逐帧跟踪、生成序列
# --------------------------------------------------------------------------- #

def process_frames(frameset: FrameSet, x0: float, y0: float,
                   aperture_radius: float, bg_inner_radius: float,
                   bg_outer_radius: float, max_shift: float) -> dict:
    """按曝光中点顺序逐帧读图、跟踪目标,返回可 JSON 序列化的结果。"""
    frames = frameset.frames
    height, width = frameset.height, frameset.width
    if not (0.0 <= x0 <= width - 1 and 0.0 <= y0 <= height - 1):
        raise PhotometryInputError(
            f"首帧目标坐标 ({x0:g}, {y0:g}) 超出图像范围 "
            f"x∈[0,{width - 1}],y∈[0,{height - 1}](x 为列、y 为行)")

    anchor = (float(x0), float(y0))
    history: list[float] = []
    for frame in frames:
        # 一次只加载一帧,处理完立即释放,整批图像不常驻内存
        frame.data = frameset.load_frame_data(frame)
        _process_frame(frame, anchor, aperture_radius, bg_inner_radius,
                       bg_outer_radius, max_shift, history)
        frame.data = None
        if frame.status == "ok":
            anchor = (frame.x, frame.y)
            history.append(frame.flux)

    valid = [f for f in frames if f.status == "ok"]
    errors: list[str] = []
    if len(valid) < MIN_VALID_FRAMES:
        errors.append(
            f"有效帧只有 {len(valid)} 帧,少于 {MIN_VALID_FRAMES} 帧,"
            "不提供光变序列")

    duplicates = _duplicate_time_groups(valid)
    if duplicates:
        groups = ";".join(
            ", ".join(f"{f.filename}({f.midpoint:.9f})" for f in group)
            for group in duplicates)
        errors.append(f"有效帧的曝光中点重复:{groups};不提供光变序列")

    sequence_available = not errors
    sequence: list[dict] = []
    csv_text: str | None = None
    if sequence_available:
        sequence = [{"time_day": f.midpoint, "flux": f.flux} for f in valid]
        lines = ["time_day,flux"]
        lines += [f"{f.midpoint:.12f},{f.flux:.8g}" for f in valid]
        csv_text = "\n".join(lines) + "\n"

    reason_counts: dict[str, int] = {}
    for frame in frames:
        if frame.status == "bad" and frame.reason_code:
            reason_counts[frame.reason_code] = (
                reason_counts.get(frame.reason_code, 0) + 1)

    return {
        "sequence_available": sequence_available,
        "sequence_error": ";".join(errors) if errors else None,
        "errors": errors,
        "n_frames": len(frames),
        "n_valid": len(valid),
        "n_bad": len(frames) - len(valid),
        "image_width": width,
        "image_height": height,
        "params": {
            "x": x0, "y": y0,
            "aperture_radius": aperture_radius,
            "bg_inner_radius": bg_inner_radius,
            "bg_outer_radius": bg_outer_radius,
            "max_shift": max_shift,
        },
        "frames": [f.as_dict() for f in frames],
        "bad_frames": [f.as_dict() for f in frames if f.status == "bad"],
        "reason_counts": reason_counts,
        "sequence": sequence,
        "csv": csv_text,
    }


def _duplicate_time_groups(frames: list[Frame]) -> list[list[Frame]]:
    """按曝光中点分组,返回重复(容差 DUP_TIME_TOL_DAY)的组。"""
    ordered = sorted(frames, key=lambda f: f.midpoint)
    groups: list[list[Frame]] = []
    current: list[Frame] = []
    for frame in ordered:
        if (current
                and abs(frame.midpoint - current[-1].midpoint)
                <= DUP_TIME_TOL_DAY):
            current.append(frame)
        else:
            if len(current) > 1:
                groups.append(current)
            current = [frame]
    if len(current) > 1:
        groups.append(current)
    return groups


# --------------------------------------------------------------------------- #
# 结果写盘(供现有周期筛选入口使用)
# --------------------------------------------------------------------------- #

def write_sequence_csv(result: dict, path) -> None:
    """把已生成的光变序列写入 CSV(time_day,flux)。"""
    csv_text = result.get("csv")
    if not csv_text:
        raise ValueError("没有可写出的光变序列")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(csv_text)
