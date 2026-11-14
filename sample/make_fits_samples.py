"""生成虚构 FITS 巡测样例(两个 ZIP)。

每个 ZIP 含 24 帧 128×128 二维图像,帧头写 MJD-OBS 与 EXPTIME。
* photometry_neighbor.zip:目标东侧 9 像素处有一颗邻星,第 9～16 帧邻星变亮,
  在允许位移 10 像素内与目标同时成为可信位置,这些帧被逐帧记坏(不猜、不把
  邻星当目标);其余帧邻星很暗,目标正常测光,最终仍能给出光变序列;
* photometry_loss.zip:目标在第 9～11 帧短暂消失,其余帧正常,用于演示
  「跳过坏帧后仍从最后可信位置继续跟踪」并得到可周期筛查的序列。

目标亮度按约 3.1 天的周期变化;背景、读出噪声与泊松噪声均为虚构。
固定随机种子,结果可复现。
"""
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import numpy as np
from astropy.io import fits

OUT_DIR = Path(__file__).resolve().parent
WIDTH = HEIGHT = 128
N_FRAMES = 24
EXPTIME = 300.0
MJD0 = 60000.0
CADENCE_DAY = 0.35           # 帧间约 0.35 天,总基线约 8 天
PERIOD_DAY = 3.1
TARGET_X, TARGET_Y = 60.0, 62.0
NEIGHBOR_OFFSET = 9.0
NEIGHBOR_FLARE_FRAMES = range(8, 16)   # 邻星在这几帧变亮(0 起,含两端)
BG_LEVEL = 200.0
READ_NOISE = 6.0
PSF_SIGMA = 1.30
SEED = 20261114


def psf(image: np.ndarray, x: float, y: float, total_flux: float,
        sigma: float = PSF_SIGMA) -> None:
    """把高斯点源(总通量 total_flux)叠加到图像上。"""
    half = int(math.ceil(4.0 * sigma))
    x_lo = max(0, int(math.floor(x)) - half)
    x_hi = min(WIDTH - 1, int(math.ceil(x)) + half)
    y_lo = max(0, int(math.floor(y)) - half)
    y_hi = min(HEIGHT - 1, int(math.ceil(y)) + half)
    yy, xx = np.mgrid[y_lo:y_hi + 1, x_lo:x_hi + 1]
    kernel = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    image[y_lo:y_hi + 1, x_lo:x_hi + 1] += total_flux * kernel


def make_frame(rng: np.random.Generator, target_flux: float,
               target_x: float, target_y: float,
               neighbor_flux: float | None,
               neighbor_x: float, neighbor_y: float) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH), BG_LEVEL, dtype=np.float64)
    image += rng.normal(0.0, READ_NOISE, size=image.shape)
    if neighbor_flux:
        psf(image, neighbor_x, neighbor_y, neighbor_flux)
    if target_flux > 0:
        psf(image, target_x, target_y, target_flux)
    # 光子噪声:以背景+星光的泊松涨落为主
    image = rng.poisson(np.clip(image, 0.0, None)).astype(np.float64)
    return image


def write_zip(path: Path, frames: list[tuple[np.ndarray, float]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, (image, mjd_obs) in enumerate(frames, start=1):
            hdu = fits.PrimaryHDU(image.astype(np.float32))
            hdu.header["MJD-OBS"] = (mjd_obs, "exposure start (MJD)")
            hdu.header["EXPTIME"] = (EXPTIME, "exposure seconds")
            hdu.header["BUNIT"] = ("ADU", "fictional counts")
            buf = io.BytesIO()
            hdu.writeto(buf, output_verify="ignore")
            # 固定成员时间戳,使重新生成的文件逐字节可复现
            info = zipfile.ZipInfo(f"frame_{i:02d}.fits",
                                   date_time=(2026, 11, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, buf.getvalue())
    print(f"已写出 {path}({len(frames)} 帧)")


def target_brightness(frame_index: int) -> float:
    phase = 2.0 * math.pi * frame_index * CADENCE_DAY / PERIOD_DAY
    return 42_000.0 * (1.0 + 0.30 * math.sin(phase))


def make_neighbor_zip() -> None:
    rng = np.random.default_rng(SEED)
    frames = []
    for i in range(N_FRAMES):
        mjd_obs = MJD0 + i * CADENCE_DAY
        x = TARGET_X + 0.8 * math.sin(2.0 * math.pi * i / 17.0)
        y = TARGET_Y + 0.6 * math.cos(2.0 * math.pi * i / 13.0)
        flux = target_brightness(i)
        # 邻星平时很暗;第 9～16 帧变亮到目标的 55%,进入搜索范围后
        # 与目标同时成为可信位置
        neighbor = 0.55 * flux if i in NEIGHBOR_FLARE_FRAMES else 0.012 * flux
        image = make_frame(rng, flux, x, y, neighbor,
                           TARGET_X + NEIGHBOR_OFFSET, TARGET_Y)
        frames.append((image, mjd_obs))
    write_zip(OUT_DIR / "photometry_neighbor.zip", frames)


def make_loss_zip() -> None:
    rng = np.random.default_rng(SEED + 1)
    frames = []
    for i in range(N_FRAMES):
        mjd_obs = MJD0 + i * CADENCE_DAY
        x = TARGET_X + 0.5 * math.sin(2.0 * math.pi * i / 23.0) - 0.02 * i
        y = TARGET_Y + 0.4 * math.cos(2.0 * math.pi * i / 19.0)
        flux = 0.0 if 8 <= i <= 10 else target_brightness(i)
        image = make_frame(rng, flux, x, y, None, 0.0, 0.0)
        frames.append((image, mjd_obs))
    write_zip(OUT_DIR / "photometry_loss.zip", frames)


if __name__ == "__main__":
    make_neighbor_zip()
    make_loss_zip()
