"""测光语义自检(不依赖 pytest,直接运行)。

    python tests/test_photometry.py

生成的 FITS 只存在于内存或临时目录,不会改动 sample/ 下的样例。
"""
from __future__ import annotations

import gzip
import io
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.photometry import (  # noqa: E402
    MIN_VALID_FRAMES, PhotometryInputError, load_zip, process_frames,
)

SHAPE = (40, 40)
STAR_X, STAR_Y = 20.0, 20.0
TARGET = 1500.0
NEIGHBOR = 700.0


def make_image(star: bool = True, x: float = STAR_X, y: float = STAR_Y,
               neighbor_at: tuple[float, float] | None = None) -> np.ndarray:
    rng = np.random.default_rng(1)
    image = rng.normal(100.0, 3.0, SHAPE)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    if star:
        image += TARGET * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 8.0)
    if neighbor_at is not None:
        nx, ny = neighbor_at
        image += NEIGHBOR * np.exp(-((xx - nx) ** 2 + (yy - ny) ** 2) / 8.0)
    return image


def fits_bytes(image: np.ndarray, mjd: float, exptime: float = 300.0,
               mjd_obs: bool = True) -> bytes:
    hdu = fits.PrimaryHDU(np.asarray(image, dtype=np.float32))
    if mjd_obs:
        hdu.header["MJD-OBS"] = mjd
    hdu.header["EXPTIME"] = exptime
    buf = io.BytesIO()
    hdu.writeto(buf)
    return buf.getvalue()


def zip_bytes(items: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in items:
            zf.writestr(name, payload)
    return buf.getvalue()


def run(items: list[tuple[str, bytes]], x: float = STAR_X, y: float = STAR_Y,
        aperture: float = 3.0, r_in: float = 5.0, r_out: float = 8.0,
        shift: float = 3.0) -> dict:
    frameset = load_zip(zip_bytes(items))
    return process_frames(frameset, x, y, aperture, r_in, r_out, shift)


def frames(mjd_of=lambda i: 60000.0 + 0.2 * i, image_of=lambda i: make_image(),
           n: int = 12) -> list[tuple[str, bytes]]:
    return [(f"f{i:02d}.fits", fits_bytes(image_of(i), mjd_of(i)))
            for i in range(n)]


def expect(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def test_sorted_by_exposure_midpoint() -> None:
    print("曝光中点排序")
    items = [(f"f{i:02d}.fits", fits_bytes(make_image(), 60000.0 + (11 - i) * 0.2))
             for i in range(12)]
    result = run(items)
    times = [f["time_day"] for f in result["frames"]]
    expect(times == sorted(times), "time_day 按曝光中点升序")
    expect(result["n_valid"] == 12 and result["sequence_available"],
           "12 帧全部有效并提供序列")
    midpoints = [60000.0 + (11 - k) * 0.2 + 150.0 / 86400.0
                 for k in range(12)]
    expect(all(abs(f["time_day"] - expected) < 1e-9
               for f, expected in zip(result["frames"], sorted(midpoints))),
           "time_day 为 MJD-OBS + EXPTIME/2 的曝光中点")


def test_neighbor_two_positions() -> None:
    print("邻星干扰")
    near = (STAR_X + 9.0, STAR_Y)
    items = frames(image_of=lambda i: make_image(neighbor_at=near), n=14)
    wide = run(items, shift=10.0)
    expect(wide["n_valid"] == 0 and wide["reason_counts"].get("two_positions") == 14,
           "位移 10 像素时每帧都出现两个可信位置")
    reasons = [f["reason"] for f in wide["bad_frames"]]
    expect(all("两个可信位置" in reason for reason in reasons),
           "每帧原因写明出现两个可信位置")
    expect(not wide["sequence_available"], "不提供序列")
    narrow = run(items, shift=5.0)
    expect(narrow["n_valid"] == 14 and narrow["sequence_available"]
           and not narrow["reason_counts"],
           "允许位移收窄到 5 像素后隔离邻星,得到完整序列")
    chosen = [f["x"] for f in narrow["frames"]]
    expect(max(chosen) < STAR_X + 1.0, "测光位置始终在目标,而不是邻星")


def test_short_loss_resumes_tracking() -> None:
    print("短时丢失")
    gone = (5, 6, 7)
    items = frames(image_of=lambda i: make_image(star=i not in gone, x=STAR_X + 0.05 * i),
                   n=14)
    result = run(items)
    expect(result["n_valid"] == 11, "11 帧有效")
    expect(result["reason_counts"].get("target_lost") == 3,
           "3 帧因目标丢失记坏")
    expect(result["sequence_available"], "仍提供序列")
    expect(all("目标丢失" in f["reason"] for f in result["bad_frames"]),
           "坏帧原因写明目标丢失")
    after = [f for f in result["frames"] if f["index"] > 7]
    expect(after and all(f["status"] == "ok" for f in after),
           "坏帧之后从最后可信位置继续跟踪")


def test_too_few_valid_frames() -> None:
    print("有效帧不足")
    items = frames(image_of=lambda i: make_image(star=i < 4), n=12)
    result = run(items)
    expect(result["n_valid"] == 4 and not result["sequence_available"],
           "有效帧不足 10 帧时不提供序列")
    expect(f"少于 {MIN_VALID_FRAMES} 帧" in (result["sequence_error"] or ""),
           "说明有效帧不足的具体数量")


def test_duplicate_times() -> None:
    print("时刻重复")
    items = frames(mjd_of=lambda i: 60000.0 if i in (2, 3) else 60000.0 + 0.2 * i)
    result = run(items)
    expect(not result["sequence_available"], "有效帧时刻重复时不提供序列")
    expect("重复" in (result["sequence_error"] or ""), "说明时刻重复")


def test_out_of_bounds() -> None:
    print("孔径/背景环越界")
    items = frames(image_of=lambda i: make_image(x=1.0, y=1.0))
    result = run(items, x=1.0, y=1.0)
    expect(result["n_valid"] == 0 and result["reason_counts"].get("out_of_bounds") == 12,
           "越界帧全部记坏")
    expect("超出图像范围" in result["frames"][0]["reason"], "原因写明越界")


def test_validation() -> None:
    print("输入校验")
    try:
        load_zip(zip_bytes(frames(n=9)))
    except PhotometryInputError as exc:
        expect("至少需要 10" in str(exc), "少于 10 帧被拒绝")
    else:
        raise AssertionError("少于 10 帧应被拒绝")

    mixed = [(f"f{i:02d}.fits",
              fits_bytes(np.zeros((40, 40) if i % 2 else (32, 32)), 60000.0 + i))
             for i in range(12)]
    try:
        load_zip(zip_bytes(mixed))
    except PhotometryInputError as exc:
        expect("同尺寸" in str(exc), "尺寸不一致被拒绝")
    else:
        raise AssertionError("尺寸不一致应被拒绝")

    no_mjd = [(f"f{i:02d}.fits", fits_bytes(make_image(), 60000.0 + i, mjd_obs=False))
              for i in range(12)]
    try:
        load_zip(zip_bytes(no_mjd))
    except PhotometryInputError as exc:
        expect("MJD-OBS" in str(exc), "缺少 MJD-OBS 被拒绝")
    else:
        raise AssertionError("缺少 MJD-OBS 应被拒绝")

    negative = [(f"f{i:02d}.fits", fits_bytes(make_image(), 60000.0 + i, exptime=-5.0))
                for i in range(12)]
    try:
        load_zip(zip_bytes(negative))
    except PhotometryInputError as exc:
        expect("EXPTIME" in str(exc), "负数 EXPTIME 被拒绝")
    else:
        raise AssertionError("负数 EXPTIME 应被拒绝")


def test_gzip_and_extension_hdu() -> None:
    print("gzip 成员与扩展 HDU")
    gz_items = [(f"f{i:02d}.fits.gz",
                 gzip.compress(fits_bytes(make_image(), 60000.0 + 0.2 * i)))
                for i in range(12)]
    result = run(gz_items)
    expect(result["n_valid"] == 12, ".fits.gz 成员可读")

    ext_items = []
    for i in range(12):
        primary = fits.PrimaryHDU()
        primary.header["MJD-OBS"] = 60000.0 + 0.2 * i
        primary.header["EXPTIME"] = 300.0
        ext = fits.ImageHDU(make_image().astype(np.float32))
        buf = io.BytesIO()
        fits.HDUList([primary, ext]).writeto(buf)
        ext_items.append((f"f{i:02d}.fits", buf.getvalue()))
    result = run(ext_items)
    expect(result["n_valid"] == 12, "图像在扩展 HDU、观测头在主 HDU 时可读")


def test_sample_zips() -> None:
    print("内置样例 ZIP")
    root = Path(__file__).resolve().parents[1] / "sample"
    loss = root / "photometry_loss.zip"
    neighbor = root / "photometry_neighbor.zip"
    if not (loss.exists() and neighbor.exists()):
        print("  (样例 ZIP 不存在,跳过)")
        return
    result = process_frames(load_zip(loss.read_bytes()), 60, 62, 4, 8, 12, 10)
    expect(result["n_valid"] == 21 and result["sequence_available"],
           "loss 样例:21 帧有效,3 帧目标丢失,仍提供序列")
    expect(result["reason_counts"].get("target_lost") == 3,
           "loss 样例:3 帧目标丢失")
    expect(abs(result["sequence"][0]["flux"]) < 1e6, "loss 样例通量为有限值")

    wide = process_frames(load_zip(neighbor.read_bytes()), 60, 62, 4, 8, 12, 10)
    expect(wide["reason_counts"].get("two_positions") == 8,
           "neighbor 样例:8 帧出现两个可信位置")
    expect(wide["n_valid"] == 16, "neighbor 样例:其余 16 帧有效")
    narrow = process_frames(load_zip(neighbor.read_bytes()), 60, 62, 3, 5, 8, 4)
    expect(narrow["n_valid"] == 24 and not narrow["reason_counts"],
           "neighbor 样例:收窄位移后 24 帧全部有效")


def test_trend_survives_loss() -> None:
    print("序列可还原周期趋势")
    period = 2.0
    items = []
    for i in range(24):
        t = i * 0.25
        flux = 1.0 + 0.3 * math.sin(2 * math.pi * t / period)
        image = make_image(star=i not in (6, 7, 8))
        yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
        if i not in (6, 7, 8):
            image += (TARGET * flux - TARGET) * np.exp(
                -((xx - STAR_X) ** 2 + (yy - STAR_Y) ** 2) / 8.0)
        items.append((f"f{i:02d}.fits", fits_bytes(image, 60000.0 + t)))
    result = run(items)
    expect(result["sequence_available"] and result["n_valid"] == 21,
           "丢失 3 帧后仍有 21 帧序列")
    fluxes = np.array([s["flux"] for s in result["sequence"]])
    times = np.array([s["time_day"] for s in result["sequence"]])
    # 用 Lomb–Scargle 在 1～4 天范围找主周期,应接近 2 天
    from astropy.timeseries import LombScargle
    frequency = np.linspace(1.0 / 4.0, 1.0, 4000)
    power = LombScargle(times, fluxes, fit_mean=True).power(frequency)
    best_period = 1.0 / frequency[int(np.argmax(power))]
    expect(abs(best_period - period) < 0.15,
           f"序列主周期 {best_period:.2f} 天,接近真周期 {period} 天")


def main() -> None:
    tests = [
        test_sorted_by_exposure_midpoint,
        test_neighbor_two_positions,
        test_short_loss_resumes_tracking,
        test_too_few_valid_frames,
        test_duplicate_times,
        test_out_of_bounds,
        test_validation,
        test_gzip_and_extension_hdu,
        test_sample_zips,
        test_trend_survives_loss,
    ]
    for test in tests:
        test()
    print(f"\n全部 {len(tests)} 组自检通过")


if __name__ == "__main__":
    main()
