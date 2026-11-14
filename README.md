# 巡测测光 · 周期变星候选筛选

面向天文巡测数据员的网页工具,分两步:

1. **FITS 孔径测光**:上传连续曝光的二维 FITS 帧 ZIP,按首帧目标坐标逐帧
   测光,得到 `time_day`、`flux` 光变序列,并逐帧列出可信/坏帧原因;
2. **周期筛查**:把测光序列送入原有的 Lomb–Scargle + 置换检验入口,输出
   最高峰周期、功率谱、相位折叠散点与置换尾概率。

## 快速开始

```bash
docker compose up --build
# 打开 http://localhost:8000
```

页面上可直接点击两个虚构 FITS 样例:

- **载入样例:短时丢失** —— 目标在第 9～11 帧短暂消失,坏帧之后仍从最后
  可信位置继续跟踪,得到 21 帧有效序列,可下载 CSV 并直接送入周期筛查
  (真周期约 3.1 天);
- **载入样例:邻星干扰** —— 目标东侧 9 像素处有一颗邻星,第 9～16 帧邻星变亮,
  搜索半径 10 像素内每帧都出现两个可信位置,这些帧逐帧说明原因且不猜目标;
  其余帧正常。再点 **邻星干扰 · 收窄搜索半径**(允许位移改为 4 像素),
  即可隔离邻星并得到 24 帧完整序列。

原有点击「载入虚构样例并提交」仍可使用 `sample/sample_star.csv`
(周期约 4.25 天的虚构脉动变星,160 个观测点)。

无 Docker 时本地运行:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 输入要求:测光 ZIP

- ZIP 内含 **10～30 个**二维 FITS 文件(支持 `.fits/.fit/.fts`,也支持
  `.fits.gz`;忽略 `__MACOSX` 等非 FITS 成员);
- 所有帧**同尺寸**;单帧不超过 64 MB,解压后总体积不超过 400 MB,
  合计像素数不超过 3200 万;
- 每帧头含 **`MJD-OBS`**(曝光起点,允许数值或 ISO 时间字符串)与
  **`EXPTIME`**(秒,须为正数);扩展 HDU 缺这两个头时回退到主头;
- 页面填写:首帧目标坐标 `x`(列)、`y`(行),均从 0 起;孔径半径;
  背景环内、外半径;允许帧间位移(像素);约束
  `0 < 孔径半径 < 环内半径 < 环外半径`,`允许位移 ≥ 0`;
- 首帧坐标须落在图像内。

## 测光算法与语义

1. **排序**:曝光中点 `MJD-OBS + EXPTIME / 2 / 86400`(天),按升序处理;
   `time_day` 输出同一中点值。
2. **检测与跟踪**:在「最后可信位置 ± 允许帧间位移」范围内做 3×3 盒式平滑,
   在平滑图上找显著局部极大值(阈值 5σ,σ 由全图 1.4826×MAD 估计),
   用 3×3 亮权重质心细化到亚像素;距离小于 2 像素的极大值合并为同一颗星。
3. **测光**:圆孔径内像素总量 − 背景环(环带像素)中位数 × 孔径有效像素数;
   通量误差由环内散度估计;孔径通量须为正且 S/N ≥ 5 才算可信。
4. **逐帧判定**:
   - **出现两个可信位置**:不猜哪个是目标,本帧记坏并列出每个候选的坐标、
     通量与 S/N,绝不把邻星当目标;
   - **目标丢失**:搜索范围内没有可信候选,本帧记坏;
   - **孔径/背景环越界**:候选位置放不下完整背景环(或首帧坐标本身越界),
     本帧记坏并给出坐标、半径与图像尺寸;
   - **疑似邻星**:只剩一个可信候选,但通量与近期目标中位数相差超过 2 倍,
     视为邻星,本帧记坏。
5. **恢复**:坏帧不改变跟踪锚点;下一帧仍从**最后一个可信位置**继续搜索。
6. **不提供序列的情形**:有效帧少于 10 帧,或有效帧的曝光中点重复
   (容差约 0.9 毫秒)。网页列出具体原因,且不提供 CSV;接口返回 409。
7. 非有限像素(缺测/饱和标记)按图像中位数填补。

## 输入要求:周期 CSV

- CSV 含 `time_day`、`flux` 两列(允许其他列存在,只取这两列);
- 至少 10 行;两列均为有限数值(无空值/NaN/Inf);观测时刻互异;
- 周期范围须满足 `0 < 下限 < 上限`。

## 周期计算流程(沿用原有实现)

1. 周期范围取倒数得到频率范围 `[1/上限, 1/下限]`,均匀取 **10000** 个频率点;
2. 用**带浮动均值**的 Lomb–Scargle(`astropy.timeseries.LombScargle`,
   `fit_mean=True`)计算功率谱,取最高峰周期;
3. **置换检验**:固定时刻,随机重排光通量 **200** 次,每次重算全频段
   最大功率;尾概率 = `(不低于原峰的次数 + 1) / 201`;
4. 展示最高峰周期、峰值功率、尾概率、功率谱图与按峰周期折叠的相位散点图。

## 并发与生命周期语义

- 测光最多 **2 批并行**,周期筛查最多 **2 批并行**,超额提交返回 **HTTP 429**;
- 周期筛查每批使用 **2 个计算子进程**(各承担 100 次置换,worker 0 另算原始
  功率谱);
- 任务可查询、可取消(周期批:`POST /api/jobs/{id}/cancel`);
- 无论成功、失败还是取消,都要等本批全部子进程退出、临时文件清理完毕,
  才公布结局并释放名额;另一批不受影响;
- 取消一经受理即终止子进程,**不会发布迟到结果**;重复取消幂等,不会
  重复释放名额;
- 测光在**线程池**中执行(不阻塞事件循环),图像按帧从 ZIP 读取、处理完即释放,
  整批像素不常驻内存;
- 全部任务状态只存内存(重启即清空);计算期临时文件随收尾删除。

## API 摘要

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 网页 |
| POST | `/api/photometry` | 提交测光(multipart:`file`、`x`、`y`、`aperture_radius`、`bg_inner_radius`、`bg_outer_radius`、`max_shift`)→ 201 / 400 / 429 |
| GET | `/api/photometry` | 测光任务列表与名额占用 |
| GET | `/api/photometry/{id}` | 测光详情(逐帧状态、原因、候选) |
| GET | `/api/photometry/{id}/sequence.csv` | 下载 `time_day,flux`(无序列时 409) |
| POST | `/api/photometry/{id}/send-to-period` | 把序列送入周期入口(form:`period_min`、`period_max`)→ 201 / 409 / 429 |
| GET | `/api/photometry/samples` | 内置 FITS 样例清单 |
| GET | `/api/photometry/sample/{name}.zip` | 内置 FITS 样例下载(`neighbor` / `loss`) |
| POST | `/api/jobs` | 提交周期批处理(multipart:`file`、`period_min`、`period_max`)→ 201 / 400 / 429 |
| GET | `/api/jobs` | 周期批处理列表与名额占用 |
| GET | `/api/jobs/{id}` | 单批详情(含指标) |
| POST | `/api/jobs/{id}/cancel` | 取消(幂等) |
| GET | `/api/jobs/{id}/plot/power.png` | 功率谱图(成功后) |
| GET | `/api/jobs/{id}/plot/phase.png` | 相位折叠散点图(成功后) |
| GET | `/api/sample.csv` | 原有 CSV 样例 |

示例:

```bash
curl -F "file=@sample/photometry_loss.zip" \
     -F "x=60" -F "y=62" -F "aperture_radius=4" \
     -F "bg_inner_radius=8" -F "bg_outer_radius=12" \
     -F "max_shift=10" http://localhost:8000/api/photometry

curl -F "period_min=1" -F "period_max=6" \
     http://localhost:8000/api/photometry/<id>/send-to-period
```

## 目录结构

```
app/main.py                  FastAPI 主程序:校验、任务队列、周期子进程编排、出图
app/photometry.py            FITS ZIP 读取、孔径测光、目标跟踪与逐帧判定
app/worker.py                周期计算子进程:Lomb–Scargle 功率谱 + 置换检验
app/static/index.html        网页(原生 JS,无外部依赖)
sample/make_fits_samples.py  虚构 FITS 样例生成脚本(固定种子,可复现)
sample/photometry_loss.zip   短时丢失样例(24 帧)
sample/photometry_neighbor.zip  邻星干扰样例(24 帧)
sample/make_sample.py        原有 CSV 样例生成脚本
sample/sample_star.csv       原有 CSV 样例(160 行,真周期 4.25 天)
tests/test_photometry.py     测光语义自检(直接运行,无需 pytest)
Dockerfile / docker-compose.yml
```

重新生成样例:

```bash
python sample/make_fits_samples.py   # 注意:会覆盖两个 ZIP
python sample/make_sample.py
```

## 自检

```bash
python tests/test_photometry.py
```

覆盖:排序与中点、邻星两可信位置、收窄位移后隔离邻星、丢失后从最后可信位置
继续、有效帧不足 10、有效帧时刻重复、越界、同尺寸与头校验、gzip/扩展 HDU。
