# RMVPE f0 契约 — 16 kHz 波形 → f0[Hz]@100fps + uv（Rust 消费方）

产物（`converter\export_rmvpe.py` 生成，gate：`converter\verify\voice\gate_rmvpe.py`）：

| 文件 | 内容 |
|---|---|
| `converter\test_output\rmvpe_e2e.onnx` | E2E 模型 + **in-graph** pad-to-32 + **in-graph** decode（f0 直接出 Hz），opset 17，fp32 |
| `converter\test_output\rmvpe_mel_filters.npy` | mel 滤波器组 f32 `[128, 513]`，Rust 直接加载，**不要重算** |

原版参照：`D:\MyDev\RVC\RVC20240604Nvidia\infer\lib\rmvpe.py`（`RMVPE.infer_from_audio(audio, thred=0.03)`，fp32 CPU）。
官方 `assets\rmvpe\rmvpe.onnx`（opset 10，输入 log-mel `[1,128,T]`，输出 salience，
T 需调用方 pad 到 32 倍数 + 调用方自己做 numpy decode）**不用**——我们的重导出把
padding 和 decode 都收进图里，Rust 侧 DSP 只剩 log-mel。

## ONNX I/O（rmvpe_e2e.onnx）

| 张量 | dtype/shape | 语义 |
|---|---|---|
| 输入 `mel` | f32 `[1, 128, T]` | log-mel（见下方 DSP 步骤），T 任意 ≥ 1，**不需要**是 32 的倍数 |
| 输入 `threshold` | f32 `[1]` | salience 门限，RVC 默认 **0.03**（pipeline.py rmvpe 路径固定用 0.03） |
| 输出 `f0` | f32 `[1, T]` | f0（Hz），100 fps；**清音帧 == 0.0**（精确零） |

图内已含（Rust 不做）：T→32 倍数零填充→E2E(DeepUnet+BiGRU)→salience `[1,T,360]`→
切回 T→to_local_average_cents decode（argmax + 9-bin 加权平均 + 门限）→cents→Hz→
f0==10→0 清音规则。

## Rust 侧 DSP（唯一职责：波形 → log-mel）

输入：**单声道 16 kHz f32** 波形 `x[N]`，幅度 [-1,1]，**不做任何归一化/预加重**（原版没有）。

常量：`SR=16000, N_FFT=1024, WIN=1024, HOP=160, N_MELS=128, CLAMP=1e-5`
（fmin=30 / fmax=8000 / htk=True / slaney-norm 已烧进 npy，Rust 无需关心）。

1. **reflect pad**：两端各补 `N_FFT/2 = 512` 样本，`mode=reflect`（不复制端点，
   即 torch `center=True` 语义）。⚠️ 要求 `N ≥ 513`，否则 reflect 未定义——短于此的
   输入调用方先补零到 513。
2. **分帧**：帧数 `T = 1 + N / HOP`（整除向下取整）。第 `t` 帧 = padded 信号
   `[t*160 .. t*160+1024)`。
3. **加窗**：periodic hann `w[k] = 0.5 - 0.5*cos(2*pi*k/1024)`，k=0..1023
   （f64 建表后转 f32，= torch.hann_window 默认）。
4. **rFFT 1024** → 513 bins，**幅度谱** `sqrt(re^2+im^2)`（不是功率谱，无任何 normalize）。
   这与已有 torch-lineage `stft.rs` 同语义（同款 reflect pad + periodic hann）。
5. **mel**：`mel[128,T] = filters[128,513] @ magnitude[513,T]`（npy 逐值用，f32）。
6. **log**：`log_mel = ln(max(mel, 1e-5))`（自然对数）。

全程 f32 即可——gate b2 实测 f32 链 vs 原版 fp32 的线性域 max_abs_diff ≤ 1.2e-6
（fp32 噪声底），端到端 f0 影响 ≤ 1.3e-6 相对误差（gate a）。

## 帧对齐 / 采样率换算

- `T = 1 + floor(N / 160)`；帧 `t` 的中心 = 原始信号第 `t*160` 个样本（center=True）。
- 帧率 = 16000/160 = **100 fps**，即 `f0[t]` 对应时间 `t * 10 ms`。
- 任意采样点 `s` → 帧 `round(s / 160)`；任意时刻 `sec` → 帧 `round(sec * 100)`。
- 输出长度 = 输入 mel 的 T（图内 pad/切片对调用方透明）。

## thred 与 uv 规则

- `threshold`（原版 `thred`）：某帧 salience 最大值 ≤ threshold ⇒ 该帧判清音。
  在图内实现为 decode 的 `devided[maxx <= thred] = 0` + `f0[f0==10] = 0` 原版语义。
- **uv 规则（Rust 侧唯一判据）**：`uv[t] = (f0[t] == 0.0)`。清音帧输出为精确 0.0f32
  （图内 Where 直接放常数 0，不经浮点运算），可安全用 `== 0.0` 判断。
- RVC 管线语义：清音帧 f0=0 参与后续 f0_to_coarse 时 coarse=1（f0_mel<=0 分支）。

## Gate 记录（2026-07-04，gate_rmvpe.py，全过）

素材：`D:\MyDev\TESTING\ikanaiteyo\vocal.wav` 前 15 s（T=1501）+
`D:\MyDev\TESTING\MSST\leak_fix_ab\FIXED_vocals_20s.wav`（T=2001），均重采样 16k 单声道。

| Gate | 读数 | 线 |
|---|---|---|
| b1 结构（f64 sim vs 原版代码强制 f64） | max_abs_diff(log) **1.5e-12 / 3.6e-12** | < 1e-4 |
| b2 fp32 噪声（f32 sim vs 原版 fp32，线性域） | **7.2e-07 / 1.2e-06** | < 1e-5 |
| a 全链 f0（vs 原版 infer_from_audio） | 100% 浊音帧 rel<0.1%，worst **1.28e-06 / 1.05e-06** | ≥99% |
| a uv 一致率 | **100.000% / 100.000%** | > 99% |
| c 动态 T（1501/2001，均非 32 倍数） | 长度全对；导出冒烟另测 T=417/256/100 | — |
| isolate（原版 torch mel → 我们的 onnx） | worst voiced rel 1.26e-06，uv 100% | — |

⚠️ 方法论备注：对 fp32 参照做**全局 log 域** max_abs_diff 门是构造性不可达的——
紧贴 clamp(1e-5) 的 mel bin 会把参照自身的 fp32 舍入（线性域 ~1e-6）放大到 log 域
~1e-3。因此结构等价用 f64-vs-f64 证（b1），fp32 噪声用线性域界定（b2），
f0 影响由 gate a 封顶。近 clamp 的 log 域差按电平分层已在 gate 输出留档
（>-7 电平的 bin 全部 < 1e-4）。

其他备注：
- 模型 361 MB fp32。fp16 未验证未开放（按项目规则需单独 CUDA 关卡 3）。
- decode 的 argmax 平局取首个（torch/ONNX/numpy 三方同语义，实数据不会触发）。
- 重导出命令：`.venv\Scripts\python.exe converter\export_rmvpe.py`（内置 3 长度冒烟）。
