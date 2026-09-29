#!/usr/bin/env bash
# 安装 FlashAttention（本机：2 x TITAN RTX，sm_75 Turing，CUDA 11.8）
#
# 为什么是 v1.0.9：
#   * FlashAttention-2（flash-attn >= 2.x）只支持 Ampere(sm80) 及以上，Turing 跑不了
#   * v1.0.9 的 setup.py 明确编译 sm_75，可以跑
#   * v1 的限制：只支持 fp16（bf16 要 Ampere）；head_dim 必须是 8 的倍数且 <= 64（反向）
#
# 用法：  bash install_flash_attn.sh
# 中途失败可重复执行（幂等）；想改并发：MAX_JOBS=8 bash install_flash_attn.sh
set -euo pipefail

VENV="${VENV:-$HOME/venvs/flash-attn}"
PYVER="3.10"
export CUDA_HOME="${CUDA_HOME:-$HOME/cuda-11.8}"
export PATH="$HOME/.local/bin:$CUDA_HOME/bin:$PATH"

echo "== 0/5 uv =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "== 1/5 Python $PYVER + venv =="
uv python install "$PYVER"
[ -d "$VENV" ] || uv venv --python "$PYVER" "$VENV"

echo "== 2/5 PyTorch 2.1.2+cu118（阿里云镜像，官方源只有 ~120KB/s）=="
# 换机器/换 Python 版本时改文件名里的 cp310 和 cu118
uv pip install --python "$VENV/bin/python" --no-deps \
  "https://mirrors.aliyun.com/pytorch-wheels/cu118/torch-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl"

echo "== 3/5 其余依赖（torch 是 --no-deps 装的，缺的补上；triton 用不到）=="
# 注意两个固定版本，装错会直接构建失败：
#   * setuptools>=81 删掉了 pkg_resources，而 torch 2.1.2 的 cpp_extension 还在 import 它
#   * numpy 2.x 与 torch 2.1.2 不同 ABI（报 _ARRAY_API not found），必须 <2
uv pip install --python "$VENV/bin/python" \
  filelock typing-extensions sympy networkx jinja2 fsspec einops ninja packaging psutil wheel \
  "setuptools==69.5.1" "numpy<2"

echo "== 4/5 编译安装 flash-attn 1.0.9（会编 sm_75/80/90，约 15-30 分钟）=="
# 本机 gcc 只有 7.5，而 uv 自带 CPython 的 CFLAGS 里带 gcc 8+ 才支持的
# -fstack-clash-protection，会被当成非法选项。包一层编译器把它吞掉。
mkdir -p "$HOME/tools/bin"
printf '#!/bin/bash\nargs=()\nfor a in "$@"; do [ "$a" = "-fstack-clash-protection" ] && continue; args+=("$a"); done\nexec /usr/bin/gcc "${args[@]}"\n' > "$HOME/tools/bin/gcc-wrap"
printf '#!/bin/bash\nargs=()\nfor a in "$@"; do [ "$a" = "-fstack-clash-protection" ] && continue; args+=("$a"); done\nexec /usr/bin/g++ "${args[@]}"\n' > "$HOME/tools/bin/gxx-wrap"
chmod +x "$HOME/tools/bin/gcc-wrap" "$HOME/tools/bin/gxx-wrap"
export CC="$HOME/tools/bin/gcc-wrap"
export CXX="$HOME/tools/bin/gxx-wrap"
export CUDAHOSTCXX="$HOME/tools/bin/gxx-wrap"
unset CFLAGS CXXFLAGS
MAX_JOBS="${MAX_JOBS:-16}" uv pip install --python "$VENV/bin/python" \
  --no-build-isolation "flash-attn==1.0.9"

echo "== 5/5 冒烟测试 =="
"$VENV/bin/python" - <<'PY'
import torch
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
B, S, H, D = 2, 128, 4, 64          # Turing 上 head_dim 必须 <= 64
qkv = torch.randn(B * S, 3, H, D, device="cuda", dtype=torch.float16)
cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device="cuda")
out, _, _ = flash_attn_unpadded_qkvpacked_func(qkv, cu, S, 0.0, causal=False, return_attn_probs=True)
print("flash-attn OK | out:", tuple(out.shape), out.dtype, "| finite:", bool(torch.isfinite(out).all()))
PY

echo
echo "完成。用法：  source $VENV/bin/activate"
