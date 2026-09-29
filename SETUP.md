# 环境准备（CUDA Training Series 本地跑法）

原仓库的 readme 只写了 ORNL Summit / NERSC Cori 的 `module load` + `bsub/srun` 流程。
本文件补齐**本地单机**需要的组件、安装命令，以及每个作业的编译/运行命令。

## 1. 需要哪些组件

| 组件 | 用途 | 用到的作业 |
| --- | --- | --- |
| NVIDIA 驱动 + GPU（计算能力 >= 7.0 推荐） | 实际运行 kernel | 全部 |
| CUDA Toolkit 11.x（`nvcc`、`thrust`、`cuBLAS`、`cuda-gdb`、`compute-sanitizer`） | 编译/调试/内存检查 | 全部、hw13、hw12 |
| Nsight Compute（`ncu`，旧名 `nv-nsight-cu-cli`） | kernel profiling | hw3 hw4 hw5 hw8 hw9 |
| Nsight Systems（`nsys`） | 时间线/重叠分析 | hw6 hw7 hw10 hw11 |
| GCC/g++ >= 5（C++11） | nvcc 的主机编译器 | 全部 |
| OpenMP（`gcc -fopenmp`） | 多线程提交 stream | hw10 |
| OpenMPI（`mpicxx` / `mpirun`） | 多进程 | hw11（没 MPI 可用 `-DNO_MPI` 替代） |
| MPS 守护进程（`nvidia-cuda-mps-control`） | 多进程共享 GPU | hw11 |

CUDA 11.8 需要驱动 >= 520；只想跑 `-arch=sm_70` 的话驱动 >= 470 也行。
hw9 的 `task2` 用到 grid 同步，需要计算能力 >= 6.0。
hw9 / hw12 / hw13 的 readme 用 `-arch=sm_70`，请按自己的卡替换（见第 5 节）。

## 2. 当前环境（已配置好）

```
Ubuntu 18.04.6 LTS, gcc 7.5.0
CUDA Toolkit 11.8  ->  $HOME/cuda-11.8   (nvcc 11.8.89)
有：nvcc  ncu  nv-nsight-cu-cli  nsys  compute-sanitizer  cuda-gdb  thrust  cuBLAS  OpenMP
已装：OpenMPI 4.1.5（CUDA-aware）-> $HOME/openmpi-4.1.5（源码编译，不需要 root）
      mpicxx -> g++ 7.5.0；mpirun -np 4 已验证可用
      nvcc -o test -ccbin=mpicxx test.cu（hw11）已验证编译通过
GPU：2 x NVIDIA TITAN RTX，compute capability 7.5，24GB/卡，驱动 550.54.14
      已用 `nvcc -arch=native` + 真 kernel 验证可用（Linux 下不需要 WSL/容器特判）

环境变量由 ~/.bashrc 自动加载（~/.cuda-env.sh + ~/.mpi-env.sh），新开 shell 直接就有
nvcc / mpicxx；不想重开 shell 就 `source env.sh`。readme 里的 `module load cuda`、
`module load openmpi/4.0.3` 已在本地模拟。
```

> 注：如果 `nvidia-smi` 报 “couldn't communicate with the NVIDIA driver”，先确认不是在受限沙箱里跑
> （沙箱会屏蔽 `/dev/nvidia*`，宿主机上其实是正常的）。

自检命令：

```bash
for t in nvcc ncu nv-nsight-cu-cli nsys compute-sanitizer cuda-gdb mpicxx mpirun nvidia-cuda-mps-control; do
  printf '%-24s %s\n' "$t" "$(command -v $t || echo MISSING)"
done
nvcc --version
nvidia-smi
```

实测结果（真在 GPU 上跑过）：

```
hw3/vector_add           通过
hw5/reductions           三个 kernel 结果全对
hw5/matrix_sums          row/column sums 正确
hw6/array_inc            success!
hw7/overlap_solution     非重叠 0.053s -> streams 0.029s（约 1.8x）
hw8/task3                PASS，339 GB/s
hw10/streams -DUSE_STREAMS  非重叠 0.052s -> 重叠 0.0275s（约 1.9x）
hw11/test（mpirun -np 2）  每 kernel 约 3.0 ms
hw12/task1               compute-sanitizer 正确报出越界（645 errors）——这就是题目要你查的 bug
hw12/task2               输出 -inf —— 这就是题目要你查的 bug
hw13/axpy_stream_capture 正常运行
hw5/max_reduction        输出的是 sum 而非 max —— 题目要求你改成 max
hw6/linked_list          骨架用 malloc，GPU 解引用报 illegal access —— 题目要求改成 cudaMallocManaged
```

## 3. 补环境的安装命令（本机已执行过，换机器时照抄）

### 3.1 CUDA 环境变量（每个新 shell，或写进 ~/.bashrc）

```bash
export CUDA_HOME=$HOME/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 3.2 安装 OpenMPI（hw11 用，Ubuntu/Debian）

```bash
sudo apt-get update
sudo apt-get install -y openmpi-bin libopenmpi-dev
mpicxx --version && mpirun --version
```

本机没有可用的 sudo（`sudo` 没带 setuid 位），所以改成源码装到 `$HOME`（✔ 已执行）；
有 root 的机器用上面那段 apt 就够了。Ubuntu 18.04 自带 OpenMPI 3.1，也够用。

```bash
cd /tmp
curl -LO https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
tar xzf openmpi-4.1.5.tar.gz && cd openmpi-4.1.5
./configure --prefix=$HOME/openmpi-4.1.5
make -j"$(nproc)" && make install
export PATH=$HOME/openmpi-4.1.5/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openmpi-4.1.5/lib:$LD_LIBRARY_PATH
```

### 3.3 驱动 / Nsight 工具（缺的时候才装）

```bash
# NVIDIA 驱动（Ubuntu 仓库版，版本号按显卡选）
sudo apt-get install -y nvidia-driver-525-server
sudo reboot

# Nsight Compute / Systems 若不在 CUDA Toolkit 里，去官网单独下：
#   https://developer.nvidia.com/nsight-compute
#   https://developer.nvidia.com/nsight-systems
```

### 3.4 如果在 HPC 上（原 readme 那套）

```bash
# ORNL Summit
module load cuda gcc nsight-compute nsight-systems

# NERSC Cori / Perlmutter
module load cgpu gcc/8.3.0 cuda/11.4.0 openmpi/4.0.3 nsight-compute nsight-systems
```

## 4. 每个作业的编译 / 运行命令（本地，单 GPU）

环境已经配好：新开 shell 里 `nvcc` / `mpicxx` 直接可用；当前 shell 里先 `source env.sh`。
所有 `-arch=sm_70` 都可以换成 `-arch=native`（CUDA 11.5+ 自动识别本机架构）。

```bash
# ---------- hw1 ----------
cd exercises/hw1
nvcc -o hello hello.cu                     && ./hello
nvcc -o vector_add vector_add.cu           && ./vector_add
nvcc -o matrix_mul matrix_mul.cu           && ./matrix_mul

# ---------- hw2 ----------
cd ../hw2
nvcc -o stencil_1d stencil_1d.cu           && ./stencil_1d
nvcc -o matrix_mul matrix_mul_shared.cu    && ./matrix_mul
ncu ./stencil_1d                           # 可选 profiling

# ---------- hw3 ----------
cd ../hw3
nvcc -o vector_add vector_add.cu           && ./vector_add
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./vector_add

# ---------- hw4 ----------
cd ../hw4
nvcc -o matrix_sums matrix_sums.cu         && ./matrix_sums
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./matrix_sums

# ---------- hw5 ----------
cd ../hw5
nvcc -o reductions reductions.cu           && ncu ./reductions
nvcc -o max_reduction max_reduction.cu     && ./max_reduction
nvcc -o matrix_sums matrix_sums.cu         && ncu ./matrix_sums

# ---------- hw6 ----------
cd ../hw6
nvcc -o linked_list linked_list.cu         && ./linked_list
nvcc -o array_inc array_inc.cu             && nsys profile --stats=true ./array_inc

# ---------- hw7 ----------
cd ../hw7
nvcc -o overlap overlap.cu                 && ./overlap          # 非重叠版
nvcc -o overlap overlap.cu -DUSE_STREAMS   && ./overlap          # 重叠版
nsys profile -o overlap.qdrep ./overlap
nvcc -o multi multi.cu                     && ./multi            # 需要 4 张 GPU

# ---------- hw8 ----------
# 注意：task*/build_nvcc 里只写了 `nvcc -O3 -o task1 task1.cu`，没带 -arch；
# 想指定架构就编辑这个脚本，或直接用 `nvcc -arch=native -O3 -o task1 task1.cu`
cd ../hw8/task1 && bash build_nvcc && ./task1 && bash profile
cd ../task2     && bash build_nvcc && ./task2 && bash profile
cd ../task3     && bash build_nvcc && ./task3 && bash profile

# ---------- hw9 ----------
cd ../../hw9
nvcc -arch=sm_70 -o task1 task1.cu -std=c++11               && ./task1
nvcc -arch=sm_70 -o task2 task2.cu -rdc=true -std=c++11     && ./task2
ncu ./task2

# ---------- hw10 ----------
cd ../hw10
nvcc -o streams streams.cu                 && ./streams
nvcc -Xcompiler -fopenmp -o streams streams.cu -DUSE_STREAMS
OMP_NUM_THREADS=8 ./streams
nsys profile -o streams.qdrep ./streams

# ---------- hw11（MPI + MPS）----------
cd ../hw11
nvcc -o test -ccbin=mpicxx test.cu
mpirun -np 1 ./test 1073741824
mpirun -np 4 ./test 1073741824
nvidia-cuda-mps-control -d                                     # 开 MPS
nsys profile --stats=true -t nvtx,cuda -s none -o 4ranks_mps -f true mpirun -np 4 ./test 1073741824
echo "quit" | nvidia-cuda-mps-control                          # 关 MPS
# 没有 MPI 的机器：
nvcc -DNO_MPI -o test test.cu && ./run_no_mpi.sh

补充：用 `nsys` 抓 `mpirun` 时加 `--target-processes all` 才会记录各个 rank 子进程；
`mpirun -np 4` 在单张卡上若报资源不足，加 `--oversubscribe`。

# ---------- hw12 ----------
cd ../hw12
nvcc -arch=sm_70 task1.cu -o task1 -lineinfo && compute-sanitizer ./task1 && ./task1
nvcc -arch=sm_70 task2.cu -o task2 -G -g -std=c++14 && cuda-gdb ./task2

# ---------- hw13 ----------
cd ../hw13
nvcc -arch=sm_70 axpy_stream_capture_with_fixme.cu -o axpy_stream_capture_with_fixme && ./axpy_stream_capture_with_fixme
nvcc -arch=sm_70 -lcublas axpy_cublas_with_fixme.cu -o axpy_cublas_with_fixme         && ./axpy_cublas_with_fixme
```

## 5. 常见问题

- `nvcc: command not found`：先做 3.1 的 `PATH` 导出，或直接用绝对路径 `$HOME/cuda-11.8/bin/nvcc`。
- `mpicxx: command not found`：没装 OpenMPI，做 3.2；或 hw11 改用 3.4 的 `-DNO_MPI` 方案。
- 运行任何可执行文件都报 `no CUDA-capable device is detected`：只装了 toolkit、没有可用 GPU/驱动（本容器就是这种情况）。这时只能做**编译检查**：

  ```bash
  nvcc -arch=sm_70 -c file.cu -o /dev/null      # 只编译
  nvcc -arch=sm_70 -ptx file.cu -o file.ptx     # 出 PTX
  ```

- `-arch` 报错：把 `sm_70` 换成你 GPU 的架构，或用 `-arch=native`。
  查本机架构：
  ```bash
  nvidia-smi --query-gpu=compute_cap --format=csv
  ```
- hw9 `task2` 需要 cooperative launch，必须 CC >= 6.0 且用 `-rdc=true` 编译。
- `nsys` 不用 root，已验证可用。`ncu` 需要 GPU 性能计数器权限，否则报
  `ERR_NVGPUCTRPERM: The user does not have permission to access NVIDIA GPU Performance Counters`。
  在**宿主机 root** 下执行下面之一，然后重启（或重载 nvidia 模块）：

  ```bash
  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiler.conf
  sudo update-initramfs -u && sudo reboot
  ```

  改完 `ncu ./xxx` 就能采到计数器；改不了就用 `nsys` 或普通 `./xxx` 先做实验。
- 已知上游问题：`hw13/axpy_cublas_from_scratch.cu` 本身编译不过（第 79 行 `thread` 应为 `threads`，
  且两条 `kernel_a<<<...>>>` 缺分号）。这是原仓库就带的错，和 FIXME 无关；
  要跑 cuBLAS + graph 的版本请用 `axpy_cublas_with_fixme.cu`（填完 FIXME）或 `Solutions/` 里的版本。
- `git` 写 .git 报 Read-only（本仓库 `.git` 是只读挂载）：不要指望 `git add/commit`，改文件本身没问题。

## 6. FlashAttention（额外依赖，和本仓库作业无关）

本机是 Turing `sm_75`，所以：

| 版本 | 是否可用 | 原因 |
| --- | --- | --- |
| `flash-attn >= 2.x`（FA2） | 不行 | 只支持 Ampere(sm80)/Ada/Hopper |
| `flash-attn == 1.0.9`（FA1） | 可以 | setup.py 明确编 `sm_75`；只支持 fp16，head_dim 必须是 8 的倍数且 <= 64（反向） |

**状态：已装好并验证通过。** 一键脚本：`bash install_flash_attn.sh`（约 10 分钟下载 torch + 15–30 分钟编译）。

```
venv        : ~/venvs/flash-attn  (Python 3.10.21)
torch       : 2.1.2+cu118
flash-attn  : 1.0.9  (kernel: sm_75 + sm_80 + sm_90；cuobjdump 确认)
验证        : 前向/反向/causal 均通过；head_dim=64, seq=512 与参考实现 max|diff| = 6e-5
激活        : source ~/venvs/flash-attn/bin/activate
```

装这个版本踩到 3 个坑（脚本里都已处理）：

1. `setuptools>=81` 删掉了 `pkg_resources`，而 torch 2.1.2 的 `cpp_extension` 还在 import → 固定 `setuptools==69.5.1`
2. `numpy>=2` 与 torch 2.1.2 ABI 不兼容（`_ARRAY_API not found`）→ 固定 `numpy<2`
3. 本机 gcc 7.5 不认 uv 自带 CPython 的 CFLAGS 里的 `-fstack-clash-protection`（gcc 8+ 才有）
   → 用 `~/tools/bin/{gcc,gxx}-wrap` 包一层，把该标志过滤掉后再调 `/usr/bin/gcc`

`pip install` 时若报 “Failed to acquire lock on the distribution cache / is another uv process running?”，
说明有另一个 uv 在跑（或上次的构建还活着），先 `pgrep -af "uv pip"` 确认。
另外 `triton` 没有装也不需要：FA1 用不到它，只是 torch 的依赖会顺带解析。

```bash
export PATH=$HOME/.local/bin:$HOME/cuda-11.8/bin:$PATH
export CUDA_HOME=$HOME/cuda-11.8

# PyTorch 2.1.2 + cu118（官方源 ~120KB/s，换阿里云镜像 ~4MB/s）
uv pip install --python ~/venvs/flash-attn/bin/python --no-deps \
  "https://mirrors.aliyun.com/pytorch-wheels/cu118/torch-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl"

# 其余依赖（torch 用了 --no-deps；triton 不需要）
# 两个必须固定的版本：setuptools>=81 删了 pkg_resources（torch 2.1.2 要用），numpy 2.x 与 torch 2.1.2 ABI 不兼容
uv pip install --python ~/venvs/flash-attn/bin/python \
  filelock typing-extensions sympy networkx jinja2 fsspec einops ninja packaging psutil wheel \
  "setuptools==69.5.1" "numpy<2"

# 编译安装 FA1（走 PyPI sdist，自带 cutlass；use MAX_JOBS 控制并发/内存）
MAX_JOBS=16 uv pip install --python ~/venvs/flash-attn/bin/python \
  --no-build-isolation "flash-attn==1.0.9"
```

验证：

```bash
~/venvs/flash-attn/bin/python - <<'PY'
import torch
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
B, S, H, D = 2, 128, 4, 64
qkv = torch.randn(B*S, 3, H, D, device="cuda", dtype=torch.float16)
cu = torch.arange(0, (B+1)*S, S, dtype=torch.int32, device="cuda")
out, _, _ = flash_attn_unpadded_qkvpacked_func(qkv, cu, S, 0.0, causal=False, return_attn_probs=True)
print("OK", tuple(out.shape), out.dtype)
PY
```
