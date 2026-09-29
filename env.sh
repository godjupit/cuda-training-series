#!/usr/bin/env bash
# 用法：source ./env.sh
# 一次性把本仓库需要的工具链加进当前 shell（等价于 readme 里的 module load 那套）。
# 新开 shell 通常不需要它：~/.bashrc 已经自动 source 了 ~/.cuda-env.sh 和 ~/.mpi-env.sh。

export CUDA_HOME="${CUDA_HOME:-$HOME/cuda-11.8}"
export CUDA_PATH="$CUDA_HOME"

case ":$PATH:" in
    *":$CUDA_HOME/bin:"*) ;;
    *) PATH="$CUDA_HOME/bin:$PATH" ;;
esac
case ":${LD_LIBRARY_PATH:-}:" in
    *":$CUDA_HOME/lib64:"*) ;;
    *) LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

# OpenMPI 是可选的：没装也能用 hw11 的 -DNO_MPI 方案
export MPI_HOME="${MPI_HOME:-$HOME/openmpi-4.1.5}"
if [ -x "$MPI_HOME/bin/mpicxx" ]; then
    case ":$PATH:" in
        *":$MPI_HOME/bin:"*) ;;
        *) PATH="$MPI_HOME/bin:$PATH" ;;
    esac
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$MPI_HOME/lib:"*) ;;
        *) LD_LIBRARY_PATH="$MPI_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi

export PATH LD_LIBRARY_PATH

# 让 readme 里的 `module load cuda` / `module load openmpi` 也能用
if ! type module >/dev/null 2>&1; then
    module() {
        case "$1" in
            load)
                shift
                for m in "$@"; do
                    case "$m" in
                        cuda|cuda/*) . "$HOME/.cuda-env.sh" ;;
                        openmpi|openmpi/*|mpi) [ -f "$HOME/.mpi-env.sh" ] && . "$HOME/.mpi-env.sh" ;;
                        gcc|gcc/*|cgpu|esslurm|nsight-compute|nsight-compute/*|nsight-systems|nsight-systems/*) : ;;
                        *) printf 'module: %s 在本机不可用（cuda/openmpi/gcc/nsight-* 已模拟）\n' "$m" >&2 ;;
                    esac
                done
                ;;
            purge) : ;;
            list|avail) printf 'cuda/11.8  openmpi/4.1.5\n' ;;
            '') : ;;
            *) printf 'module: unsupported command: %s\n' "$1" >&2; return 1 ;;
        esac
    }
fi
