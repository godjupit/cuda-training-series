// myprof.cuh -- 没有 GPU 计数器权限时的 Nsight Compute 替身
//
// 为什么需要它：ncu 的绝大多数 section（SpeedOfLight / MemoryWorkloadAnalysis /
// Occupancy / LaunchStats）都靠硬件性能计数器，而 CUPTI 计数器要求在宿主机上把
// nvidia 内核模块的 NVreg_RestrictProfilingToAdminUsers 关掉（或直接 root）。
// 拿不到那就没有任何命令行开关能绕过，唯一现实的办法是换成不受限的 CUDA
// Runtime API 自己量。这个头文件做的就是这件事。
//
// 数据来源与 ncu 的对应关系：
//   cudaEventRecord            -> Duration / 每次 launch 的耗时          真测量
//   cudaDeviceProp             -> SpeedOfLight 里的理论峰值内存带宽        真测量
//   cudaFuncGetAttributes      -> LaunchStats（寄存器、静态 shared mem）   真测量
//   cudaOccupancyMaxActive*    -> Occupancy（理论占用率）                 真测量
//   地址模型（见下）            -> L1TEX sectors/request 等计数器           解析推导
//
// 最后一项是唯一的"假"数据：它不跑硬件计数器，而是从访问模式直接算出一次
// warp load 会碰到多少个 32B sector。对 hw4 这种规整访问它和 ncu 的结果一致，
// 对访存地址不规则的 kernel 只能给上界。
//
// 用法（所有接口都在 namespace myprof 下）：
//
//   myprof::print_device_info();
//   double bytes = (double)DSIZE * DSIZE * sizeof(float);
//   auto t = myprof::time_launches([&]{ row_sums<<<grid, block>>>(d_A, d_sums, DSIZE); }, bytes);
//   int sectors = myprof::sectors_uniform_stride(/*stride_bytes=*/4);
//   myprof::print_row("row_sums", t, sectors, /*elemsize=*/4);
//
// 编译：不需要额外链接任何东西，nvcc 直接编即可。

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace myprof {

#define MYPROF_CHECK(call)                                                     \
  do {                                                                         \
    cudaError_t myprof_err_ = (call);                                          \
    if (myprof_err_ != cudaSuccess) {                                          \
      std::fprintf(stderr, "myprof: %s failed: %s (%s:%d)\n", #call,           \
                   cudaGetErrorString(myprof_err_), __FILE__, __LINE__);       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------- 设备信息 --

inline const cudaDeviceProp& device_info(int dev = 0) {
  static cudaDeviceProp props[16];
  static bool loaded[16] = {false};
  int d = (dev >= 0 && dev < 16) ? dev : 0;
  if (!loaded[d]) {
    MYPROF_CHECK(cudaGetDeviceProperties(&props[d], d));
    loaded[d] = true;
  }
  return props[d];
}

// 理论峰值 DRAM 带宽，单位 GB/s。对应 ncu SpeedOfLight 里那个 100% 的刻度。
inline double peak_bandwidth_gbs(int dev = 0) {
  const cudaDeviceProp& p = device_info(dev);
  // memoryClockRate 是 kHz；GDDR/HBM 每个时钟沿传两次。
  double hz = (double)p.memoryClockRate * 1e3;
  return hz * 2.0 * (double)p.memoryBusWidth / 8.0 / 1e9;
}

inline void print_device_info(int dev = 0) {
  const cudaDeviceProp& p = device_info(dev);
  std::printf("myprof (ncu 替身，无需计数器权限)\n");
  std::printf("device : %s  sm_%d%d  %d SMs  %d threads/SM\n", p.name, p.major,
              p.minor, p.multiProcessorCount, p.maxThreadsPerMultiProcessor);
  std::printf("dram   : %.1f GB/s peak (%.2f GHz x %d-bit)\n",
              peak_bandwidth_gbs(dev),
              (double)p.memoryClockRate / 1e6, p.memoryBusWidth);
}

// ------------------------------------------------------------------ 计时 ----

struct Timing {
  double min_ms = 0.0;    // 最快的一次，ncu 报的 Duration 也接近这个口径
  double med_ms = 0.0;
  double mean_ms = 0.0;
  double gbs = 0.0;       // useful_bytes / 最快耗时
  double pct_peak = 0.0;  // gbs 占理论峰值的百分比
  int reps = 0;
};

// 反复 launch 一个 kernel（用一个只负责放 kernel 的 lambda 包起来），
// 每次 launch 前后打 event。warmup 次不计入统计。
inline Timing time_launches(const std::function<void()>& launch,
                            double useful_bytes, int reps = 30, int warmup = 5,
                            int dev = 0) {
  for (int i = 0; i < warmup; ++i) launch();
  MYPROF_CHECK(cudaDeviceSynchronize());

  std::vector<cudaEvent_t> start(reps), stop(reps);
  for (int i = 0; i < reps; ++i) {
    MYPROF_CHECK(cudaEventCreate(&start[i]));
    MYPROF_CHECK(cudaEventCreate(&stop[i]));
  }
  for (int i = 0; i < reps; ++i) {
    MYPROF_CHECK(cudaEventRecord(start[i]));
    launch();
    MYPROF_CHECK(cudaEventRecord(stop[i]));
  }
  MYPROF_CHECK(cudaDeviceSynchronize());

  std::vector<double> t(reps);
  for (int i = 0; i < reps; ++i) {
    float ms = 0.0f;
    MYPROF_CHECK(cudaEventElapsedTime(&ms, start[i], stop[i]));
    t[i] = (double)ms;
    MYPROF_CHECK(cudaEventDestroy(start[i]));
    MYPROF_CHECK(cudaEventDestroy(stop[i]));
  }
  std::sort(t.begin(), t.end());

  Timing r;
  r.reps = reps;
  r.min_ms = t.front();
  r.med_ms = (reps % 2) ? t[reps / 2] : 0.5 * (t[reps / 2 - 1] + t[reps / 2]);
  double sum = 0.0;
  for (double x : t) sum += x;
  r.mean_ms = sum / reps;
  if (r.min_ms > 0.0) {
    r.gbs = useful_bytes / (r.min_ms * 1e-3) / 1e9;
    double peak = peak_bandwidth_gbs(dev);
    r.pct_peak = (peak > 0.0) ? 100.0 * r.gbs / peak : 0.0;
  }
  return r;
}

// ------------------------------------------------------- 访问模式 -> sector --
//
// ncu 用 l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum 和
// l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum 这两个计数器算
// "transactions per request"。计数器拿不到，我们直接用地址算：
// L1TEX 以 32B 为单位在内存里搬运，所以一个 warp 的一条 load 指令碰到的
// 不同 32B sector 个数，就是它要付的"交易数"。

inline int sectors_per_request(const std::vector<long long>& addr_bytes) {
  std::vector<long long> s;
  s.reserve(addr_bytes.size());
  for (long long a : addr_bytes) s.push_back(a / 32);
  std::sort(s.begin(), s.end());
  s.erase(std::unique(s.begin(), s.end()), s.end());
  return (int)s.size();
}

// 相邻线程地址差固定为 stride_bytes 的情形（连续访问 stride=sizeof(T)）。
inline int sectors_uniform_stride(long long stride_bytes, int warp = 32) {
  std::vector<long long> a;
  a.reserve(warp);
  for (int t = 0; t < warp; ++t) a.push_back((long long)t * stride_bytes);
  return sectors_per_request(a);
}

// 一次 warp load 真正需要的字节 / L1TEX 实际搬运的字节。1.0 = 完美合并。
inline double coalescing_efficiency(int elemsize, int sectors, int warp = 32) {
  double need = (double)warp * (double)elemsize;
  double moved = (double)sectors * 32.0;
  return (moved > 0.0) ? need / moved : 0.0;
}

// ------------------------------------------------------- kernel 资源 / 占用 --

// 对应 ncu 的 LaunchStats + Occupancy section。
inline void print_kernel_resources(const char* name, const void* kernel,
                                   int block_threads, size_t dynamic_smem = 0) {
  cudaFuncAttributes attr{};
  MYPROF_CHECK(cudaFuncGetAttributes(&attr, kernel));
  int blocks_per_sm = 0;
  MYPROF_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, block_threads, dynamic_smem));
  const cudaDeviceProp& p = device_info();
  double occ = (p.maxThreadsPerMultiProcessor > 0)
                   ? 100.0 * blocks_per_sm * block_threads /
                         p.maxThreadsPerMultiProcessor
                   : 0.0;
  std::printf(
      "  %-14s regs/thread=%-3d static_smem=%-5zuB dyn_smem=%-4zuB "
      "blocks/SM=%-2d occupancy=%.1f%%\n",
      name, attr.numRegs, attr.sharedSizeBytes, dynamic_smem, blocks_per_sm,
      occ);
}

// ------------------------------------------------------------------ 报表 ----

inline void print_table_header() {
  std::printf("\n%-24s %10s %10s %10s %8s %9s %7s\n", "kernel", "min(ms)",
              "med(ms)", "GB/s", "%peak", "sect/req", "eff");
  std::printf("%s\n", std::string(84, '-').c_str());
}

// sectors < 0 表示该 kernel 的访问模式没有建模，只报耗时。
inline void print_row(const char* name, const Timing& t, int sectors,
                      int elemsize, int warp = 32) {
  if (sectors >= 0) {
    char sect[32], eff[32];
    std::snprintf(sect, sizeof(sect), "%d", sectors);
    std::snprintf(eff, sizeof(eff), "%.1f%%",
                  100.0 * coalescing_efficiency(elemsize, sectors, warp));
    std::printf("%-24s %10.4f %10.4f %10.1f %7.1f%% %9s %7s\n", name,
                t.min_ms, t.med_ms, t.gbs, t.pct_peak, sect, eff);
  } else {
    std::printf("%-24s %10.4f %10.4f %10.1f %7.1f%% %9s %7s\n", name,
                t.min_ms, t.med_ms, t.gbs, t.pct_peak, "n/a", "n/a");
  }
}

}  // namespace myprof
