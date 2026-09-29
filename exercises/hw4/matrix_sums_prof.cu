// hw4 的 profiling 练习，不用 ncu，也不需要任何计数器权限。
//
// 编译：nvcc -arch=native -o matrix_sums_prof matrix_sums_prof.cu
//       （没有 GPU 时用 -arch=sm_75 之类的具体架构）
// 运行：./matrix_sums_prof [dsize]      dsize 默认 16384，调小可以跑得快
//
// 它回答 hw4 readme 的两个问题：
//   1) 两个 kernel 的 duration 一样吗？          -> min/med 列，cudaEvent 实测
//   2) global load 的效率一样吗？为什么？        -> sect/req + eff 列，由访问模式推出
// ncu 的对应关系：Duration <- cudaEvent，l1tex__*_sectors/requests <- 地址模型，
// SpeedOfLight 的百分比 <- 实测带宽 / cudaDeviceProp 里的理论峰值。
#include "../../tools/myprof.cuh"

#include <cstdio>
#include <cstdlib>

static size_t DSIZE = 16384;
static const int BLOCK = 256;

// 一行一个线程，沿行方向迭代。注意：同一个 warp 里相邻线程的 idx 差 1，
// 而地址是 idx*ds + i，所以两条相邻线程的地址差 ds*4 字节 —— 一条 load
// 指令跨 32 条不同的 cache line。
__global__ void row_sums(const float *A, float *sums, size_t ds) {
  size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
  if (idx < ds) {
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++) sum += A[idx * ds + i];
    sums[idx] = sum;
  }
}

// 一列一个线程，沿列方向迭代。地址是 idx + ds*i，i 固定时相邻线程地址差
// 4 字节 —— 一个 warp 一条指令正好 128B 连续，4 个 sector。
__global__ void column_sums(const float *A, float *sums, size_t ds) {
  size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
  if (idx < ds) {
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++) sum += A[idx + ds * i];
    sums[idx] = sum;
  }
}

int main(int argc, char **argv) {
  if (argc > 1) DSIZE = (size_t)std::strtoull(argv[1], nullptr, 10);
  const size_t n = DSIZE * DSIZE;
  const double mat_bytes = (double)n * sizeof(float);

  myprof::print_device_info();
  std::printf("DSIZE  : %zu  (matrix %.0f MiB, block %d threads)\n", DSIZE,
              mat_bytes / 1048576.0, BLOCK);

  float *h_A = new float[n];
  float *h_sums = new float[DSIZE]();
  for (size_t i = 0; i < n; i++) h_A[i] = 1.0f;

  float *d_A = nullptr, *d_sums = nullptr;
  MYPROF_CHECK(cudaMalloc(&d_A, n * sizeof(float)));
  MYPROF_CHECK(cudaMalloc(&d_sums, DSIZE * sizeof(float)));
  MYPROF_CHECK(cudaMemcpy(d_A, h_A, n * sizeof(float),
                                  cudaMemcpyHostToDevice));

  const unsigned grid = (unsigned)((DSIZE + BLOCK - 1) / BLOCK);

  // 读整个矩阵 + 写 DSIZE 个结果，这就是"有用字节"。
  const double useful_bytes = mat_bytes + (double)DSIZE * sizeof(float);

  myprof::Timing t_row = myprof::time_launches(
      [&] { row_sums<<<grid, BLOCK>>>(d_A, d_sums, DSIZE); }, useful_bytes);
  MYPROF_CHECK(cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float),
                                  cudaMemcpyDeviceToHost));
  bool row_ok = true;
  for (size_t i = 0; i < DSIZE; i++)
    if (h_sums[i] != (float)DSIZE) { row_ok = false; break; }

  MYPROF_CHECK(cudaMemset(d_sums, 0, DSIZE * sizeof(float)));
  myprof::Timing t_col = myprof::time_launches(
      [&] { column_sums<<<grid, BLOCK>>>(d_A, d_sums, DSIZE); }, useful_bytes);
  MYPROF_CHECK(cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float),
                                  cudaMemcpyDeviceToHost));
  bool col_ok = true;
  for (size_t i = 0; i < DSIZE; i++)
    if (h_sums[i] != (float)DSIZE) { col_ok = false; break; }

  // readme 想看的 l1tex__t_sectors / l1tex__t_requests，这里按地址模式推：
  // row_sums    -> warp 内相邻线程差 ds*sizeof(float) 字节，32 条不同 cache line
  // column_sums -> warp 内相邻线程差 sizeof(float) 字节，128B 连续
  const int row_sectors =
      myprof::sectors_uniform_stride((long long)DSIZE * sizeof(float));
  const int col_sectors = myprof::sectors_uniform_stride(sizeof(float));

  myprof::print_table_header();
  myprof::print_row("row_sums", t_row, row_sectors, sizeof(float));
  myprof::print_row("column_sums", t_col, col_sectors, sizeof(float));

  std::printf("\n");
  myprof::print_kernel_resources("row_sums", (const void *)row_sums, BLOCK);
  myprof::print_kernel_resources("column_sums", (const void *)column_sums, BLOCK);

  std::printf(
      "\nrow_sums    : warp 内相邻线程地址差 %lld B -> 一条 load 指令打 "
      "%d 个 sector\n"
      "column_sums : warp 内 128B 连续            -> 一条 load 指令只要 "
      "%d 个 sector\n"
      "=> 搬同样的有用字节，row_sums 在 L1TEX 侧要多付 %.1fx 的 sector。\n"
      "   DRAM 侧两者最终都会把整行读完（row 靠 L1 保留 line 复用），流量相同，\n"
      "   所以差距体现在 L1TEX/LSU 吞吐上，而不是峰值 DRAM 带宽上。\n",
      (long long)DSIZE * (long long)sizeof(float), row_sectors, col_sectors,
      (double)row_sectors / (double)col_sectors);

  std::printf("\nresult : row sums %s, column sums %s\n",
              row_ok ? "correct" : "MISMATCH", col_ok ? "correct" : "MISMATCH");

  delete[] h_A;
  delete[] h_sums;
  MYPROF_CHECK(cudaFree(d_A));
  MYPROF_CHECK(cudaFree(d_sums));
  return 0;
}
