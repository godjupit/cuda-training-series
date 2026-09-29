// vector_add_timing.cu
//
// Timing harness around the hw3 grid-stride vector add.
//
// The kernel is identical to the one in vector_add.cu. Because it uses a
// grid-stride loop, ANY (blocks, threads) combination produces correct
// results -- which is exactly what makes it a good vehicle for a sweep.
//
// Usage:
//   ./vector_add_timing                              # built-in config sweep
//   ./vector_add_timing <dsize>                      # sweep, custom length
//   ./vector_add_timing <dsize> <blocks> <threads>   # time one config only
//
// Examples:
//   ./vector_add_timing
//   ./vector_add_timing 16777216 160 1024
//
// NVTX ranges: every phase below is bracketed by an NVTX range so the run is
// readable on a timeline. The range nesting is
//
//     setup: malloc + H2D
//     <config label>
//         warmup
//         timed            <- the reps that produce the reported number
//     verify
//
// which means the timeline shows exactly which block of kernels belongs to
// which (blocks, threads) configuration and to the untimed warm-up:
//
//   nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true \
//        -o /tmp/vadd_timing ./vector_add_timing
//
// Build with -DNVTX_DISABLE to compile the range calls out entirely.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// NVTX: push/pop named ranges onto the timeline. NVTX3 ships with the CUDA
// toolkit as a header-only API, so no extra link flag is required.
#ifdef NVTX_DISABLE
  #define NVTX_PUSH(name) ((void)0)
  #define NVTX_POP()      ((void)0)
#else
  #include <nvtx3/nvToolsExt.h>
  #define NVTX_PUSH(name) nvtxRangePushA(name)
  #define NVTX_POP()      nvtxRangePop()
#endif

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


// vector add kernel: C = A + B  (grid-stride loop, straight from hw3)
__global__ void vadd(const float *A, const float *B, float *C, int ds) {

  for (int idx = threadIdx.x+blockDim.x*blockIdx.x; idx < ds; idx+=gridDim.x*blockDim.x)
    C[idx] = B[idx] + A[idx];
}

struct BenchResult {
  double ms;    // best observed kernel duration, milliseconds
  double gbs;   // achieved memory bandwidth, GB/s
};

// Time one launch configuration with CUDA events.
//
// Why events instead of a host-side clock? The device is asynchronous: the
// host runs ahead of the GPU, so wrapping the launch in clock() measures
// launch overhead, not the kernel. cudaEventRecord inserts a timestamp into
// the GPU's own stream, so we get the actual kernel duration.
//
// We report the *best* of several repetitions: the GPU is shared and noisy,
// and the fastest run is the one least perturbed by other activity.
//
// `label` names the NVTX range covering the whole measurement, and the warm-up
// and timed repetitions get their own nested ranges inside it.
static struct BenchResult bench(const char *label, int blocks, int threads,
                                const float *d_A, const float *d_B, float *d_C,
                                int ds) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  NVTX_PUSH(label);

  // Warm-up launch: pays for lazily-created context/module state once, so it
  // is not charged to the measurement.
  NVTX_PUSH("warmup");
  vadd<<<blocks, threads>>>(d_A, d_B, d_C, ds);
  cudaCheckErrors("kernel launch failure (warmup)");
  cudaDeviceSynchronize();
  NVTX_POP();

  NVTX_PUSH("timed");
  // First timed run; also tells us how many repetitions fit in our budget.
  cudaEventRecord(start);
  vadd<<<blocks, threads>>>(d_A, d_B, d_C, ds);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float first = 0.0f;
  cudaEventElapsedTime(&first, start, stop);

  // Fast kernels get many repetitions; a slow config (1 block x 1 thread)
  // gets measured only once.
  int reps = (first > 0.0f) ? (int)(200.0f / first) : 1;
  if (reps < 1)  reps = 1;
  if (reps > 50) reps = 50;

  double best = (double)first;
  for (int r = 1; r < reps; r++) {
    cudaEventRecord(start);
    vadd<<<blocks, threads>>>(d_A, d_B, d_C, ds);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    if ((double)ms < best) best = (double)ms;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  NVTX_POP();   // timed

  // Traffic per element: read A, read B, write C -> 3 * ds * sizeof(float).
  double bytes = 3.0 * (double)ds * sizeof(float);
  struct BenchResult res;
  res.ms  = best;
  res.gbs = bytes / (best * 1.0e6);
  NVTX_POP();   // label
  return res;
}

int main(int argc, char **argv) {

  int ds = 4*1024*1024;                       // default: 4M elements
  if (argc >= 2) ds = atoi(argv[1]);

  int dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  cudaCheckErrors("cudaGetDeviceProperties failure");

  int sms             = prop.multiProcessorCount;
  int threadsPerSM    = prop.maxThreadsPerMultiProcessor;
  int residentThreads = sms * threadsPerSM;   // threads needed to fill every SM

  printf("Device: %s\n", prop.name);
  printf("SMs: %d   maxThreadsPerSM: %d   =>  a fully-occupying grid is ~%d threads\n",
         sms, threadsPerSM, residentThreads);
  printf("Vector length: %d elements  (%.1f MB per array, %.1f MB total traffic)\n\n",
         ds, ds*sizeof(float)/1.0e6, 3.0*ds*sizeof(float)/1.0e6);

  NVTX_PUSH("setup: malloc + H2D");

  // Host + device buffers.
  float *h_A = (float*)malloc(ds*sizeof(float));
  float *h_B = (float*)malloc(ds*sizeof(float));
  float *h_C = (float*)malloc(ds*sizeof(float));
  for (int i = 0; i < ds; i++) {
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
    h_C[i] = 0;
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, ds*sizeof(float));
  cudaMalloc(&d_B, ds*sizeof(float));
  cudaMalloc(&d_C, ds*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");

  cudaMemcpy(d_A, h_A, ds*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, ds*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  NVTX_POP();

  // ------------------------------------------------------------------
  // Single-configuration mode: ./vector_add_timing <dsize> <blocks> <threads>
  // (handy for the three case studies in the readme, or under a profiler)
  // ------------------------------------------------------------------
  if (argc >= 4) {
    int blocks  = atoi(argv[2]);
    int threads = atoi(argv[3]);
    char label[64];
    snprintf(label, sizeof(label), "blocks=%d threads=%d", blocks, threads);
    struct BenchResult r = bench(label, blocks, threads, d_A, d_B, d_C, ds);
    printf("%8s %8s %12s %10s\n", "blocks", "threads", "time(ms)", "GB/s");
    printf("%8d %8d %12.4f %10.1f\n", blocks, threads, r.ms, r.gbs);
  } else {
    // ----------------------------------------------------------------
    // Sweep mode: a few pedagogical points plus occupancy-matched grids.
    // ----------------------------------------------------------------
    struct Config { int blocks; int threads; const char *label; } cfgs[24];
    char dynlabel[24][48];
    int n = 0;

    cfgs[n].blocks = 1; cfgs[n].threads = 1;
    cfgs[n].label = "1 block x 1 thread (baseline)";                    n++;
    cfgs[n].blocks = 1; cfgs[n].threads = 32;
    cfgs[n].label = "1 block x 32 threads";                             n++;
    cfgs[n].blocks = 1; cfgs[n].threads = 1024;
    cfgs[n].label = "1 block x 1024 threads";                           n++;

    // "Occupancy 1x" == exactly enough blocks to fill the whole GPU.
    int threadChoices[] = {128, 256, 512, 1024};
    for (int t = 0; t < 4; t++) {
      int th = threadChoices[t];
      int bl = residentThreads / th;
      if (bl < 1) bl = 1;
      snprintf(dynlabel[n], sizeof(dynlabel[n]), "occupancy 1x (%d blocks %d thr)", bl, th);
      cfgs[n].blocks = bl; cfgs[n].threads = th; cfgs[n].label = dynlabel[n]; n++;
    }

    {
      int th = 1024, bl = 2 * residentThreads / 1024;
      snprintf(dynlabel[n], sizeof(dynlabel[n]), "occupancy 2x (%d blocks %d thr)", bl, th);
      cfgs[n].blocks = bl; cfgs[n].threads = th; cfgs[n].label = dynlabel[n]; n++;
    }

    printf("%-32s %8s %8s %12s %10s\n",
           "config", "blocks", "threads", "time(ms)", "GB/s");
    printf("%-32s %8s %8s %12s %10s\n",
           "--------------------------------", "------", "-------", "--------", "------");

    for (int c = 0; c < n; c++) {
      struct BenchResult r = bench(cfgs[c].label, cfgs[c].blocks, cfgs[c].threads,
                                   d_A, d_B, d_C, ds);
      cudaCheckErrors("kernel execution failure");
      printf("%-32s %8d %8d %12.4f %10.1f\n",
             cfgs[c].label, cfgs[c].blocks, cfgs[c].threads, r.ms, r.gbs);
    }
  }

  // Correctness check (the last configuration that ran left h_C populated).
  NVTX_PUSH("verify");
  cudaMemcpy(h_C, d_C, ds*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy D2H failure");
  double maxerr = 0.0;
  for (int i = 0; i < ds; i++) {
    double e = fabs((double)h_C[i] - ((double)h_A[i] + (double)h_B[i]));
    if (e > maxerr) maxerr = e;
  }
  printf("\nverify: max abs error = %g  ->  %s\n",
         maxerr, (maxerr < 1.0e-5) ? "PASS" : "FAIL");
  NVTX_POP();

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(h_A); free(h_B); free(h_C);
  return (maxerr < 1.0e-5) ? 0 : 1;
}
