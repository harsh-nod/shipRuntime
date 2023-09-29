#include <iostream>
#include <vector>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

using float16_t = _Float16;

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(status)                   \
    if(status != hipSuccess)                      \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(status),        \
                status,                           \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

inline double calculateGFlops(uint32_t m, uint32_t n, uint32_t k) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * 1.0e-9;
}

template <typename DataT>
__host__ static inline void fillRand(DataT* mat, uint32_t m, uint32_t n)
{
    auto randInit = []() {
        srand(time(0));
        return 0u;
    };
    static auto init = randInit();
#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
        auto rando = rand() % 5u;
        for(int j = 0; j < n; j++)
        {
            // Assign random integer values within 0-64, alternating
            // sign if the value is a multiple of 3
            auto value     = (rando + j) % 5u;
            mat[i * n + j] = ((value % 3u == 0u) && std::is_signed<DataT>::value)
                                 ? -static_cast<DataT>(value)
                                 : static_cast<DataT>(value);
        }
    }
}

void benchmark_module(int m, int n, int k) {

    // Initialize input matrices
    std::vector<float16_t> matrixA(m * k);
    std::vector<float16_t> matrixB(k * n);
    std::vector<float16_t> matrixC(m * n);
    // Fill outputs with NaN to catch contamination
    std::vector<float16_t> matrixD(m * n, std::numeric_limits<float16_t>::signaling_NaN());

    fillRand(matrixA.data(), m, k);
    fillRand(matrixB.data(), k, n);
    fillRand(matrixC.data(), m, n);

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    float16_t* d_a;
    float16_t* d_b;
    float16_t* d_c;
    float16_t* d_d;

    const size_t bytesA = matrixA.size() * sizeof(float16_t);
    const size_t bytesB = matrixB.size() * sizeof(float16_t);
    const size_t bytesC = matrixC.size() * sizeof(float16_t);
    const size_t bytesD = matrixD.size() * sizeof(float16_t);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

    std::cout << "Launching GEMM kernel..." << std::endl;

    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // GEMM flops converge to 2*mnk
    auto gFlops       = calculateGFlops(m, n, k);
    auto tFlopsPerSec = gFlops / static_cast<double>(elapsedTimeMs);

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "TFLOPS/sec = " << tFlopsPerSec << std::endl;
    std::cout << "Finished!" << std::endl;

}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " M N K" << std::endl;
    return 1;
  }
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  std::cout << "Benchmarking matmul with M = " << M << " , N = " << N << " , K = " << K << std::endl;
  benchmark_module(M, N, K);
  return 0;
}
