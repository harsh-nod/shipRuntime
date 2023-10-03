#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <mutex>
#include "utils.h"

// Large parts of this code have been borrowed from
// the rocWMMA repository.

using float16_t = uint16_t;
using float32_t = float;
constexpr uint32_t recordRuns = 100u;

template <typename DataT>
std::pair<bool, double>
compareEqual(DataT const* a, DataT const* b, uint32_t size, uint32_t reductionDim, double tolerance = 10.0) {
    bool   retval             = true;
    double max_relative_error = 0.0;
    // Scale tolerance with reduction dim
    tolerance *= reductionDim < 1024 ? 1 : reductionDim / 1024;

    // Some types don't have direct conversion to double.
    // Convert to float first then to double.
    auto toDouble = [](DataT const& val) { return static_cast<double>(static_cast<float>(val)); };

    bool       isInf = false;
    bool       isNaN = false;
    std::mutex writeMutex;

#pragma omp parallel for
    for(int i = 0; i < size; ++i)
    {
        auto valA = a[i];
        auto valB = b[i];

        auto numerator = fabs(toDouble(valA) - toDouble(valB));
        auto divisor   = fabs(toDouble(valA)) + fabs(toDouble(valB)) + 1.0;

        if(std::isinf(numerator) || std::isinf(divisor))
        {
#pragma omp atomic
            isInf |= true;
        }
        else
        {
            auto relative_error = numerator / divisor;
            if(std::isnan(relative_error))
            {
#pragma omp atomic
                isNaN |= true;
            }
            else if(relative_error > max_relative_error)
            {
                const std::lock_guard<std::mutex> guard(writeMutex);
                // Double check in case of stall
                if(relative_error > max_relative_error)
                {
                    max_relative_error = relative_error;
                }
            }
        }

        if(isInf || isNaN)
        {
            i = size;
        }
    }

    auto eps = toDouble(std::numeric_limits<DataT>::epsilon());
    if(isInf)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DataT>::infinity();
    }
    else if(isNaN)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DataT>::signaling_NaN();
    }
    else if(max_relative_error > (eps * tolerance))
    {
        retval = false;
    }

    return std::make_pair(retval, max_relative_error);
}

struct row_major{};
struct col_major{};

// Host GEMM validation
template <typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutD>
__host__ void gemm_cpu_h(uint32_t       m,
                         uint32_t       n,
                         uint32_t       k,
                         InputT const*  a,
                         InputT const*  b,
                         OutputT*       d,
                         uint32_t       lda,
                         uint32_t       ldb,
                         uint32_t       ldd,
                         ComputeT       alpha,
                         ComputeT       beta)
{
    auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
    auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

    auto aIndex = std::is_same<LayoutA, row_major>::value ? rowMjr : colMjr;
    auto bIndex = std::is_same<LayoutB, row_major>::value ? rowMjr : colMjr;
    auto dIndex = std::is_same<LayoutD, row_major>::value ? rowMjr : colMjr;

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            ComputeT accum = static_cast<ComputeT>(0);
            for(int h = 0; h < k; ++h) {
                accum += static_cast<ComputeT>(half2float(a[aIndex(i, h, lda)], FP16_EXP_BITS))
                         * static_cast<ComputeT>(half2float(b[bIndex(h, j, ldb)], FP16_EXP_BITS));
            }
            d[dIndex(i, j, ldd)] = static_cast<OutputT>(alpha * accum);
        }
    }
}

inline double calculateGFlops(uint32_t m, uint32_t n, uint32_t k) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * 1.0e-9;
}

inline double calculateTFlopsPerSec(
    uint32_t m, uint32_t n, uint32_t k, double elapsedTimeMs, uint32_t repeats = 1u) {
    // elapsedTimeMs is over all iterations
    return calculateGFlops(m, n, k) / elapsedTimeMs * static_cast<double>(repeats);
}

template <typename DataT>
static inline void fillRand(DataT* mat, uint32_t m, uint32_t n)
{
#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; j++)
        {
	    // Random values normalized such that output is between 0 and 1
	    float original = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	    float16_t value = float2half(original, FP16_EXP_BITS);
            mat[i * n + j] = static_cast<DataT>(value);
        }
    }
}

void benchmark_module(int m, int n, int k, int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, int sharedMemBytes, const char *data, const char *name) {

    // Initialize input matrices
    std::vector<float16_t> matrixA(m * k);
    std::vector<float16_t> matrixB(k * n);
    // Fill outputs with NaN to catch contamination
    std::vector<float32_t> matrixD(m * n, std::numeric_limits<float32_t>::signaling_NaN());

    fillRand(matrixA.data(), m, k);
    fillRand(matrixB.data(), k, n);

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    float16_t* d_a;
    float16_t* d_b;
    float32_t* d_d;

    const size_t bytesA = matrixA.size() * sizeof(float16_t);
    const size_t bytesB = matrixB.size() * sizeof(float16_t);
    const size_t bytesD = matrixD.size() * sizeof(float32_t);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

    hipModule_t module;
    hipFunction_t function;
    // Open HSACO file
    FILE *hsaco_file;
    if ((hsaco_file = fopen(data, "rb")) == NULL) {
      return;
    }

    // Read HSCAO file into Buffer
    fseek(hsaco_file, 0L, SEEK_END);
    size_t hsaco_file_size = ftell(hsaco_file);
    unsigned char *hsaco =
        (unsigned char *)malloc(hsaco_file_size * sizeof(unsigned char));
    rewind(hsaco_file);
    size_t result = fread(hsaco, sizeof(unsigned char), hsaco_file_size, hsaco_file);
    printf("Read %zu bytes\n", result);
    fclose(hsaco_file);

    // set HIP options
    hipJitOption opt[] = {hipJitOptionErrorLogBufferSizeBytes,
                          hipJitOptionErrorLogBuffer,
                          hipJitOptionInfoLogBufferSizeBytes,
                          hipJitOptionInfoLogBuffer, hipJitOptionLogVerbose};
    const unsigned int errbufsize = 8192;
    const unsigned int logbufsize = 8192;
    char _err[errbufsize];
    char _log[logbufsize];
    void *optval[] = {(void *)(uintptr_t)errbufsize, (void *)_err,
                      (void *)(uintptr_t)logbufsize, (void *)_log, (void *)1};

    CHECK_HIP_ERROR(hipModuleLoadDataEx(&module, hsaco, 5, opt, optval));
    CHECK_HIP_ERROR(hipModuleGetFunction(&function, module, name));
    free(hsaco);

    // Create and fill array with kernel arguments
    struct {
      hipDeviceptr_t _d_a;
      hipDeviceptr_t _d_b;
      hipDeviceptr_t _d_d;
    } args{d_a, d_b, d_d};

    size_t args_size = sizeof(args);

    // Create array with kernel arguments and its size.
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &args_size,
                      HIP_LAUNCH_PARAM_END};

    std::cout << "Launching GEMM kernel..." << std::endl;

    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    uint64_t _stream;
    for (uint32_t i = 0; i < recordRuns; ++i) {
      CHECK_HIP_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
        nullptr, nullptr, (void **)&config));
    }
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));
    CHECK_HIP_ERROR(hipModuleUnload(module));

    // GEMM flops converge to 2*mnk
    auto gFlops       = calculateGFlops(m, n, k);
    auto tFlopsPerSec = calculateTFlopsPerSec(m, n, k, static_cast<double>(elapsedTimeMs), recordRuns);

#if !NDEBUG

    std::cout << "Validating result with reference..." << std::endl;

    // Bring kernel result back to host
    CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

    // Setup and run reference computation
    std::vector<float32_t> matrixD_ref(m * n, std::numeric_limits<float32_t>::signaling_NaN());
    int lda = k;
    int ldb = k;
    int ldd = n;
    float alpha = 1.0;
    float beta = 1.0;
    gemm_cpu_h<float16_t, float32_t, float32_t, row_major, col_major, row_major>(m,
                                                                                 n,
                                                                                 k,
                                                                                 matrixA.data(),
                                                                                 matrixB.data(),
                                                                                 matrixD_ref.data(),
                                                                                 lda,
                                                                                 ldb,
                                                                                 ldd,
                                                                                 alpha,
                                                                                 beta);

    auto res = compareEqual<float32_t>(matrixD.data(), matrixD_ref.data(), m * n, k);

    if(std::get<0>(res) == false) {
        std::cout << "FAILED!\n";
    } else {
        std::cout << "PASSED!\n";
    }

    std::cout << "Max relative error: " << std::get<1>(res) << std::endl;

#endif // !NDEBUG

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "TFLOPS/sec = " << tFlopsPerSec << std::endl;
    std::cout << "Finished!" << std::endl;

}

int main(int argc, char *argv[]) {
  if (argc != 13) {
    std::cout << "Usage: " << argv[0] << " M N K gX gY gZ bX bY bZ sMemBytes hsaco-file func-name" << std::endl;
    return 1;
  }
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int gridX = atoi(argv[4]);
  int gridY = atoi(argv[5]);
  int gridZ = atoi(argv[6]);
  int blockX = atoi(argv[7]);
  int blockY = atoi(argv[8]);
  int blockZ = atoi(argv[9]);
  int sharedMemBytes = atoi(argv[10]);
  const char *data = argv[11];
  const char *name = argv[12];
  std::cout << "Benchmarking matmul with M = " << M << " , N = " << N << " , K = " << K << std::endl;
  std::cout << "Launch Config: GridX = " << gridX << " , GridY = " << gridY << " , GridZ = " << gridZ << std::endl;
  std::cout << "Launch Config: BlockX = " << blockX << " , BlockY = " << blockY << " , BlockZ = " << blockZ << std::endl;
  std::cout << "Launch Config: SMemBytes = " << sharedMemBytes << std::endl;
  std::cout << "Executing ... " << name << " from " << data << std::endl;
  benchmark_module(M, N, K, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes, data, name);
  return 0;
}
