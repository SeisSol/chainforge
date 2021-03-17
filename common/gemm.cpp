#include "gemm.h"
#include <iostream>

#define GEMMGEN 1
#define OPENBLAS 2

#define CPU_BACKEND CONCRETE_CPU_BACKEND

#if CPU_BACKEND == OPENBLAS
#include <cblas.h>
#endif

using namespace cf::reference;

namespace cf {
  namespace reference {

    void singleGemm(LayoutType TypeA,
                    LayoutType TypeB,
                    int M, int N, int K,
                    real Alpha, real *A, int Lda,
                    real *B, int Ldb,
                    real Beta, real *C, int Ldc) {

      int NumRowA{}, NumColA{}, NumRowB{};

      if (TypeA == LayoutType::NoTrans) {
        NumRowA = M;
        NumColA = K;
      } else {
        NumRowA = K;
        NumColA = M;
      }

      if (TypeB == LayoutType::NoTrans) {
        NumRowB = K;
      } else {
        NumRowB = N;
      }

      if (Alpha == 0.0) {
        for (int j = 0; j < N; ++j) {
          for (int i = 0; i < M; ++i) {
            C[i + j * Ldc] = Beta * C[i + j * Ldc];
          }
        }
        return;
      }

      if (TypeB == LayoutType::NoTrans) {
        if (TypeA == LayoutType::NoTrans) {
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              real Temp{0.0};
              for (int k = 0; k < K; ++k) {
                Temp += A[m + k * Lda] * B[k + n * Ldb];
              }
              C[m + n * Ldc] = Alpha * Temp + Beta * C[m + n * Ldc];
            }
          }
        } else {
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              real Temp{0.0};
              for (int k = 0; k < K; ++k) {
                Temp += A[k + m * Lda] * B[k + n * Ldb];
              }
              C[m + n * Ldc] = Alpha * Temp + Beta * C[m + n * Ldc];
            }
          }
        }
      } else {
        if (TypeA == LayoutType::NoTrans) {
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              real Temp{0.0};
              for (int k = 0; k < K; ++k) {
                Temp += A[m + k * Lda] * B[n + k * Ldb];
              }
              C[m + n * Ldc] = Alpha * Temp + Beta * C[m + n * Ldc];
            }
          }
        } else {
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              real Temp{0.0};
              for (int k = 0; k < K; ++k) {
                Temp += A[k + m * Lda] * B[n + k * Ldb];
              }
              C[m + n * Ldc] = Alpha * Temp + Beta * C[m + n * Ldc];
            }
          }
        }
      }
    }

    void gemm(LayoutType typeA,
              LayoutType typeB,
              int m, int n, int k,
              real alpha, real *a, int lda,
              real *b, int ldb,
              real beta, real *c, int ldc) {

#if CPU_BACKEND == GEMMGEN
      singleGemm(typeA, typeB,
                 m, n, k,
                 alpha, a, lda,
                 b, ldb,
                 beta, c, ldc);

#elif CPU_BACKEND == OPENBLAS
      CBLAS_LAYOUT layout = CblasColMajor;
      CBLAS_TRANSPOSE transA = typeA == LayoutType::Trans ? CblasTrans : CblasNoTrans;
      CBLAS_TRANSPOSE transB = typeB == LayoutType::Trans ? CblasTrans : CblasNoTrans;

#if REAL_SIZE == 4
      cblas_sgemm(layout, transA, transB,
                  m, n, k,
                  alpha, a, lda,
                  b, ldb,
                  beta, c, ldc);
#elif REAL_SIZE == 8
      cblas_dgemm(layout, transA, transB,
                  m, n, k,
                  alpha, a, lda,
                  b, ldb,
                  beta, c, ldc);
#endif

#else
#error "Chosen reference CPU-GEMM impl. is not supported"
#endif
    }


    real *findData(real *data, unsigned stride, unsigned blockId) {
      return &data[stride * blockId];
    }

    real *findData(real **data, unsigned stride, unsigned blockId) {
      return &(data[blockId][stride]);
    }
  }
}