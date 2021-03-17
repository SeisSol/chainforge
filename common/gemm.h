#ifndef CHAINFORGE_BENCHMARK_GEMM_H
#define CHAINFORGE_BENCHMARK_GEMM_H

#include "typedef.h"

namespace cf {
  namespace reference {
    enum class LayoutType {
      Trans, NoTrans
    };

    real *findData(real *data, unsigned stride, unsigned blockId);
    real *findData(real **data, unsigned stride, unsigned blockId);

    void gemm(LayoutType typeA,
              LayoutType typeB,
              int m, int n, int k,
              real alpha, real *a, int lda,
              real *b, int ldb,
              real beta, real *c, int ldc);
  }
}

#endif //CHAINFORGE_BENCHMARK_GEMM_H