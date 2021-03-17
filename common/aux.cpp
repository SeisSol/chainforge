#include "aux.h"
#include <algorithm>
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <iostream>


namespace cf {
  namespace aux {
    long long computeNumFlops(int m, int n, int k, real alpha, real beta) {
      long long flops = (k + (k - 1)) * m * n;

      if (alpha != 1.0) {
        flops += (m * n);
      }

      if (beta != 0.0) {
        flops += (m * n);
      }

      return flops;
    }

    std::vector<real*> shuffleMatrices(real* matrices, int size, int numElements) {
      std::vector<real*> ptrs(numElements, nullptr);
      for (int index = 0; index < numElements; ++index) {
        ptrs.push_back(&matrices[index * size]);
      }
      std::random_shuffle(ptrs.begin(), ptrs.end());
      return ptrs;
    }

    real getRandomNumber() {
      return static_cast<real>(std::rand()) / RAND_MAX;
    }

    void initMatrix(real *matrix, int size, size_t numElements) {
      for (int element = 0; element < numElements; ++element) {
        for (int index = 0; index < size; ++index) {
          matrix[index + size * element] = getRandomNumber();
        }
      }
    }

    bool compare(real *host, const real *device, unsigned size, size_t numElements, real eps) {
      bool isEqual = true;
      size_t counter = 0;

      for (int element = 0; element < numElements; ++element) {
        for (int index = 0; index < size; ++index) {
          real difference = std::abs(host[index + element * size] - device[index + element * size]);
          if (difference > eps) {
            
            std::cout << "at index " << index << ") " << "(bad) "
                      << difference << " = "
                      << host[index + element * size] << " - " << device[index + element * size]
                      << std::endl;
            
            isEqual = false;
            ++counter;
          }
        }
      }

      if (!isEqual) {
        std::cout << "num conflicts: " << counter << std::endl;
      }

      return isEqual;
    }

  } // namespace aux
} // namespace cf