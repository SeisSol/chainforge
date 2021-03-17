#include <iostream>
#include <cuda_runtime.h>

namespace cf {
    std::string prevFile = "";
    int prevLine = 0;

    void checkErr(const std::string &file, int line) {
      cudaError_t error = cudaGetLastError();
      if (error != cudaSuccess) {
        std::cout << std::endl << file
                  << ", line " << line
                  << ": " << cudaGetErrorString(error)
                  << " (" << error << ")"
                  << std::endl;

        if (prevLine > 0)
          std::cout << "Previous CUDA call:" << std::endl
                    << prevFile << ", line " << prevLine << std::endl;
        throw;
      }
      prevFile = file;
      prevLine = line;
    }

  void synchDevice() {
    cudaDeviceSynchronize();
    checkErr(__FILE__, __LINE__);
  }
}
