#include <iostream>
#include <hip/hip_runtime.h>

namespace cf {
    std::string prevFile = "";
    int prevLine = 0;

    void checkErr(const std::string &file, int line) {
      hipError_t error = hipGetLastError();
      if (error != hipSuccess) {
        std::cout << std::endl << file
                  << ", line " << line
                  << ": " << hipGetErrorString(error)
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
    hipDeviceSynchronize();
    checkErr(__FILE__, __LINE__);
  }
}
