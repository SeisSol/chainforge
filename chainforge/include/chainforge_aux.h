#ifndef CHAINFORGE_AUX_H
#define CHAINFORGE_AUX_H

#include <string>

#ifdef HIP_UNDERHOOD
#include <hip/hip_runtime.h>
#endif

#define CHECK_ERR cf::checkErr(__FILE__,__LINE__)
namespace cf {
  void checkErr(const std::string &file, int line);
  void synchDevice();
}

#endif  // CHAINFORGE_AUX_H
