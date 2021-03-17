#ifndef CHAINFORGE_AUX_H
#define CHAINFORGE_AUX_H

#include <string>

#define CHECK_ERR cf::checkErr(__FILE__,__LINE__)
namespace cf {
  void checkErr(const std::string &file, int line);
  void synchDevice();
}

#endif  // CHAINFORGE_AUX_H