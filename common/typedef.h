#ifndef CHAINFORGE_BENCHMARK_TYPEDEF_H
#define CHAINFORGE_BENCHMARK_TYPEDEF_H

#include <vector>

#if REAL_SIZE == 8
typedef double real;
#elif REAL_SIZE == 4
typedef float real;
#else
#  error REAL_SIZE not supported.
#endif

#endif //CHAINFORGE_BENCHMARK_TYPEDEF_H
