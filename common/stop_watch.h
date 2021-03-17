/**
 * @class utils::StopWatch
 *
 * @brief Measures time intervals
 * */

#ifndef CHAINFORGE_BENCHMARK_STOP_WATCH_H
#define CHAINFORGE_BENCHMARK_STOP_WATCH_H

#include <chrono>

namespace utils {
  template <class D>
  class StopWatch {
  public:

    /**
     * @brief Sets up a beginning of a new time interval.
     * */
    void start() { m_TimePoint = std::chrono::high_resolution_clock::now(); };

    /**
     * @brief Interrupts counting from the last start() call and accumulate time.
     * */
    void stop() { m_Duration += (std::chrono::high_resolution_clock::now() - m_TimePoint); };

    /**
     * @brief Sets duration time buffer to zero.
     * */
    void reset() { m_Duration = std::chrono::high_resolution_clock::duration::zero(); }

    /**
     * @brief Returns accumulated time duration from the last call to reset().
     * */
    typename D::rep getTime() { return std::chrono::duration_cast<D>(m_Duration).count(); }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_TimePoint;
    std::chrono::high_resolution_clock::duration m_Duration = std::chrono::high_resolution_clock::duration::zero();
  };
}

#endif //CHAINFORGE_BENCHMARK_STOP_WATCH_H
