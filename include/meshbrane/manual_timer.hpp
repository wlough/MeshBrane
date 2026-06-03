#pragma once

/**
 * @file manual_timer.hpp
 * @brief A timer.
 */

#include <chrono> // std::chrono::high_resolution_clock and std::chrono::duration

class manual_timer {
  std::chrono::high_resolution_clock::time_point t0;
  double timestamp{0.0};

public:
  void start() { t0 = std::chrono::high_resolution_clock::now(); }
  void stop() {
    // timestamp = std::chrono::duration<double>(
    //                 std::chrono::high_resolution_clock::now() - t0)
    //                 .count() *
    //             1000.0;
    timestamp = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0)
                    .count();
  }
  const double &get() { return timestamp; }
};
