#pragma once

/**
 * @file simple_generator.hpp
 * @brief Generator functions using C++20 coroutines

  See
 [this](https://www.sparkslabs.com/blog/posts/coroutines-0-modern-cpp-part2-coroutines.html)
 and
 [this](https://github.com/sparkslabs/blog-extras/blob/main/by-date/2023/Coroutines/0/cpp20simple_generator.hpp)

 */

#include <coroutine>
#include <exception>
#include <unordered_set> // std::unordered_set
#include <vector>

namespace meshbrane {
namespace utils {
/**
 * @brief Template class for a coroutine that yields values of type T.
 * @tparam T The type of values generated.
 */

template <typename T> class SimpleGenerator {
public:
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

private:
  handle_type mCoro;

public:
  explicit SimpleGenerator(handle_type h) : mCoro(h) {}

  SimpleGenerator(SimpleGenerator &&other_sg) noexcept : mCoro(other_sg.mCoro) {
    other_sg.mCoro = nullptr;
  }
  SimpleGenerator &operator=(SimpleGenerator &&other) noexcept {
    if (this != other) {
      mCoro = other.mCoro;
      other.mCoro = nullptr;
      return *this;
    }
  }
  SimpleGenerator(const SimpleGenerator &) = delete;
  SimpleGenerator &operator=(const SimpleGenerator &) = delete;
  ~SimpleGenerator() {
    if (mCoro) {
      mCoro.destroy();
    }
  }

  // Implementation of the external API called by the user to actually use the
  // generator
  void start() { try_next(); }
  bool running() { return not mCoro.done(); }
  void try_next() {
    mCoro.resume();
    if (mCoro.promise().m_latest_exception) {
      std::rethrow_exception(mCoro.promise().m_latest_exception);
    }
  }
  T take() { return std::move(mCoro.promise().m_current_value); }

  // Implementation of the internal API called when co_yield/etc are triggered
  // inside the coroutine
  class promise_type {
    T m_current_value;
    std::exception_ptr m_latest_exception;
    friend SimpleGenerator;

  public:
    auto get_return_object() {
      return SimpleGenerator{handle_type::from_promise(*this)};
    }
    auto yield_value(T some_value) {
      m_current_value = some_value; // Capture the yielded value
      return std::suspend_always{};
    }
    auto unhandled_exception() {
      m_latest_exception = std::current_exception();
    }
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto return_void() { return std::suspend_never{}; }
  };

private:
  // Implementation of the iterator protocol
  class iterator {
    SimpleGenerator<T> &owner;
    bool done;
    void iter_next() {
      owner.try_next();
      done = not owner.running();
    }

  public:
    bool operator!=(const iterator &r) const { return done != r.done; }
    auto operator*() const { return owner.take(); }
    iterator &operator++() {
      iter_next();
      return *this;
    }
    iterator(SimpleGenerator<T> &o, bool d) : owner(o), done(d) {
      if (not done)
        iter_next();
    }
  };

public:
  // Public access to the internal iterator protocol

  iterator begin() { return iterator{*this, false}; }
  iterator end() { return iterator{*this, true}; }

public:
  // Convert generator to a unordered_set
  std::unordered_set<T> to_unordered_set() {
    std::unordered_set<T> result;
    for (auto &value : *this) {
      result.insert(value);
    }
    return result;
  }

  // Convert generator to a vector
  std::vector<T> to_vector() {
    std::vector<T> result;
    for (auto &value : *this) {
      result.push_back(value);
    }
    return result;
  }
};

} // namespace utils

} // namespace meshbrane

/**
 * @example simple_generator.hpp
 *
 * Generate the Fibonacci sequence:
 *
 * @code
 * SimpleGenerator<int> fibs(int max) {
 *   int a{1}, b{1}, n{0};
 *   for (int i = 0; i < max; i++) {
 *     co_yield a;
 *     n = a + b;
 *     a = b;
 *     b = n;
 *   }
 * }
 * @endcode
 *
 *
 * @code
 * int main() {
 *     // Create a generator for Fibonacci numbers
 *     auto generator = fibs(10);
 *
 *     // Use the generator with a range-based for loop
 *     for (int value : generator) {
 *         std::cout << value << " ";
 *     }
 *     std::cout << std::endl;
 *
 *     return 0;
 * }
 * @endcode
 *
 * Generate a sequence of vertices in a clockwise order around a vertex:
 * @code
 * #include "meshbrane/simple_generator.hpp"
 * #include <meshbrane/half_edge_mesh.hpp>
 *
 * using Vert = meshbrane::HalfEdgeVertexBase;
 * SimpleGenerator<Vert> *CCW_vertex_cycle(Vert &v) {
 *  auto h = v.h_out_;
 * auto h_start = h;
 * do {...} while (h != h_start);
 * @endcode
 */
