// src/system_utils.cpp

#include "meshbrane/system_utils.hpp"

namespace meshbrane {

std::string shell_quote(const std::string &s) {
#ifdef _WIN32
  std::string quoted = "\"";
  for (char c : s) {
    if (c == '"') {
      quoted += "\\\"";
    } else {
      quoted += c;
    }
  }
  quoted += "\"";
  return quoted;
#else
  std::string quoted = "'";
  for (char c : s) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
#endif
}

} // namespace meshbrane
