#ifndef NONG_UTIL_H
#define NONG_UTIL_H

#include "common.h"

std::string PrintBytes(long bytes);
double timestamp_ms();

class ScopedTimeMeasure {
 public:
  ScopedTimeMeasure(const char* label) : label_(label), start_(timestamp_ms()) {}
  ~ScopedTimeMeasure() {
    double end = timestamp_ms();
    double delta = end - start_;
    printf("%s: %fms\n", label_, delta);
  }

 private:
  const char* label_;
  double start_;
};

#endif
