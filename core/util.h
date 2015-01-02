#ifndef NONG_UTIL_H
#define NONG_UTIL_H

#include "common.h"

std::string PrintBytes(long bytes);
long timestamp_ms();

class ScopedTimeMeasure {
 public:
  ScopedTimeMeasure(const char* label) : label_(label), start_(timestamp_ms()) {}
  ~ScopedTimeMeasure() {
    long end = timestamp_ms();
    long delta = end - start_;
    printf("%s: %ldms\n", label_, delta);
  }

 private:
  const char* label_;
  long start_;
};

#endif
