#include "util.h"

#include <sys/time.h>

using namespace std;

string PrintBytes(long bytes) {
  stringstream ss;
  if (bytes < 1024) {
    ss << bytes;
  } else if (bytes < 1024L * 1024L) {
    ss << bytes / (1024.) << "KB";
  } else if (bytes < 1024L * 1024L * 1024L) {
    ss << bytes / (1024. * 1024.) << "MB";
  } else {
    ss << bytes / (1024. * 1024. * 1024.) << "GB";
  }
  return ss.str();
}

string PrintNanos(long value) {
  stringstream ss;
  if (value < 1000) {
    ss << value << "ns";
  } else if (value < 1000L * 1000L) {
    ss << value / (1000.) << "us";
  } else if (value < 1000L * 1000L * 1000L) {
    ss << value / (1000. * 1000.) << "ms";
  } else {
    ss << value / (1000. * 1000. * 1000.) << "s";
  }
  return ss.str();
}

#ifdef __APPLE__
#include <CoreServices/CoreServices.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <unistd.h>
#else
#include <sys/time.h>
#endif

double timestamp_ms() {
  struct timeval t;
  gettimeofday(&t, 0);
  return t.tv_sec * 1000L + t.tv_usec / 1000.;
}
