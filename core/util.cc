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

#include <CoreServices/CoreServices.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <unistd.h>

double timestamp_ms() {
  struct timeval t;
  gettimeofday(&t, 0);
  return t.tv_sec * 1000L + t.tv_usec / 1000.;
}
