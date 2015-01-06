#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "core/context.h"
#include "core/platform.h"
#include "core/util.h"

using namespace std;

void DumpDevices() {
  if (Platform::num_devices() == 0) {
    printf("No devices.\n");
    return;
  }

  for (int i = 0; i < Platform::num_devices(); ++i) {
    printf("%s\n", Platform::device(i)->ToString(true).c_str());
  }
  printf("\n");

  printf("Default Device: %s\n", Platform::default_device()->ToString().c_str());
  printf("CPU Device: %s\n", Platform::cpu_device()->ToString().c_str());
  printf("GPU Device: %s\n", Platform::gpu_device()->ToString().c_str());
}

int main(int argc, char** argv) {
  ScopedTimeMeasure m("Init");
  Platform::Init();
  DumpDevices();
  printf("Done.\n");
  return 0;
}
