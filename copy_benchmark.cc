#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "core/context.h"
#include "core/platform.h"
#include "core/util.h"

using namespace std;

int main(int argc, char** argv) {
  Platform::Init();

  printf("Done.\n");
  return 0;
}
