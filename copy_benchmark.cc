#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "core/context.h"
#include "core/platform.h"
#include "core/util.h"

using namespace std;

const size_t TO_COPY = 20L * 1024L * 1024L * 1024L;

#define CPU_CPU 0
#define CPU_GPU 1

template <int mode>
void Copy(size_t num_bytes, size_t batch_size, char* dummy) {
  if (batch_size > num_bytes) batch_size = num_bytes;

  char* src = (char*)malloc(num_bytes);
  char* dst = (char*)malloc(batch_size);
  double total_bytes = 0;

  Context* gpu_ctx = Context::Create(Platform::gpu_device());
  Buffer* gpu_buffer = gpu_ctx->CreateBufferFromMem(Buffer::READ_ONLY, dst, batch_size);

  long start = timestamp_ms();
  while (total_bytes < TO_COPY) {
    size_t bytes_copied = 0;

    while (bytes_copied != num_bytes) {
      size_t to_copy = std::min(batch_size, num_bytes - bytes_copied);
      if (mode == CPU_CPU) {
        memcpy(dst, src + bytes_copied, to_copy);
      } else if (mode == CPU_GPU) {
        gpu_buffer->CopyFrom(gpu_ctx->default_queue(), src + bytes_copied, to_copy);
        gpu_ctx->default_queue()->Flush();
      }
      bytes_copied += to_copy;
    }
    *dummy += dst[0];
    total_bytes += bytes_copied;
  }
  double elapsed = timestamp_ms() - start;

  free(src);
  free(dst);

  double mbs = total_bytes / (1024 * 1024 * 1024);
  mbs /= (elapsed / 1000.);

  if (mode == CPU_CPU) {
    cout << "CPU->CPU";
  } else if (mode == CPU_GPU) {
    cout << "CPU->GPU";
  } else {
    cout << "GPU->CPU";
  }

  cout << " (" << PrintBytes(num_bytes) << " src, "
       << PrintBytes(batch_size) << " batch): " << mbs << " GB/s" << endl;

  delete gpu_ctx;
}

int main(int argc, char** argv) {
  Platform::Init();

  char dummy;

  /*
  Copy<CPU_CPU>(16 * 1024L * 1024L, 64 * 1024L, &dummy);
  Copy<CPU_CPU>(16 * 1024L * 1024L, 512 * 1024L, &dummy);
  Copy<CPU_CPU>(16 * 1024L * 1024L, 8 * 1024L * 1024L, &dummy);
  Copy<CPU_CPU>(256 * 1024L * 1024L, 64 * 1024L, &dummy);
  Copy<CPU_CPU>(256 * 1024L * 1024L, 512 * 1024L, &dummy);
  Copy<CPU_CPU>(256 * 1024L * 1024L, 8 * 1024L * 1024L, &dummy);
  Copy<CPU_CPU>(1024L * 1024L * 1024L, 64 * 1024L, &dummy);
  Copy<CPU_CPU>(1024L * 1024L * 1024L, 512 * 1024L, &dummy);
  Copy<CPU_CPU>(1024L * 1024L * 1024L, 8 * 1024L * 1024L, &dummy);
  */

  Copy<CPU_CPU>(1024 * 1024L * 1024L, 1024 * 1024L, &dummy);
  Copy<CPU_GPU>(1024 * 1024L * 1024L, 1024 * 1024L, &dummy);
  Copy<CPU_GPU>(1024 * 1024L * 1024L, 16 * 1024 * 1024L, &dummy);
  Copy<CPU_GPU>(1024 * 1024L * 1024L, 64 * 1024 * 1024L, &dummy);

  printf("Done.\n");
  return dummy;
}
