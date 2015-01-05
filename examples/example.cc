#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "core/context.h"
#include "core/platform.h"
#include "core/util.h"

using namespace std;

// This is a very simple example that doesn't handle different input very well
// or run the cpu device.
void ArraySum() {
  const int size = 64;
  const int global_size = size / 8;
  const int local_size = global_size / 2;

  float data[size];
  float sum[local_size];

  for (int i = 0; i < size; ++i) data[i] = i;
  for (int i = 0; i < local_size; ++i) sum[i] = 0;

  Context* ctx = Context::Create(Platform::gpu_device());
  Kernel* kernel = ctx->CreateKernel("kernels/add_numbers.cl", "add_numbers");
  Buffer* input_buffer =
      ctx->CreateBufferFromMem(Buffer::READ_ONLY, data, sizeof(data));
  Buffer* sum_buffer =
      ctx->CreateBufferFromMem(Buffer::READ_WRITE, sum, sizeof(sum));

  kernel->SetArg(0, input_buffer);
  kernel->SetLocalArg(1, sizeof(sum));
  kernel->SetArg(2, sum_buffer);

  ctx->default_queue()->EnqueueKernel(kernel, global_size, local_size);
  sum_buffer->Read(ctx->default_queue());

  float cpu_sum = 0;
  for (int i = 0; i < size; ++i) cpu_sum += data[i];

  float cl_sum = 0;
  for (int i = 0; i < local_size; ++i) cl_sum += sum[i];

  printf("%s\n", kernel->ToString(true).c_str());

  printf("CPU Sum:    %f\n", cpu_sum);
  printf("OpenCl Sum: %f\n", cl_sum);

  delete ctx;
}

void NBody() {
}

template<typename T>
void Map(int num_values, int work_items, int iters,
    const char* program_path, const char* kernel_name) {
  printf("\nRunning: %s\n", kernel_name);
  T* input = new T[num_values];
  T* output = new T[num_values];
  srand(1234);
  for (int i = 0; i < num_values; ++i) {
    input[i] = rand() / (float)RAND_MAX * 10;
  }
  memset(output, 0, sizeof(T) * num_values);

  Context* ctx;
  Kernel* kernel;
  Buffer* input_buffer;
  Buffer* output_buffer;
  {
    ScopedTimeMeasure m("Map setup");
    ctx = Context::Create(Platform::default_device());
    kernel = ctx->CreateKernel(program_path, kernel_name);
    input_buffer = ctx->CreateBufferFromMem(
        Buffer::READ_ONLY, input, sizeof(T) * num_values);
    output_buffer = ctx->CreateBufferFromMem(
        Buffer::WRITE_ONLY, output, sizeof(T) * num_values);

    kernel->SetArg(0, input_buffer);
    kernel->SetArg(1, output_buffer);
  }

  {
    ScopedTimeMeasure m("Map queue");
    for (int i = 0; i < iters; ++i) {
      ctx->default_queue()->EnqueueKernel(kernel, work_items, -1);
    }
  }

  {
    ScopedTimeMeasure m("Map execute");
    output_buffer->Read(ctx->default_queue());
  }

  T total = 0;
  for (int i = 0; i < num_values; ++i) {
    total += output[i];
  }
  cout << "Result: " << total << endl;
}

void BitonicSort() {
  const int input_size = pow(8, 4) * 4;
  const cl_uint ascending = true;
  int* input = new int[input_size];
  int* ref = new int[input_size];

  for (int i = 0; i < input_size; ++i) {
    input[i] = rand() % 999;
  }
  memcpy(ref, input, sizeof(int) * input_size);
  sort(ref, ref + input_size);

  int num_stages = 0;
  for (int i = input_size; i > 2; i >>= 1) ++num_stages;

  Context* ctx;
  Kernel* kernel;
  Buffer* buffer;
  {
    ScopedTimeMeasure m("BitonicSort setup");
    ctx = Context::Create(Platform::default_device());
    kernel = ctx->CreateKernel("kernels/bitonic_sort.cl", "BitonicSort");
    buffer = ctx->CreateBufferFromMem(
        Buffer::READ_WRITE, input, sizeof(int) * input_size);

    kernel->SetArg(0, buffer);
    kernel->SetArg(3, ascending);
  }

  {
    ScopedTimeMeasure m("BitonicSort queue");
    for (int stage = 0; stage < num_stages; ++stage) {
      kernel->SetArg(1, (cl_uint)stage);
      for (int pass_of_stage = stage; pass_of_stage >= 0; --pass_of_stage) {
        kernel->SetArg(2, (cl_uint)pass_of_stage);
        size_t global_size = input_size / (2 * 4);
        if (pass_of_stage == 0) global_size = global_size << 1;
        ctx->default_queue()->EnqueueKernel(kernel, global_size, -1);
      }
    }
  }

  {
    ScopedTimeMeasure m("BitonicSort execute");
    buffer->Read(ctx->default_queue());
  }

  for (int i = 0; i < input_size; ++i) {
    assert(input[i] == ref[i]);
  }
  delete ctx;
}

int main(int argc, char** argv) {
  {
    ScopedTimeMeasure m("Init");
    Platform::Init();
  }

  ArraySum();
//  BitonicSort();
//  Map<float>(1024 * 1024, 1024 * 1024, 10000, "kernels/kernels.cl", "SimpleKernel");
//  Map<float>(1024 * 1024, 1024 * 1024 / 4, 10000, "kernels/kernels.cl", "SimpleKernel4");

  printf("Done.\n");
  return 0;
}
