#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "core/context.h"
#include "core/platform.h"
#include "core/util.h"

using namespace std;

// OCL - 2.20s
// CPU - 81.7s
#define WIDTH        512
#define HEIGHT       512
#define NSUBSAMPLES  3
#define NAO_SAMPLES  1000

// Include the opencl file and cross compile it.
#include "shim/shim_begin.h"
#include "kernels/ao.cl"
#include "shim/shim_end.h"

Sphere spheres[3];
Plane  plane;

unsigned char Clamp(float f) {
  int i = (int)(f * 255.5);
  if (i < 0) i = 0;
  if (i > 255) i = 255;
  return (unsigned char)i;
}

void Render(unsigned char* img, int w, int h, int nsubsamples) {
  long gid = 0;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      long seed = gid;
      float ao = 0;
      for (int v = 0; v < nsubsamples; v++) {
        for (int u = 0; u < nsubsamples; u++) {
          float px = (x + (u / (float)nsubsamples) - (w / 2.0)) / (w / 2.0);
          float py = -(y + (v / (float)nsubsamples) - (h / 2.0)) / (h / 2.0);
          ao += AmbientOcclusion(px, py, spheres, &plane, &seed, NAO_SAMPLES);
        }
      }

      ao /= (float)(nsubsamples * nsubsamples);
      img[3 * (y * w + x) + 0] = Clamp(ao);
      img[3 * (y * w + x) + 1] = img[3 * (y * w + x)];
      img[3 * (y * w + x) + 2] = img[3 * (y * w + x)];
    }
  }
}

void InitScene() {
  spheres[0].center = to_float4(-2.0, 0, -3.5, 0);
  spheres[0].radius2 = 0.5 * 0.5;
  spheres[1].center = to_float4(-0.5, 0, -3.0, 0);
  spheres[1].radius2 = 0.5 * 0.5;
  spheres[2].center = to_float4(1.0, 0, -2.2, 0);
  spheres[2].radius2 = 0.5 * 0.5;

  float4 plane_p(0, -0.5, 0, 0);
  plane.n = to_float4(0, 1, 0, 0);
  plane.d = -dot(plane_p, plane.n);
}

void SavePPM(const char* fname, int w, int h, unsigned char* img) {
  FILE* fp = fopen(fname, "wb");
  fprintf(fp, "P6\n");
  fprintf(fp, "%d %d\n", w, h);
  fprintf(fp, "255\n");
  fwrite(img, w * h * 3, 1, fp);
  fclose(fp);
}

void RenderOpenCl(unsigned char* img) {
  const bool enable_profiling = true;

  float* ao = (float*)malloc(sizeof(float) * WIDTH * HEIGHT);
  memset(ao, 0, sizeof(float) * WIDTH * HEIGHT);

  Context* ctx = Context::Create(Platform::default_device(), enable_profiling);
  Buffer* result_buffer = ctx->CreateBufferFromMem(
      Buffer::READ_WRITE, ao, sizeof(float) * WIDTH * HEIGHT);
  Buffer* spheres_buffer = ctx->CreateBufferFromMem(
      Buffer::READ_ONLY, spheres, sizeof(spheres));
  Buffer* plane_buffer  = ctx->CreateBufferFromMem(
      Buffer::READ_ONLY, &plane, sizeof(plane));

  Kernel* kernel = ctx->CreateKernel("kernels/ao.cl", "TracePixel");
  kernel->SetArg(0, result_buffer);
  kernel->SetArg(1, spheres_buffer);
  kernel->SetArg(2, plane_buffer);
  kernel->SetArg(3, (cl_int)HEIGHT);
  kernel->SetArg(4, (cl_int)WIDTH);
  kernel->SetArg(5, (cl_int)NSUBSAMPLES);
  kernel->SetArg(6, (cl_int)NAO_SAMPLES);

  ctx->default_queue()->EnqueueKernel(kernel, WIDTH * HEIGHT, -1);
  result_buffer->Read(ctx->default_queue());

  for (int i = 0; i < WIDTH * HEIGHT; ++i) {
    img[i * 3 + 0] = Clamp(ao[i]);
    img[i * 3 + 1] = img[i * 3];
    img[i * 3 + 2] = img[i * 3];
  }

  if (enable_profiling) {
    printf("Profile Events:\n\n%s", ctx->default_queue()->GetEventsProfile().c_str());
  }

  free(ao);
  delete ctx;
}

int main(int argc, char** argv) {
  Platform::Init();

  unsigned char* img = (unsigned char*)malloc(WIDTH * HEIGHT * 3);
  InitScene();

#if 1
  printf("Rendering with opencl.\n");
  RenderOpenCl(img);
  SavePPM("ao_cl.ppm", WIDTH, HEIGHT, img);
#else
  printf("Rendering with cpu.\n");
  Render(img, WIDTH, HEIGHT, NSUBSAMPLES);
  SavePPM("ao_cpu.ppm", WIDTH, HEIGHT, img);
#endif

  free(img);
  printf("Done.\n");
  return 0;
}
