#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "core/context.h"
#include "core/platform.h"
#include "core/util.h"

using namespace std;

#define WIDTH        512
#define HEIGHT       512
#define NSUBSAMPLES  3
#define NAO_SAMPLES  32
#define EPSILON      0.001

struct Vec {
  float x;
  float y;
  float z;
  float w;
};

struct Isect {
  float     t;
  Vec        p;
  Vec        n;
  Vec const* basis;
};

struct Sphere {
  Vec    center;
  float radius2;
};

struct Plane {
  Vec    p;
  Vec    n;
  Vec    basis[3];
};

struct Ray {
  Vec    org;
  Vec    dir;
};

Sphere spheres[3];
Plane  plane;

static void VAssign(Vec* v, float x, float y, float z) {
  v->x = x;
  v->y = y;
  v->z = z;
}

static inline float VDot(const Vec& v0, const Vec& v1) {
  return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

static void VCross(Vec* c, const Vec& v0, const Vec& v1) {
  c->x = v0.y * v1.z - v0.z * v1.y;
  c->y = v0.z * v1.x - v0.x * v1.z;
  c->z = v0.x * v1.y - v0.y * v1.x;
}

static void VNormalize(Vec* c) {
  float length = sqrt(VDot((*c), (*c)));
  if (fabs(length) > 1.0e-17) {
    c->x /= length;
    c->y /= length;
    c->z /= length;
  }
}

template<bool occlusion_only>
static bool SphereIntersection(Isect* isect, const Ray* ray, const Sphere* sphere) {
  Vec rs;
  rs.x = ray->org.x - sphere->center.x;
  rs.y = ray->org.y - sphere->center.y;
  rs.z = ray->org.z - sphere->center.z;

  float B = VDot(rs, ray->dir);
  float C = VDot(rs, rs) - sphere->radius2;
  float D = B * B - C;

  if (D <= 0) return false;

  float t = -B - sqrt(D);
  if (t < EPSILON) return false;
  if (t > isect->t) return false;

  if (occlusion_only) return true;

  isect->t = t;
  isect->n.x = isect->p.x - sphere->center.x;
  isect->n.y = isect->p.y - sphere->center.y;
  isect->n.z = isect->p.z - sphere->center.z;
  isect->basis = NULL;
  return true;
}

template<bool occlusion_only>
static bool PlaneIntersection(Isect* isect, const Ray* ray, const Plane* plane) {
  float v = VDot(ray->dir, plane->n);
  if (fabs(v) < 1.0e-17) return false;

  float d = -VDot(plane->p, plane->n);
  float t = -(VDot(ray->org, plane->n) + d) / v;

  if ((t > EPSILON) && (occlusion_only || t < isect->t)) {
    if (occlusion_only) return true;
    isect->t = t;
    isect->n = plane->n;
    isect->basis = plane->basis;
    return true;
  }
  return false;
}

void OrthoBasis(Vec* basis, const Vec& n) {
  basis[2] = n;
  basis[1].x = 0.0; basis[1].y = 0.0; basis[1].z = 0.0;

  if ((n.x < 0.6) && (n.x > -0.6)) {
    basis[1].x = 1.0;
  } else if ((n.y < 0.6) && (n.y > -0.6)) {
    basis[1].y = 1.0;
  } else if ((n.z < 0.6) && (n.z > -0.6)) {
    basis[1].z = 1.0;
  } else {
    basis[1].x = 1.0;
  }

  VCross(&basis[0], basis[1], basis[2]);
  VNormalize(&basis[0]);

  VCross(&basis[1], basis[2], basis[0]);
  VNormalize(&basis[1]);
}

float AmbientOcclusion(Isect* isect) {
  int    i, j;
  int    ntheta = NAO_SAMPLES;
  int    nphi   = NAO_SAMPLES;

  Vec basis_mem[3];
  Vec const* basis = isect->basis;
  if (basis == NULL) {
    VNormalize(&isect->n);
    OrthoBasis(basis_mem, isect->n);
    basis = basis_mem;
  }

  float occlusion = 0.0;

  for (j = 0; j < ntheta; j++) {
    for (i = 0; i < nphi; i++) {
      float theta = sqrt(drand48());
      float phi   = 2.0 * M_PI * drand48();

      float x = cos(phi) * theta;
      float y = sin(phi) * theta;
      float z = sqrt(1.0 - theta * theta);

      // local -> global
      float rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
      float ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
      float rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

      Ray ray;
      ray.org = isect->p;
      VAssign(&ray.dir, rx, ry, rz);

      Isect occIsect;
      occIsect.t   = 1.0e+17;

      if (PlaneIntersection<true>(&occIsect, &ray, &plane)
          ||SphereIntersection<true>(&occIsect, &ray, &spheres[0])
          || SphereIntersection<true>(&occIsect, &ray, &spheres[1])
          || SphereIntersection<true>(&occIsect, &ray, &spheres[2])) {
        occlusion += 1.0;
      }
    }
  }

  return (ntheta * nphi - occlusion) / (float)(ntheta * nphi);
}

unsigned char Clamp(float f) {
  int i = (int)(f * 255.5);
  if (i < 0) i = 0;
  if (i > 255) i = 255;
  return (unsigned char)i;
}

void Render(unsigned char* img, int w, int h, int nsubsamples) {
  int x, y;
  int u, v;

  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      float ao = 0;
      for (v = 0; v < nsubsamples; v++) {
        for (u = 0; u < nsubsamples; u++) {
          float px = (x + (u / (float)nsubsamples) - (w / 2.0)) / (w / 2.0);
          float py = -(y + (v / (float)nsubsamples) - (h / 2.0)) / (h / 2.0);

          Ray ray;
          VAssign(&ray.org, 0, 0, 0);
          VAssign(&ray.dir, px, py, -1);
          VNormalize(&ray.dir);

          Isect isect;
          isect.t   = 1.0e+17;
          isect.basis = NULL;

          bool hit = false;
          hit |= SphereIntersection<false>(&isect, &ray, &spheres[0]);
          hit |= SphereIntersection<false>(&isect, &ray, &spheres[1]);
          hit |= SphereIntersection<false>(&isect, &ray, &spheres[2]);
          hit |= PlaneIntersection<false>(&isect, &ray, &plane);
          if (!hit) continue;

          isect.p.x = ray.org.x + ray.dir.x * isect.t;
          isect.p.y = ray.org.y + ray.dir.y * isect.t;
          isect.p.z = ray.org.z + ray.dir.z * isect.t;
          ao += AmbientOcclusion(&isect);
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
  spheres[0].center.x = -2.0;
  spheres[0].center.y =  0.0;
  spheres[0].center.z = -3.5;
  spheres[0].radius2 = 0.5 * 0.5;

  spheres[1].center.x = -0.5;
  spheres[1].center.y =  0.0;
  spheres[1].center.z = -3.0;
  spheres[1].radius2 = 0.5 * 0.5;

  spheres[2].center.x =  1.0;
  spheres[2].center.y =  0.0;
  spheres[2].center.z = -2.2;
  spheres[2].radius2 = 0.5 * 0.5;

  plane.p.x = 0.0;
  plane.p.y = -0.5;
  plane.p.z = 0.0;
  plane.n.x = 0.0;
  plane.n.y = 1.0;
  plane.n.z = 0.0;
  OrthoBasis(plane.basis, plane.n);
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
  float* image = new float[WIDTH * HEIGHT];
  memset(image, 0, sizeof(float) * WIDTH * HEIGHT);

  Context* ctx = Context::Create(Platform::default_device());
  Buffer* image_buffer = ctx->CreateBufferFromMem(
      Buffer::READ_WRITE, image, sizeof(float) * WIDTH * HEIGHT);
  Buffer* spheres_buffer = ctx->CreateBufferFromMem(
      Buffer::READ_ONLY, spheres, sizeof(spheres));
  Buffer* plane_buffer  = ctx->CreateBufferFromMem(
      Buffer::READ_ONLY, &plane, sizeof(plane));

  Kernel* kernel = ctx->CreateKernel("kernels/ao.cl", "traceOnePixel");
  kernel->SetArg(0, image_buffer);
  kernel->SetArg(1, spheres_buffer);
  kernel->SetArg(2, plane_buffer);
  kernel->SetArg(3, (cl_int)HEIGHT);
  kernel->SetArg(4, (cl_int)WIDTH);
  kernel->SetArg(5, (cl_int)NSUBSAMPLES);
  kernel->SetArg(6, (cl_int)NAO_SAMPLES);

  ctx->default_queue()->EnqueueKernel(kernel, WIDTH * HEIGHT);
  image_buffer->Read(ctx->default_queue());

  for (int i = 0; i < WIDTH * HEIGHT; ++i) {
    img[i * 3 + 0] = Clamp(image[i]);
    img[i * 3 + 1] = img[i * 3];
    img[i * 3 + 2] = img[i * 3];
  }

  delete ctx;
}

int main(int argc, char** argv) {
  Platform::Init();

  unsigned char* img = (unsigned char*)malloc(WIDTH * HEIGHT * 3);
  InitScene();
#if 1
  RenderOpenCl(img);
  SavePPM("ao_cl.ppm", WIDTH, HEIGHT, img);
#else
  Render(img, WIDTH, HEIGHT, NSUBSAMPLES);
  SavePPM("ao_cpu.ppm", WIDTH, HEIGHT, img);
#endif

  printf("Done.\n");
  return 0;
}
