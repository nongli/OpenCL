inline float4 to_float4(float x, float y, float z, float w) {
  float4 result;
  result.x = x;
  result.y = y;
  result.z = z;
  result.w = w;
  return result;
}

#ifndef M_PI
#define M_PI 3.14159265f
#endif
#define EPSILON 0.001f

typedef struct _sphere {
  float4 center;
  float radius2;
  float pad[3];
} Sphere;

typedef struct _plane {
  //float4 p;
  float4 n;
  float d;
  float pad[3];
} Plane;

typedef struct _intersect_pt {
  float t;
  float4 p;
  float4 n;
} Isect;

typedef struct _ray {
  float4 orig; 
  float4 dir;
} Ray;

inline float UniformRand(long* seed) {
  *seed = (*seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
  int v = (int)*seed;
  return fabs(v / (float)INT_MAX);
}

inline int RaySphereOcclude(const Ray* ray, constant Sphere* sphere) {
  float4 rs = ray->orig - sphere->center;
  float B = dot(rs, ray->dir);
  float C = dot(rs, rs) - sphere->radius2;
  float D = B * B - C;
  if (D > 0.0f) {
    float t = -B - sqrt(D);
    return t > EPSILON;
  }
  return 0;
}

inline int RaySphereIntersect(Isect* isect, const Ray* ray, constant Sphere* sphere) {
  float4 rs = ray->orig - sphere->center;
  float B = dot(rs, ray->dir);
  float C = dot(rs, rs) - sphere->radius2;
  float D = B * B - C;
  if (D > 0.0f) {
    float t = -B - sqrt(D);
    if ((t > EPSILON) && (t < isect->t )) {
      isect->t = t;
      isect->p = ray->orig + ray->dir * t;
      isect->n = normalize(isect->p - sphere->center);
      return 1;
    }
  }
  return 0;
}

inline int RayPlaneOcclude(const Ray* ray, const Plane* plane) {
  float v = dot(ray->dir, plane->n);
  if (fabs(v) < 1.0e-17f) return 0;
  float t = -(dot(ray->orig, plane->n) + plane->d) / v;
  return t > EPSILON;
}

inline int RayPlaneIntersect(Isect* isect, const Ray* ray, const Plane* plane) {
  float v = dot(ray->dir, plane->n);
  if (fabs(v) < 1.0e-17f) return 0;
  float t = -(dot(ray->orig, plane->n) + plane->d) / v;
  if ((t > EPSILON) && (t < isect->t)) {
    isect->t = t;
    isect->p = ray->orig + ray->dir * t;
    isect->n = plane->n;
    return 1;
  }
  return 0;
}

inline void OrthoBasis(float4 basis[3], float4 n) {
  basis[1] = 0;
  basis[2] = n;

  if ((n.x < 0.6f) && (n.x > -0.6f)) {
    basis[1].x = 1.0f;
  } else if((n.y < 0.6f) && (n.y > -0.6f)) {
    basis[1].y = 1.0f;
  } else if((n.z < 0.6f) && (n.z > -0.6f)) {
    basis[1].z = 1.0f;
  } else {
    basis[1].x = 1.0f;
  }

  basis[0] = normalize(cross(basis[1], basis[2]));
  basis[1] = normalize(cross(basis[2], basis[0]));
}

inline float AmbientOcclusion(float px, float py,
    constant Sphere* spheres, Plane* plane, long* seed, int nao_samples) {
  Ray ray;
  ray.orig = 0.0f;
  ray.dir = normalize(to_float4(px, py, -1.0f, 0));

  Isect isect;
  isect.t = 1.0e+17f;

  int hit = 0;
  for (int s = 0; s < 3; ++s) {
    hit |= RaySphereIntersect(&isect, &ray, spheres + s);
  }
  hit |= RayPlaneIntersect(&isect, &ray, plane);
  if (!hit) return 0;

  ray.orig = isect.p + isect.n * EPSILON;
  float4 basis[3];
  OrthoBasis(basis, isect.n);

  float visible = 0.0f;
  for (int i = 0; i < nao_samples; ++i) {
    float theta = sqrt(UniformRand(seed));
    float phi = 2.0f * M_PI * UniformRand(seed);

    float x;
    float y = sincos(phi, &x) * theta;
    x *= theta;
    float z = sqrt(1.0f - theta * theta);

    // local->global
    float rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
    float ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
    float rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

    // Already normalized.
    ray.dir = to_float4(rx, ry, rz, 0);

    hit = RayPlaneOcclude(&ray, plane);
    for (int s = 0; s < 3 && hit == 0; ++s) {
      hit |= RaySphereOcclude(&ray, spheres + s);
    }
    visible += !hit;
  }

  return visible / (float)(nao_samples);
}

kernel void traceOnePixel(global float *fimg, 
    constant Sphere* spheres, constant Plane* planes, int h, int w, 
    int nsubsamples, int nao_samples) {
  long gid = get_global_id(0);
  int x = gid % w;
  int y = gid / w;

  Plane plane = planes[0];
  long seed = gid;

  for (int v = 0; v <  nsubsamples; ++v) {
    for (int u = 0; u< nsubsamples; ++u) {
      float px = (x + (u/(float)nsubsamples) - (w/2.0f)) / (w/2.0f);
      float py = -(y + (v/(float)nsubsamples) - (h/2.0f)) / (h/2.0f);
      fimg[gid] += AmbientOcclusion(px, py, spheres, &plane, &seed, nao_samples);
    }
  }

  fimg[gid] /= (float)(nsubsamples * nsubsamples);
}

