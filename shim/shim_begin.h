#ifndef NONG_SHIM_BEGIN_H
#define NONG_SHIM_BEGIN_H

// This file implements the opencl language extensions.
// This should be included before including any .cl files from
// c++ src files.
// TODO: not fully implemented.
struct float4 {
  float x;
  float y;
  float z;
  float w;

  float4 operator+(const float4& o) const {
    return float4(x + o.x, y + o.y, z + o.z, w + o.w);
  }
  float4 operator-(const float4& o) const {
    return float4(x - o.x, y - o.y, z - o.z, w - o.w);
  }

  float4 operator*(float f) const {
    return float4(x*f, y*f, z*f, w*f);
  }
  float4 operator/(float f) const {
    return float4(x/f, y/f, z/f, w/f);
  }

  float4& operator=(float v) {
    x = y = z = w = v;
    return *this;
  }

  float4() { }
  float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
};

inline float dot(const float4& v1, const float4& v2) {
  return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w;
}
inline float4 cross(const float4& v1, const float4& v2) {
  float x = v1.y * v2.z - v1.z * v2.y;
  float y = v1.z * v2.x - v1.x * v2.z;
  float z = v1.x * v2.y - v1.y * v2.x;
  // TODO: w
  return float4(x, y, z, 0);
}
inline float4 normalize(const float4& v) {
  float length = sqrt(dot(v, v));
  if (fabs(length) > 1.0e-17) return v / length;
  printf("So confused\n");
  return v;
}
inline float sincos(float v, float* cos_result) {
  *cos_result = cos(v);
  return sin(v);
}

size_t get_global_id(int) { return 0; }

// Include shared code between opencl and host code.
#define constant const
#define global
#define kernel

#endif
