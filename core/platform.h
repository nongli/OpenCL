#ifndef NONG_PLATFORM_H
#define NONG_PLATFORM_H

#include <OpenCl/opencl.h>

#include "common.h"

struct DeviceInfo {
  struct Vendor {
    enum Type {
      UNKNOWN,
      INTEL,
      AMD,
      NVIDIA,
    };
  };

  std::string ToString(bool detail = false) const;

  bool is_cpu() const { return type == CL_DEVICE_TYPE_CPU; }
  bool is_gpu() const { return type == CL_DEVICE_TYPE_GPU; }
  const cl_device_id& device() const { return id; }

  cl_device_id id;
  std::string name;
  std::string version;
  std::string vendor_string;
  Vendor::Type vendor;
  cl_device_type type;
  int num_compute_units;

  cl_ulong max_global_mem;
  cl_ulong max_local_mem;

  // Optimal alignmenet to use when sharing between host and device memory.
  cl_uint ptr_alignment;

  // The maximum number of work items in a work group.
  size_t max_work_group_size;
};

class Platform {
 public:
  static bool Init();

  static int num_devices() { return num_devices_; }
  static const DeviceInfo* device(int idx) { return &devices_[idx]; }

  // First gpu device is available, otherwise first cpu.
  static const DeviceInfo* default_device() { return default_device_; }
  // First cpu device.
  static const DeviceInfo* cpu_device() { return cpu_device_; }
  // First gpu device.
  static const DeviceInfo* gpu_device() { return gpu_device_; }

 private:
  Platform();

  // Updates info with device specific optimizations.
  static void EnableDeviceOptimizations(DeviceInfo* info);

  static cl_uint num_devices_;
  static DeviceInfo* devices_;
  static DeviceInfo* default_device_;
  static DeviceInfo* cpu_device_;
  static DeviceInfo* gpu_device_;
};

#endif