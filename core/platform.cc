#include "platform.h"
#include "util.h"

#include <boost/algorithm/string.hpp>

using namespace boost;
using namespace std;

// Configs for intel gpus (see Intel Zero Copy)
const cl_uint INTEL_ZER_COPY_PTR_ALIGNMENT = 4096;

cl_uint Platform::num_devices_;
DeviceInfo* Platform::devices_;
DeviceInfo* Platform::default_device_;
DeviceInfo* Platform::cpu_device_;
DeviceInfo* Platform::gpu_device_;

string DeviceInfo::ToString(bool detail) const {
  stringstream ss;
  ss << "Device " << name
      << " (" << (is_cpu() ? "CPU" : (is_gpu() ? "GPU" : "UNKNOWN")) << ")";
  if (detail) {
     ss << "  Vendor: " << vendor_string << endl
        << "  NumComputeUnits: " << num_compute_units << endl
        << "  MaxWorkGroupSize: " << max_work_group_size << endl
        << "  MaxLocalMem: " << PrintBytes(max_local_mem) << endl
        << "  MaxGlobalMem: " << PrintBytes(max_global_mem) << endl
        << "  PtrAlignement: " << ptr_alignment << endl;
  }
  return ss.str();
}

DeviceInfo::Vendor::Type ParseVendor(const string& vendor_string) {
  if (iequals(vendor_string, "intel")) {
    return DeviceInfo::Vendor::INTEL;
  }
  return DeviceInfo::Vendor::UNKNOWN;
}

bool GetDeviceInfo(DeviceInfo* info, cl_device_id id) {
  char buf[128];

  info->id = id,
  clGetDeviceInfo(id, CL_DEVICE_NAME, sizeof(buf), buf, NULL);
  info->name = buf;
  clGetDeviceInfo(id, CL_DEVICE_VERSION, sizeof(buf), buf, NULL);
  info->version = buf;
  clGetDeviceInfo(id, CL_DEVICE_VENDOR, sizeof(buf), buf, NULL);
  info->vendor_string = buf;
  info->vendor = ParseVendor(info->vendor_string);

  clGetDeviceInfo(
      id, CL_DEVICE_TYPE, sizeof(cl_device_type), &info->type, NULL);
  clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
      &info->max_work_group_size, 0);

  cl_uint max_dims;
  clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t),
      &max_dims, 0);
  size_t max_per_dim[max_dims];
  clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_per_dim),
      &max_per_dim, 0);
  info->max_work_group_size =
    std::min(info->max_work_group_size, max_per_dim[0]);

  clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
      &info->num_compute_units, 0);
  clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),
      &info->max_local_mem, 0);
  clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
      &info->max_global_mem, 0);
  clGetDeviceInfo(id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint),
      &info->ptr_alignment, 0);
  return true;
}

void Platform::EnableDeviceOptimizations(DeviceInfo* info) {
  if (info->vendor == DeviceInfo::Vendor::INTEL && info->is_gpu()) {
    // Intel integrated GPUs can do zero-copy with higher memory alignment.
    // Refer to Intel Zero Copy Tutorial
    info->ptr_alignment = std::max(info->ptr_alignment, INTEL_ZER_COPY_PTR_ALIGNMENT);
  }
}

bool Platform::Init() {
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices_);
  cl_device_id* ids = (cl_device_id*)calloc(sizeof(cl_device_id), num_devices_);
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, num_devices_, ids, NULL);
  devices_ = new DeviceInfo[num_devices_];

  for (int i = 0; i < num_devices_; ++i) {
    if (!GetDeviceInfo(&devices_[i], ids[i])) return false;
    EnableDeviceOptimizations(&devices_[i]);

    if (default_device_ == NULL ||
        (default_device_->is_cpu() && devices_[i].is_gpu())) {
      default_device_ = &devices_[i];
    }
    if (cpu_device_ == NULL && devices_[i].is_cpu()) cpu_device_ = &devices_[i];
    if (gpu_device_ == NULL && devices_[i].is_gpu()) gpu_device_ = &devices_[i];
  }

  free(ids);
  return true;
}
