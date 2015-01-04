#include "platform.h"
#include "util.h"

#include <boost/algorithm/string.hpp>
#include <boost/unordered_set.hpp>

using namespace boost;
using namespace std;

// Configs for intel gpus (see Intel Zero Copy)
const cl_uint INTEL_ZER_COPY_PTR_ALIGNMENT = 4096;

cl_uint Platform::num_devices_;
DeviceInfo* Platform::devices_;
DeviceInfo* Platform::default_device_;
DeviceInfo* Platform::cpu_device_;
DeviceInfo* Platform::gpu_device_;

string VersionToString(DeviceInfo::Version::Type t) {
  switch (t) {
    case DeviceInfo::Version::OPEN_CL_1_0: return "1.0";
    case DeviceInfo::Version::OPEN_CL_1_1: return "1.1";
    case DeviceInfo::Version::OPEN_CL_1_2: return "1.2";
    case DeviceInfo::Version::OPEN_CL_2_0: return "2.0";
  }
}

string DeviceInfo::ToString(bool detail) const {
  stringstream ss;
  ss << "Device " << name
      << " (" << (is_cpu() ? "CPU" : (is_gpu() ? "GPU" : "UNKNOWN")) << ")";
  if (!detail) return ss.str();

  ss << "  Vendor: " << vendor_string << endl
     << "  Version: " << VersionToString(version) << endl
     << "  NumComputeUnits: " << num_compute_units << endl
     << "  MaxWorkGroupSize: " << max_work_group_size << endl
     << "  MaxLocalMem: " << PrintBytes(max_local_mem) << endl
     << "  MaxGlobalMem: " << PrintBytes(max_global_mem) << endl
     << "  PtrAlignement: " << ptr_alignment << endl
     << "  Extensions" << endl
     << "    AtomicsInt32: " << (extensions.atomics_int32 ? "Yes" : "No") << endl
     << "    AtomicsInt64: " << (extensions.atomics_int64 ? "Yes" : "No") << endl
     << "    ByteAddressable: " << (extensions.byte_addressable ? "Yes" : "No") << endl
     << "    Doubles: " << (extensions.double_precision ? "Yes" : "No") << endl;
  return ss.str();
}

static bool set_contains(const unordered_set<string>& s, const string& v) {
  return s.find(v) != s.end();
}

DeviceInfo::Vendor::Type ParseVendor(const string& vendor_string) {
  if (iequals(vendor_string, "intel")) {
    return DeviceInfo::Vendor::INTEL;
  }
  return DeviceInfo::Vendor::UNKNOWN;
}

DeviceInfo::Version::Type ParseVesion(const string& version_str) {
  if (iequals(version_str, "opencl 1.1")) {
    return DeviceInfo::Version::OPEN_CL_1_1;
  } else if (iequals(version_str, "opencl 1.2")) {
    return DeviceInfo::Version::OPEN_CL_1_2;
  } else if (iequals(version_str, "opencl 2.0")) {
    return DeviceInfo::Version::OPEN_CL_2_0;
  }
  return DeviceInfo::Version::OPEN_CL_1_0;
}

void ParseExtensions(const string& str, DeviceInfo* info) {
  memset(&info->extensions, 0, sizeof(info->extensions));
  vector<string> strs;
  split(strs, str, is_any_of(" "));
  unordered_set<string> extensions;
  for (int i = 0; i < strs.size(); ++i) {
    to_lower(strs[i]);
    extensions.insert(strs[i]);
  }

  if (set_contains(extensions, "cl_khr_fp64")) {
    info->extensions.double_precision = true;
  }

  if (set_contains(extensions, "cl_khr_byte_addressable_store")) {
    info->extensions.byte_addressable = true;
  }

  if (set_contains(extensions, "cl_khr_global_int32_base_atomics") &&
      set_contains(extensions, "cl_khr_global_int32_extended_atomics") &&
      set_contains(extensions, "cl_khr_global_int32_base_atomics") &&
      set_contains(extensions, "cl_khr_local_int32_extended_atomics")) {
    info->extensions.atomics_int32 = true;
  }

  if (set_contains(extensions, "cl_khr_int64_base_atomics") &&
      set_contains(extensions, "cl_khr_int64_extended_atomics")) {
    info->extensions.atomics_int64 = true;
  }
}

bool GetDeviceInfo(DeviceInfo* info, cl_device_id id) {
  char buf[1024];

  info->id = id,
  clGetDeviceInfo(id, CL_DEVICE_NAME, sizeof(buf), buf, NULL);
  info->name = buf;

  clGetDeviceInfo(id, CL_DEVICE_VERSION, sizeof(buf), buf, NULL);
  info->version_str = buf;
  info->version_str.erase(info->version_str.find_last_not_of(" ") + 1);
  info->version = ParseVesion(info->version_str);

  clGetDeviceInfo(id, CL_DEVICE_VENDOR, sizeof(buf), buf, NULL);
  info->vendor_string = buf;
  info->vendor = ParseVendor(info->vendor_string);

  clGetDeviceInfo(id, CL_DEVICE_EXTENSIONS, sizeof(buf), buf, NULL);
  ParseExtensions(buf, info);

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
