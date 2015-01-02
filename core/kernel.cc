#include "context.h"

using namespace std;

Kernel::~Kernel() {
  if (kernel_ != NULL) clReleaseKernel(kernel_);
}

string Kernel::ToString(bool detail) const {
  stringstream ss;
  ss << "Kernel '" << fn_name_ << "'" << endl;
  if (detail) {
     ss << "  MaxWorkGroupSize: " << max_work_group_size_ << endl;
  }
  return ss.str();
}

bool Kernel::SetArg(int index, Buffer* buffer) {
  cl_int err = clSetKernelArg(kernel_, index, sizeof(cl_mem), &buffer->cl_buffer());
  if (err < 0) {
    fprintf(stderr, "Could not set kernel argument: %s\n", Error(err));
    return false;
  }
  return true;
}

bool Kernel::SetArg(int index, cl_uint v) {
  cl_int err = clSetKernelArg(kernel_, index, sizeof(v), &v);
  if (err < 0) {
    fprintf(stderr, "Could not set kernel argument: %s\n", Error(err));
    return false;
  }
  return true;
}

bool Kernel::SetLocalArg(int index, size_t v) {
  cl_int err = clSetKernelArg(kernel_, index, v, NULL);
  if (err < 0) {
    fprintf(stderr, "Could not set kernel argument: %s\n", Error(err));
    return false;
  }
  return true;
}
