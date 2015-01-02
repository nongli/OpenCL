#include "context.h"

cl_mem_flags Buffer::to_cl_flags(AccessType t) {
  switch (t) {
    case READ_ONLY: return CL_MEM_READ_ONLY;
    case WRITE_ONLY: return CL_MEM_WRITE_ONLY;
    case READ_WRITE: return CL_MEM_READ_WRITE;
    default:
      assert(false);
      return CL_MEM_READ_WRITE;
  }
}

Buffer::Buffer() : cl_buffer_(NULL), host_ptr_(NULL), size_(0) {
}

Buffer::~Buffer() {
  if (cl_buffer_ != NULL) clReleaseMemObject(cl_buffer_);
}

void* Buffer::Read(CommandQueue* queue) {
  cl_int err = clEnqueueReadBuffer(queue->queue(), cl_buffer_, CL_TRUE, 0,
      size_, host_ptr_, 0, NULL, NULL);
  if (err < 0) {
    fprintf(stderr, "Could not read buffer: %s\n", Error(err));
    return NULL;
  }
  return host_ptr_;
}

bool Buffer::CopyFrom(CommandQueue* queue, const void* src_buffer, size_t buffer_len) {
  cl_int err = clEnqueueWriteBuffer(queue->queue(), cl_buffer_, false,
      0, buffer_len, src_buffer, 0, NULL, NULL);
  if (err < 0) {
    fprintf(stderr, "Could not write buffer: %s\n", Error(err));
    return false;
  }
  return true;
}
