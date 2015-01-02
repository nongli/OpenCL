#ifndef NONG_CONTEXT_H
#define NONG_CONTEXT_H

#include <OpenCl/opencl.h>
#include "common.h"

#include "platform.h"

class CommandQueue;
class Context;
class Program;
class Kernel;

const char* Error(cl_int err);

class Buffer {
 public:
  enum AccessType {
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE,
  };

  ~Buffer();

  void* Read(CommandQueue* queue);

  static cl_mem_flags to_cl_flags(AccessType t);

  bool can_read() const { return access_ == READ_ONLY || access_ == READ_WRITE; }
  bool can_write() const { return access_ == WRITE_ONLY || access_ == READ_WRITE; }
  size_t size() const { return size_; }

 private:
  friend class Context;
  friend class Kernel;

  Buffer();

  const cl_mem& cl_buffer() { return cl_buffer_; }

  cl_mem cl_buffer_;
  void* host_ptr_;
  size_t size_;
  AccessType access_;
};

class Kernel {
 public:
  ~Kernel();
  bool SetArg(int index, Buffer* buffer);
  bool SetArg(int index, cl_uint v);
  bool SetLocalArg(int index, size_t v);

  const size_t max_work_group_size() const { return max_work_group_size_; }

  std::string ToString(bool detail = false) const;

 private:
  friend class CommandQueue;
  friend class Context;

  Kernel() {}

  std::string fn_name_;
  cl_kernel kernel_;

  // The maximum number of work items in a work group when running this kernel.
  size_t max_work_group_size_;
};

class Program {
 public:
  ~Program() {
    if (program_ != NULL) clReleaseProgram(program_);
  }

 private:
  friend class Context;
  Program(cl_program p) : program_(p) { }
  cl_program program() const { return program_; }
  cl_program program_;
};

class CommandQueue {
 public:
  ~CommandQueue() {
    if (queue_ != NULL) clReleaseCommandQueue(queue_);
  }

  bool EnqueueKernel(Kernel* kernel, size_t global_size, size_t local_size) {
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel->kernel_,
        1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err < 0) {
      fprintf(stderr, "Could not queue kernel: %s\n", Error(err));
      return false;
    }
    return true;
  }

  bool EnqueueKernel(Kernel* kernel, size_t global_size) {
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel->kernel_,
        1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err < 0) {
      fprintf(stderr, "Could not queue kernel: %s\n", Error(err));
      return false;
    }
    return true;
  }

  bool Flush() {
    cl_int err = clFlush(queue_);
    if (err < 0) {
      fprintf(stderr, "Could not flush queu: %s\n", Error(err));
      return false;
    }
    return true;
  }

 private:
  friend class Buffer;
  friend class Context;

  cl_command_queue queue() { return queue_; }

  CommandQueue(cl_command_queue queue) : queue_(queue) {}
  cl_command_queue queue_;
};

class Context {
 public:
  // Creates the context object. The context is the root of all the other created
  // objects. This is the *only* object that needs to be deleted. All objects
  // created off of this have lifetime equal to the context.
  static Context* Create(const DeviceInfo* device);
  ~Context();

  const DeviceInfo* device() const { return device_; }

  // Returns the default CommandQueue.
  CommandQueue* default_queue() { return command_queues_[0]; }

  // Creates additional command queues.
  CommandQueue* CreateCommandQueue();

  // Loads a kernel from src_file with fn_name.
  Kernel* CreateKernel(const char* src_file, const char* fn_name);
  Kernel* CreateKernel(Program* program, const char* fn_name);

  // Creates a program file from a file or in memory .cl code.
  Program* CreateProgramFromFile(const char* path);
  Program* CreateProgramFromSrc(const char* source, size_t size);

  // Creates buffers shared between the host and opencl
  Buffer* CreateBufferFromMem(const Buffer::AccessType& access,
      void* buffer, size_t size);

  // Returns the error code from the last call.
  cl_int error() const { return err_; }

 private:
  Context(const DeviceInfo* device);

  const DeviceInfo* device_; // unowned
  cl_context ctx_;
  cl_int err_; // Last error.

  std::map<std::string, Program*> programs_;
  std::vector<CommandQueue*> command_queues_;
  std::vector<Buffer*> buffers_;
};

#endif
