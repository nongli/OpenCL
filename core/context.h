#ifndef NONG_CONTEXT_H
#define NONG_CONTEXT_H

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
  bool CopyFrom(CommandQueue* queue, const void* src_buffer, size_t buffer_len);
  bool CopyTo(CommandQueue* queue, void* dst_buffer, size_t buffer_len);

  static cl_mem_flags to_cl_flags(AccessType t);

  bool can_read() const { return access_ == READ_ONLY || access_ == READ_WRITE; }
  bool can_write() const { return access_ == WRITE_ONLY || access_ == READ_WRITE; }
  size_t size() const { return size_; }

 private:
  Buffer(const Buffer&);
  Buffer& operator=(const Buffer&);

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
  std::string fn_name() const { return fn_name_; }

 private:
  Kernel(const Kernel&);
  Kernel& operator=(const Kernel&);

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

  struct BuildOptions {
    bool warnings_as_errors;
    bool disable_optimizations;
    bool strict_aliasing;
    bool unsafe_math;

    BuildOptions() {
      memset(this, 0, sizeof(BuildOptions));
      warnings_as_errors = true;
      strict_aliasing = true;
    }

    std::string ToString() const;
  };

 private:
  Program(const Program&);
  Program& operator=(const Program&);

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

  // local_size can be set to -1 if the device should decide.
  bool EnqueueKernel(Kernel* kernel, size_t global_size, int64_t local_size,
      const std::string& event_name = "") {
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel->kernel_,
        1, NULL, &global_size, local_size == -1 ? NULL : (size_t*)&local_size,
        0, NULL, enable_profiling_ ? &event : NULL);
    if (err < 0) {
      fprintf(stderr, "Could not queue kernel: %s\n", Error(err));
      return false;
    }
    EnqueueEvent(event, event_name == "" ? kernel->fn_name() : event_name);
    return true;
  }

  bool Flush() {
    cl_int err = clFinish(queue_);
    if (err < 0) {
      fprintf(stderr, "Could not flush queu: %s\n", Error(err));
      return false;
    }
    return true;
  }

  std::string GetEventsProfile() const;

 private:
  CommandQueue(const CommandQueue&);
  CommandQueue& operator=(const CommandQueue&);

  friend class Buffer;
  friend class Context;

  cl_command_queue queue() { return queue_; }

  CommandQueue(cl_command_queue queue, bool enable_profiling)
    : queue_(queue), enable_profiling_(enable_profiling) {}

  cl_command_queue queue_;
  const bool enable_profiling_;

  // Only used if profiling is enabled.
  struct ProfileEvent {
    std::string name;
    cl_event e;
    ProfileEvent(const std::string& n, cl_event e) : name(n), e(e) {}
    ProfileEvent() {}
  };
  std::vector<ProfileEvent> profiling_events_;

  void EnqueueEvent(cl_event e, const std::string& name);
};

class Context {
 public:
  // Creates the context object. The context is the root of all the other created
  // objects. This is the *only* object that needs to be deleted. All objects
  // created off of this have lifetime equal to the context.
  static Context* Create(const DeviceInfo* device, bool enable_profiling = false);
  ~Context();

  const DeviceInfo* device() const { return device_; }

  // Returns the default CommandQueue.
  CommandQueue* default_queue() { return command_queues_[0]; }

  // Creates additional command queues.
  CommandQueue* CreateCommandQueue();

  // Loads a kernel from src_file with fn_name.
  Kernel* CreateKernel(const char* src_file, const char* fn_name,
      const Program::BuildOptions& = Program::BuildOptions());
  Kernel* CreateKernel(Program* program, const char* fn_name);

  // Creates a program file from a file or in memory .cl code.
  Program* CreateProgramFromFile(const char* path,
      const Program::BuildOptions& = Program::BuildOptions());
  Program* CreateProgramFromSrc(const char* source, size_t size,
    const Program::BuildOptions& = Program::BuildOptions(),
    const char* filename = NULL);

  // Creates buffers shared between the host and opencl
  Buffer* CreateBufferFromMem(const Buffer::AccessType& access,
      void* buffer, size_t size);

  // Returns the error code from the last call.
  cl_int error() const { return err_; }

 private:
  Context(const DeviceInfo* device, bool enable_profiling);
  Context(const Context&);
  Context& operator=(const Context&);

  std::string GetBuildError(cl_program program);

  const DeviceInfo* device_; // unowned
  const bool enable_profiling_;
  cl_context ctx_;
  cl_int err_; // Last error.

  std::map<std::string, Program*> programs_;
  std::vector<Kernel*> kernels_;
  std::vector<CommandQueue*> command_queues_;
  std::vector<Buffer*> buffers_;
};

#endif
