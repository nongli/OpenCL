#include "context.h"

using namespace std;

Context* Context::Create(const DeviceInfo* device) {
  Context* ctx = new Context(device);
  ctx->ctx_ = clCreateContext(NULL, 1, &device->device(), NULL, NULL, &ctx->err_);
  if (ctx->err_ < 0) {
    delete ctx;
    return NULL;
  }
  if (!ctx->CreateCommandQueue()) {
    delete ctx;
    return NULL;
  }
  return ctx;
}

Context::Context(const DeviceInfo* device) : device_(device), ctx_(NULL), err_(0) {
}

Context::~Context() {
  for (map<string, Program*>::iterator it = programs_.begin(); it != programs_.end(); ++it) {
    delete it->second;
  }
  for (int i = 0; i < kernels_.size(); ++i) {
    delete kernels_[i];
  }
  for (int i = 0; i < command_queues_.size(); ++i) {
    delete command_queues_[i];
  }
  for (int i = 0; i < buffers_.size(); ++i) {
    delete buffers_[i];
  }
  if (ctx_ != NULL) clReleaseContext(ctx_);
}

Kernel* Context::CreateKernel(const char* path, const char* fn_name) {
  Program* program = CreateProgramFromFile(path);
  if (program == NULL) return NULL;
  return CreateKernel(program, fn_name);
}

Program* Context::CreateProgramFromFile(const char* path) {
  if (programs_.find(path) != programs_.end()) return programs_[path];

  FILE* file = fopen(path, "r");
  if (file == NULL) {
    fprintf(stderr, "Could not find program: %s\n", path);
    return NULL;
  }
  fseek(file, 0, SEEK_END);
  int size = ftell(file);
  rewind(file);

  vector<char> buffer;
  buffer.resize(size + 1);
  fread(&buffer[0], sizeof(char), size, file);
  fclose(file);

  Program* program = CreateProgramFromSrc(&buffer[0], size);
  if (program == NULL) return program;
  programs_[path] = program;
  return program;
};

Program* Context::CreateProgramFromSrc(const char* source, size_t size) {
  cl_program program = clCreateProgramWithSource(ctx_, 1, &source, &size, &err_);
  if (err_ < 0) {
    fprintf(stderr, "Could not create program: %s.\n", Error(err_));
    return NULL;
  }
  err_ = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err_ < 0) {
    fprintf(stderr, "Could not build program.");
    return NULL;
  }
  return new Program(program);
}

Kernel* Context::CreateKernel(Program* program, const char* fn_name) {
  cl_int err;
  cl_kernel kern = clCreateKernel(program->program(), fn_name, &err);
  if (err < 0) {
    fprintf(stderr, "Could not create kernel: %s\n", fn_name);
    return NULL;
  }
  size_t size;
  err = clGetKernelWorkGroupInfo(kern, device_->device(), CL_KERNEL_WORK_GROUP_SIZE,
      sizeof(size), &size, 0);
  if (err < 0) {
    fprintf(stderr, "Could not get kernel info: %s\n", Error(err));
    clReleaseKernel(kern);
    return NULL;
  }

  Kernel* kernel = new Kernel();
  kernel->fn_name_ = fn_name;
  kernel->kernel_ = kern;
  kernel->max_work_group_size_ = size;
  kernels_.push_back(kernel);
  return kernel;
}

CommandQueue* Context::CreateCommandQueue() {
  cl_command_queue queue = clCreateCommandQueue(ctx_, device_->device(), 0, &err_);
  if (err_ < 0) {
    fprintf(stderr, "Could not create command queue.");
    return NULL;
  }
  CommandQueue* q = new CommandQueue(queue);
  command_queues_.push_back(q);
  return q;
}

Buffer* Context::CreateBufferFromMem(const Buffer::AccessType& access,
    void* buffer, size_t size) {
  cl_mem_flags flags = CL_MEM_COPY_HOST_PTR | Buffer::to_cl_flags(access);
  cl_int err;
  cl_mem cl_buffer = clCreateBuffer(ctx_, flags, size, buffer, &err);
  if (err < 0) {
    fprintf(stderr, "Could not create buffer.\n");
    return NULL;
  }

  Buffer* buf = new Buffer();
  buf->cl_buffer_ = cl_buffer;
  buf->host_ptr_ = buffer;
  buf->size_ = size;
  buf->access_ = access;
  buffers_.push_back(buf);
  return buf;
}
