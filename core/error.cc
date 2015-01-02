#include "context.h"

#define ADD_CASE(ERROR) \
  case CL_##ERROR: return #ERROR
const char* Error(cl_int err) {
  switch (err) {
   ADD_CASE(DEVICE_MAX_WRITE_IMAGE_ARGS);
   ADD_CASE(INVALID_COMMAND_QUEUE);
   ADD_CASE(INVALID_CONTEXT);
   ADD_CASE(INVALID_EVENT_WAIT_LIST);
   ADD_CASE(INVALID_GLOBAL_OFFSET);
   ADD_CASE(INVALID_KERNEL);
   ADD_CASE(INVALID_KERNEL_ARGS);
   ADD_CASE(INVALID_PROGRAM_EXECUTABLE);
   ADD_CASE(INVALID_VALUE);
   ADD_CASE(INVALID_WORK_DIMENSION);
   ADD_CASE(INVALID_WORK_GROUP_SIZE);
   ADD_CASE(MEM_OBJECT_ALLOCATION_FAILURE);
   ADD_CASE(OUT_OF_HOST_MEMORY);
   ADD_CASE(OUT_OF_RESOURCES);
   default: return "Unknown";
  }
}
