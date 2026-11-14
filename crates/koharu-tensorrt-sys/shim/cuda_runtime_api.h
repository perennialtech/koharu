/*
 * The TensorRT-RTX headers include the CUDA runtime API only for its stream
 * and event handle types. Declaring those two opaque handles here keeps the
 * shim buildable without a CUDA toolkit; they match the toolkit's own
 * definitions, so the ABI is the same.
 */
#ifndef KOHARU_TENSORRT_CUDA_RUNTIME_API_H
#define KOHARU_TENSORRT_CUDA_RUNTIME_API_H

typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;

#endif
