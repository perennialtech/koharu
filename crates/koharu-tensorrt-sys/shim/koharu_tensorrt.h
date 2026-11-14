/*
 * C ABI over the TensorRT-RTX C++ API.
 *
 * The shim links nothing from TensorRT-RTX. `ktrt_init` receives the
 * library's factory functions, resolved by the Rust side from libraries the
 * runtime package already loaded, and everything else is reached through the
 * objects they create.
 *
 * Functions that can fail return NULL or a non-zero status and leave a
 * message for `ktrt_last_error` on the calling thread.
 */
#ifndef KOHARU_TENSORRT_H
#define KOHARU_TENSORRT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ktrt_blob ktrt_blob;
typedef struct ktrt_runtime ktrt_runtime;
typedef struct ktrt_engine ktrt_engine;
typedef struct ktrt_context ktrt_context;

/* Severity follows nvinfer1::ILogger::Severity: 0 internal error, 1 error,
 * 2 warning, 3 info, 4 verbose. */
typedef void (*ktrt_log_fn)(int32_t severity, const char *message);

typedef struct ktrt_entry_points {
    void *(*create_infer_runtime)(void *logger, int32_t version);
    void *(*create_infer_builder)(void *logger, int32_t version);
    void *(*create_onnx_parser)(void *network, void *logger, int32_t version);
    int32_t (*get_infer_lib_version)(void);
    ktrt_log_fn log;
} ktrt_entry_points;

/* Must be called once before anything else. Fails if the loaded library's
 * major or minor version differs from the headers the shim was built with. */
int32_t ktrt_init(const ktrt_entry_points *entry_points);
const char *ktrt_last_error(void);
/* The library version, as major * 10000 + minor * 100 + patch. */
int32_t ktrt_library_version(void);

const void *ktrt_blob_data(const ktrt_blob *blob);
size_t ktrt_blob_size(const ktrt_blob *blob);
void ktrt_blob_free(ktrt_blob *blob);

/* Parses an ONNX file into a strongly typed network and builds a serialized
 * engine. Building needs no GPU; kernels are compiled when a context is
 * created. */
ktrt_blob *ktrt_build_onnx(const char *path);

ktrt_runtime *ktrt_runtime_create(void);
void ktrt_runtime_free(ktrt_runtime *runtime);

/* The runtime must outlive every engine deserialized from it. */
ktrt_engine *ktrt_engine_deserialize(ktrt_runtime *runtime, const void *data, size_t size);
void ktrt_engine_free(ktrt_engine *engine);
int32_t ktrt_engine_io_count(const ktrt_engine *engine);
const char *ktrt_engine_io_name(const ktrt_engine *engine, int32_t index);
/* 1 for an input, 0 for an output, -1 for an unknown name. */
int32_t ktrt_engine_io_is_input(const ktrt_engine *engine, const char *name);
/* nvinfer1::DataType, or -1 for an unknown name. */
int32_t ktrt_engine_io_dtype(const ktrt_engine *engine, const char *name);
/* Writes up to `capacity` extents (-1 for a dynamic one) and returns the
 * rank, or -1 for an unknown name. */
int32_t ktrt_engine_io_shape(const ktrt_engine *engine, const char *name, int64_t *dims, int32_t capacity);
int64_t ktrt_engine_device_memory(const ktrt_engine *engine);

/* Creates an execution context, compiling kernels for the current device.
 * `cache` may hold a serialized runtime cache from an earlier process, which
 * skips the compilation for kernels it already holds. The engine must
 * outlive the context. */
ktrt_context *ktrt_context_create(ktrt_engine *engine, const void *cache, size_t cache_size, int32_t cuda_graph);
void ktrt_context_free(ktrt_context *context);
/* The runtime cache with every kernel compiled so far. */
ktrt_blob *ktrt_context_runtime_cache(const ktrt_context *context);
int32_t ktrt_context_set_input_shape(ktrt_context *context, const char *name, const int64_t *dims, int32_t rank);
/* The shape a tensor has after the inputs' shapes are set: same contract as
 * ktrt_engine_io_shape. */
int32_t ktrt_context_io_shape(const ktrt_context *context, const char *name, int64_t *dims, int32_t capacity);
int32_t ktrt_context_set_tensor_address(ktrt_context *context, const char *name, void *address);
/* Enqueues inference on a cudaStream_t. */
int32_t ktrt_context_enqueue(ktrt_context *context, void *stream);

#ifdef __cplusplus
}
#endif

#endif
