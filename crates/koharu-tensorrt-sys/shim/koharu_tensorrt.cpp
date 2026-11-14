#include "koharu_tensorrt.h"

#include <memory>
#include <string>

#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace {

thread_local std::string last_error;
ktrt_entry_points entry;

void fail(std::string message) {
    last_error = std::move(message);
}

// Forwards TensorRT's messages to Rust and keeps the latest error on the
// thread that raised it, so a failed call reports TensorRT's reason.
class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const *message) noexcept override {
        if (severity <= Severity::kERROR) {
            last_error = message;
        }
        if (entry.log) {
            entry.log(static_cast<int32_t>(severity), message);
        }
    }
};

Logger logger;

bool initialized() {
    if (entry.create_infer_runtime == nullptr) {
        fail("TensorRT-RTX is not initialized");
        return false;
    }
    return true;
}

nvinfer1::Dims dims_from(int64_t const *values, int32_t rank) {
    nvinfer1::Dims dims{};
    dims.nbDims = rank;
    for (int32_t i = 0; i < rank; ++i) {
        dims.d[i] = values[i];
    }
    return dims;
}

}  // namespace

struct ktrt_blob {
    std::unique_ptr<nvinfer1::IHostMemory> memory;
};

struct ktrt_runtime {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
};

struct ktrt_engine {
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
};

// Destroyed in reverse order: the context, then the configuration that holds
// the runtime cache, then the cache.
struct ktrt_context {
    std::unique_ptr<nvinfer1::IRuntimeCache> cache;
    std::unique_ptr<nvinfer1::IRuntimeConfig> config;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
};

extern "C" {

int32_t ktrt_init(ktrt_entry_points const *entry_points) {
    if (entry_points == nullptr || entry_points->create_infer_runtime == nullptr ||
        entry_points->create_infer_builder == nullptr || entry_points->create_onnx_parser == nullptr ||
        entry_points->get_infer_lib_version == nullptr) {
        fail("missing TensorRT-RTX entry point");
        return 1;
    }
    int32_t const version = entry_points->get_infer_lib_version();
    int32_t const expected = NV_TENSORRT_MAJOR * 10000 + NV_TENSORRT_MINOR * 100;
    if (version / 100 != expected / 100) {
        fail("TensorRT-RTX " + std::to_string(version) + " does not match the shim's headers " +
             std::to_string(expected));
        return 1;
    }
    entry = *entry_points;
    return 0;
}

char const *ktrt_last_error(void) {
    return last_error.c_str();
}

int32_t ktrt_library_version(void) {
    return entry.get_infer_lib_version ? entry.get_infer_lib_version() : 0;
}

void const *ktrt_blob_data(ktrt_blob const *blob) {
    return blob->memory->data();
}

size_t ktrt_blob_size(ktrt_blob const *blob) {
    return blob->memory->size();
}

void ktrt_blob_free(ktrt_blob *blob) {
    delete blob;
}

ktrt_blob *ktrt_build_onnx(char const *path, ktrt_shape_range const *ranges, int32_t range_count) {
    if (!initialized()) {
        return nullptr;
    }
    std::unique_ptr<nvinfer1::IBuilder> builder{
        static_cast<nvinfer1::IBuilder *>(entry.create_infer_builder(&logger, NV_TENSORRT_VERSION))};
    if (!builder) {
        fail("failed to create a TensorRT-RTX builder");
        return nullptr;
    }
    // TensorRT-RTX networks are always strongly typed: precision comes from
    // the ONNX graph's own types.
    std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0)};
    if (!network) {
        fail("failed to create a TensorRT-RTX network");
        return nullptr;
    }
    std::unique_ptr<nvonnxparser::IParser> parser{static_cast<nvonnxparser::IParser *>(
        entry.create_onnx_parser(network.get(), &logger, NV_ONNX_PARSER_VERSION))};
    if (!parser) {
        fail("failed to create a TensorRT-RTX ONNX parser");
        return nullptr;
    }
    if (!parser->parseFromFile(path, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::string message = "failed to parse ";
        message += path;
        for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
            message += "\n";
            message += parser->getError(i)->desc();
        }
        fail(message);
        return nullptr;
    }
    std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config) {
        fail("failed to create a TensorRT-RTX builder configuration");
        return nullptr;
    }
    if (range_count > 0) {
        // Owned by the builder.
        nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
        for (int32_t i = 0; i < range_count; ++i) {
            ktrt_shape_range const &range = ranges[i];
            if (range.rank < 0 || range.rank > nvinfer1::Dims::MAX_DIMS ||
                !profile->setDimensions(range.name, nvinfer1::OptProfileSelector::kMIN, dims_from(range.min, range.rank)) ||
                !profile->setDimensions(range.name, nvinfer1::OptProfileSelector::kOPT, dims_from(range.opt, range.rank)) ||
                !profile->setDimensions(range.name, nvinfer1::OptProfileSelector::kMAX, dims_from(range.max, range.rank))) {
                std::string message = "invalid shape range for ";
                message += range.name;
                fail(message);
                return nullptr;
            }
        }
        if (config->addOptimizationProfile(profile) < 0) {
            fail("TensorRT-RTX rejected the optimization profile");
            return nullptr;
        }
    }
    std::unique_ptr<nvinfer1::IHostMemory> memory{builder->buildSerializedNetwork(*network, *config)};
    if (!memory) {
        // The logger has recorded TensorRT's reason.
        return nullptr;
    }
    return new ktrt_blob{std::move(memory)};
}

ktrt_runtime *ktrt_runtime_create(void) {
    if (!initialized()) {
        return nullptr;
    }
    std::unique_ptr<nvinfer1::IRuntime> runtime{
        static_cast<nvinfer1::IRuntime *>(entry.create_infer_runtime(&logger, NV_TENSORRT_VERSION))};
    if (!runtime) {
        fail("failed to create a TensorRT-RTX runtime");
        return nullptr;
    }
    return new ktrt_runtime{std::move(runtime)};
}

void ktrt_runtime_free(ktrt_runtime *runtime) {
    delete runtime;
}

ktrt_engine *ktrt_engine_deserialize(ktrt_runtime *runtime, void const *data, size_t size) {
    std::unique_ptr<nvinfer1::ICudaEngine> engine{runtime->runtime->deserializeCudaEngine(data, size)};
    if (!engine) {
        return nullptr;
    }
    return new ktrt_engine{std::move(engine)};
}

void ktrt_engine_free(ktrt_engine *engine) {
    delete engine;
}

int32_t ktrt_engine_io_count(ktrt_engine const *engine) {
    return engine->engine->getNbIOTensors();
}

char const *ktrt_engine_io_name(ktrt_engine const *engine, int32_t index) {
    return engine->engine->getIOTensorName(index);
}

int32_t ktrt_engine_io_is_input(ktrt_engine const *engine, char const *name) {
    switch (engine->engine->getTensorIOMode(name)) {
    case nvinfer1::TensorIOMode::kINPUT:
        return 1;
    case nvinfer1::TensorIOMode::kOUTPUT:
        return 0;
    default:
        return -1;
    }
}

int32_t ktrt_engine_io_dtype(ktrt_engine const *engine, char const *name) {
    if (engine->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kNONE) {
        return -1;
    }
    return static_cast<int32_t>(engine->engine->getTensorDataType(name));
}

int32_t ktrt_engine_io_shape(ktrt_engine const *engine, char const *name, int64_t *dims, int32_t capacity) {
    if (engine->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kNONE) {
        return -1;
    }
    nvinfer1::Dims const shape = engine->engine->getTensorShape(name);
    for (int32_t i = 0; i < shape.nbDims && i < capacity; ++i) {
        dims[i] = shape.d[i];
    }
    return shape.nbDims;
}

int64_t ktrt_engine_device_memory(ktrt_engine const *engine) {
    return engine->engine->getDeviceMemorySizeV2();
}

ktrt_context *ktrt_context_create(ktrt_engine *engine, void const *cache, size_t cache_size, int32_t cuda_graph) {
    auto context = std::make_unique<ktrt_context>();
    context->config.reset(engine->engine->createRuntimeConfig());
    if (!context->config) {
        fail("failed to create a TensorRT-RTX runtime configuration");
        return nullptr;
    }
    context->cache.reset(context->config->createRuntimeCache());
    if (!context->cache) {
        fail("failed to create a TensorRT-RTX runtime cache");
        return nullptr;
    }
    // A cache from another driver or GPU is rejected here; the kernels are
    // then compiled again and the caller replaces the file.
    if (cache_size > 0 && !context->cache->deserialize(cache, cache_size)) {
        context->cache.reset(context->config->createRuntimeCache());
    }
    if (!context->config->setRuntimeCache(*context->cache)) {
        fail("failed to attach the TensorRT-RTX runtime cache");
        return nullptr;
    }
    if (cuda_graph != 0 &&
        !context->config->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE)) {
        fail("failed to enable CUDA graphs");
        return nullptr;
    }
    context->context.reset(engine->engine->createExecutionContext(context->config.get()));
    if (!context->context) {
        return nullptr;
    }
    return context.release();
}

void ktrt_context_free(ktrt_context *context) {
    if (context != nullptr) {
        context->context.reset();
        context->config.reset();
        context->cache.reset();
    }
    delete context;
}

ktrt_blob *ktrt_context_runtime_cache(ktrt_context const *context) {
    std::unique_ptr<nvinfer1::IHostMemory> memory{context->cache->serialize()};
    if (!memory) {
        fail("failed to serialize the TensorRT-RTX runtime cache");
        return nullptr;
    }
    return new ktrt_blob{std::move(memory)};
}

int32_t ktrt_context_set_input_shape(ktrt_context *context, char const *name, int64_t const *dims, int32_t rank) {
    if (rank < 0 || rank > nvinfer1::Dims::MAX_DIMS) {
        fail("tensor rank out of range");
        return 1;
    }
    return context->context->setInputShape(name, dims_from(dims, rank)) ? 0 : 1;
}

int32_t ktrt_context_io_shape(ktrt_context const *context, char const *name, int64_t *dims, int32_t capacity) {
    if (context->context->getEngine().getTensorIOMode(name) == nvinfer1::TensorIOMode::kNONE) {
        return -1;
    }
    nvinfer1::Dims const shape = context->context->getTensorShape(name);
    for (int32_t i = 0; i < shape.nbDims && i < capacity; ++i) {
        dims[i] = shape.d[i];
    }
    return shape.nbDims;
}

int32_t ktrt_context_set_tensor_address(ktrt_context *context, char const *name, void *address) {
    return context->context->setTensorAddress(name, address) ? 0 : 1;
}

int32_t ktrt_context_enqueue(ktrt_context *context, void *stream) {
    return context->context->enqueueV3(static_cast<cudaStream_t>(stream)) ? 0 : 1;
}

}  // extern "C"
