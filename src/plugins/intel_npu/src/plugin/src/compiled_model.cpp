// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <chrono>
#include <fstream>
#include <string_view>

#include "async_infer_request.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "metadata.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "transformations/utils/utils.hpp"

namespace {

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};

}

namespace intel_npu {

#define USE_SINGLE_THREADED_RUN_INIT 0

using intel_npu::envVarStrToBool;

std::chrono::steady_clock::time_point begin;
std::chrono::steady_clock::time_point end;

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const std::shared_ptr<IDevice>& device,
                             const std::shared_ptr<IGraph>& graph,
                             const FilteredConfig& config,
                             const std::vector<std::shared_ptr<IGraph>>& initGraphs,
                             const std::shared_ptr<ov::Model>& initModel)
    : ICompiledModel(model, plugin),
      _config(config),
      _logger("CompiledModel", config.get<LOG_LEVEL>()),
      _device(device),
      _graph(graph),
      _initGraphs(initGraphs),
      _initModel(initModel) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::CompiledModel");

    OV_ITT_TASK_CHAIN(COMPILED_MODEL, itt::domains::NPUPlugin, "CompiledModel::CompiledModel", "initialize_properties");
    _properties = std::make_unique<Properties>(PropertiesType::COMPILED_MODEL, _config);
    _properties->registerProperties();

    configure_stream_executors();

    if (_config.get<SEPARATE_WEIGHTS_VERSION>() != 0 && !_initGraphs.empty()) {
        if (_config.get<CREATE_EXECUTOR>() && !_config.get<DEFER_WEIGHTS_LOAD>()) {
            begin = std::chrono::steady_clock::now();
#if USE_SINGLE_THREADED_RUN_INIT
            for (const auto& initGraph : _initGraphs) {
                auto [weightsInputs, initOutputsTensor] =
                    _device->runInit(initGraph, _initModel, get_context(), _config);

                add_weights_inputs(weightsInputs);
                add_init_out_tensor(std::move(initOutputsTensor));
            }
#else
            std::tie(_weightsInputs, _initOutputsTensors) =
                _device->runInitMultiThreaded(_initGraphs, _initModel, get_context(), _config);
#endif
            end = std::chrono::steady_clock::now();
            std::cout << "run_init() call within the \"CompiledModel\" ctor "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
                      << std::endl;
        }
    }

    OV_ITT_TASK_SKIP(COMPILED_MODEL);
}

CompiledModel::~CompiledModel() {
    _logger.debug("~CompiledModel()");
    std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(get_task_executor())->cpu_reset();
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::create_infer_request");

    // sanity check
    if (_device == nullptr) {
        OPENVINO_THROW("No available devices. Failed to create infer request!");
    }

    if (!_config.get<CREATE_EXECUTOR>() || _config.get<DEFER_WEIGHTS_LOAD>()) {
        if (_graph == nullptr) {
            OPENVINO_THROW("Invalid graph handle! Failed to create infer request!");
        }
        _graph->initialize(_config);
    }

    const std::shared_ptr<SyncInferRequest>& syncInferRequest =
        _device->createInferRequest(shared_from_this(), _config);
    syncInferRequest->initialize_states();

    if (_config.get<SEPARATE_WEIGHTS_VERSION>() != 0 && !_initGraphs.empty()) {
        if (!_config.get<CREATE_EXECUTOR>() || _config.get<DEFER_WEIGHTS_LOAD>()) {
            begin = std::chrono::steady_clock::now();
            // TODO: in theory, initialize() could also be pipelined with runInit?
            for (const auto& initGraph : _initGraphs) {
                initGraph->initialize(_config);
            }
            end = std::chrono::steady_clock::now();
            std::cout << "Init graph(s) initialize() "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
                      << std::endl;

            begin = std::chrono::steady_clock::now();
#if USE_SINGLE_THREADED_RUN_INIT
            for (const auto& initGraph : _initGraphs) {
                auto [weightsInputs, initOutputsTensor] =
                    _device->runInit(initGraph, _initModel, get_context(), _config);

                add_weights_inputs(weightsInputs);
                add_init_out_tensor(std::move(initOutputsTensor));
            }
#else
            std::tie(_weightsInputs, _initOutputsTensors) =
                _device->runInitMultiThreaded(_initGraphs, _initModel, get_context(), _config);
#endif
            end = std::chrono::steady_clock::now();
            std::cout << "run_init() call during inference request creation "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
                      << std::endl;
        }

        OPENVINO_ASSERT(_device != nullptr);

        begin = std::chrono::steady_clock::now();
        syncInferRequest->set_weights_inputs(_weightsInputs);
        end = std::chrono::steady_clock::now();
        std::cout << "set_weights_inputs() call "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    } else if (_config.get<SEPARATE_WEIGHTS_VERSION>() != 0 && !_initGraphs.empty()) {
        _logger.warning(
            "SEPARATE_WEIGHTS_VERSION config option was set but no compiled model for the init schedule was found. "
            "run_init() will not run.");
    }

    return std::make_shared<AsyncInferRequest>(syncInferRequest,
                                               get_task_executor(),
                                               _resultExecutor,
                                               get_callback_executor());
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    OPENVINO_THROW_NOT_IMPLEMENTED(
        "The synchronous inference request structure implemented by the NPU plugin does not inherit "
        "the \"ov::ISyncInferRequest\" class");
}

void CompiledModel::export_model(std::ostream& stream) const {
    _logger.debug("CompiledModel::export_model");
    const auto separateWeightsVersion = _config.get<SEPARATE_WEIGHTS_VERSION>();
    if (separateWeightsVersion != 0) {
        if (separateWeightsVersion == 1) {  // special
            _graph->custom_export_split_init(stream, _initGraphs, _initModel);
            return;
        }
        if (_initGraphs.size() != 1) {
            OPENVINO_THROW("Multiple inits are not supported in SEPARATE_WEIGHTS_VERSION: ", separateWeightsVersion);
        }
        _graph->custom_export(stream, _initGraphs.front(), _initModel);
        return;
    }

    size_t blobSizeBeforeVersioning = _graph->export_blob(stream);
    auto meta = Metadata<CURRENT_METADATA_VERSION>(blobSizeBeforeVersioning, CURRENT_OPENVINO_VERSION);
    meta.write(stream);
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    ov::ParameterVector parameters;
    ov::NodeVector results;

    for (const IODescriptor& inputDescriptor : _graph->get_metadata().inputs) {
        if (inputDescriptor.isStateInput || inputDescriptor.isStateOutput || inputDescriptor.isShapeTensor) {
            continue;
        }

        std::shared_ptr<ov::op::v0::Parameter> parameter =
            std::make_shared<ov::op::v0::Parameter>(inputDescriptor.precision, inputDescriptor.shapeFromCompiler);

        parameter->set_friendly_name(inputDescriptor.nodeFriendlyName);
        parameter->output(0).get_tensor().set_names(inputDescriptor.outputTensorNames);
        parameters.push_back(std::move(parameter));
    }

    // The "result" nodes require a parent node in order to satisfy the API conventions. Additionally, a dummy shape for
    // the "Constant" node was required since the specific constructor does not accept "ov::PartialShape" values (a
    // constant can't have dynamic shape). The dummy tensor was also brought in order to register the correct,
    // potentially dynamic, output shape.
    for (const IODescriptor& outputDescriptor : _graph->get_metadata().outputs) {
        if (outputDescriptor.isStateInput || outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor) {
            continue;
        }

        std::shared_ptr<ov::Node> constantDummy = std::make_shared<ov::op::v0::Constant>(
            outputDescriptor.precision,
            outputDescriptor.shapeFromCompiler.to_shape().empty() ? CONSTANT_NODE_DUMMY_SHAPE
                                                                  : outputDescriptor.shapeFromCompiler.to_shape());

        const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy =
            std::make_shared<ov::descriptor::Tensor>(outputDescriptor.precision,
                                                     outputDescriptor.shapeFromCompiler,
                                                     outputDescriptor.outputTensorNames);

        std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v0::Result>(constantDummy);
        result->output(0).set_tensor_ptr(tensorDummy);
        result->set_friendly_name(outputDescriptor.nodeFriendlyName);
        results.push_back(std::move(result));
    }

    _logger.warning("Returning a dummy ov::Model object that contains only the given parameter and result nodes");

    return std::make_shared<ov::Model>(results, parameters);
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    // 1. Set the property via Properties interface
    _properties->set_property(properties);

    // 2. Extra hooks
    if (properties.count(std::string(WORKLOAD_TYPE::key())) != 0) {
        if (_graph != nullptr) {
            const auto workloadType = properties.at(ov::workload_type.name()).as<ov::WorkloadType>();
            _graph->set_workload_type(workloadType);
        }
    }
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    // special cases
    if (name == ov::model_name.name()) {
        OPENVINO_ASSERT(_graph != nullptr, "Missing graph");
        return _graph->get_metadata().name;
    } else {
        // default behaviour
        return _properties->get_property(name);
    }
}

const std::shared_ptr<IGraph>& CompiledModel::get_graph() const {
    return _graph;
}

const FilteredConfig& CompiledModel::get_config() const {
    return _config;
}

void CompiledModel::configure_stream_executors() {
    std::shared_ptr<ov::threading::ITaskExecutor> task_executor;
    if (get_plugin()->get_property(ov::internal::exclusive_async_requests.name(), {}).as<bool>()) {
        task_executor = ov::threading::executor_manager()->get_executor("NPU");
    } else if (get_property(ov::hint::enable_cpu_pinning.name()).as<bool>()) {
        auto executor_config = ov::threading::IStreamsExecutor::Config{
            /* name = */ "Intel NPU plugin executor",
            /* streams = */ get_plugin()->get_property(ov::num_streams.name(), {}).as<ov::streams::Num>(),
            /* threads_per_stream = */ 1,
            /* thread_preferred_core_type = */ ov::hint::SchedulingCoreType::PCORE_ONLY,
            /* cpu_reservation = */ true};
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(executor_config);
    } else {
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(
            ov::threading::IStreamsExecutor::Config{"NPUPlugin executor"});
    }

    set_task_executor(std::move(task_executor));
    const auto executorId = _graph->get_metadata().name + "_NPUResultExecutor";
    _resultExecutor = ov::threading::executor_manager()->get_executor(executorId);
}

void CompiledModel::add_weights_inputs(
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& weightsInputs) const {
    _weightsInputs.merge(weightsInputs);
    OPENVINO_ASSERT(weightsInputs.empty(), "Found weights inputs collision between different inits");
}

void CompiledModel::add_init_out_tensor(ov::SoPtr<ov::ITensor> tensor) const {
    _initOutputsTensors.push_back(std::move(tensor));
}

}  // namespace intel_npu
