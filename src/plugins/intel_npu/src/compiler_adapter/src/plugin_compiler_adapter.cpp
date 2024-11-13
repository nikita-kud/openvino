// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_compiler_adapter.hpp"

#include <ze_graph_ext.h>

#include <memory>
#include <string>

#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "plugin_graph.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace {
std::shared_ptr<void> loadLibrary(const std::string& libpath) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    return ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#else
    return ov::util::load_shared_object(libpath.c_str());
#endif
}

std::shared_ptr<intel_npu::ICompiler> getCompiler(std::shared_ptr<void> so) {
    static constexpr auto CreateFuncName = "CreateNPUCompiler";
    auto symbol = ov::util::get_symbol(so, CreateFuncName);

    using CreateFuncT = void (*)(std::shared_ptr<intel_npu::ICompiler>&);
    const auto createFunc = reinterpret_cast<CreateFuncT>(symbol);

    std::shared_ptr<intel_npu::ICompiler> compilerPtr;
    createFunc(compilerPtr);
    return compilerPtr;
}

ov::SoPtr<intel_npu::ICompiler> loadCompiler(const std::string& libpath) {
    auto compilerSO = loadLibrary(libpath);
    auto compiler = getCompiler(compilerSO);

    return ov::SoPtr<intel_npu::ICompiler>(compiler, compilerSO);
}
}  // namespace

namespace intel_npu {

PluginCompilerAdapter::PluginCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct)
    : _zeroInitStruct(zeroInitStruct),
      _logger("PluginCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize PluginCompilerAdapter start");

    _logger.info("MLIR compiler will be used.");
    std::string baseName = "npu_mlir_compiler";
    auto libPath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName + OV_BUILD_POSTFIX);
    _compiler = loadCompiler(libPath);

    if (_zeroInitStruct == nullptr) {
        return;
    }

    uint32_t graphExtVersion = _zeroInitStruct->getGraphDdiTable().version();

    _logger.info("PluginCompilerAdapter creating adapter using graphExtVersion");

    switch (graphExtVersion) {
    case ZE_GRAPH_EXT_VERSION_1_3:
        _zeGraphExt = std::make_shared<ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_3>>(_zeroInitStruct);
        break;
    case ZE_GRAPH_EXT_VERSION_1_4:
        _zeGraphExt = std::make_shared<ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_4>>(_zeroInitStruct);
        break;
    case ZE_GRAPH_EXT_VERSION_1_5:
        _zeGraphExt = std::make_shared<ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_5>>(_zeroInitStruct);
        break;
    case ZE_GRAPH_EXT_VERSION_1_6:
        _zeGraphExt = std::make_shared<ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_6>>(_zeroInitStruct);
        break;
    case ZE_GRAPH_EXT_VERSION_1_7:
        _zeGraphExt = std::make_shared<ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_7>>(_zeroInitStruct);
        break;
    case ZE_GRAPH_EXT_VERSION_1_8:
        _zeGraphExt = std::make_shared<ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_8>>(_zeroInitStruct);
        break;
    default:
        _zeGraphExt = std::make_shared<ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_2>>(_zeroInitStruct);
        break;
    }

    _logger.info("initialize PluginCompilerAdapter complete, using graphExtVersion: %d.%d",
                 ZE_MAJOR_VERSION(graphExtVersion),
                 ZE_MINOR_VERSION(graphExtVersion));
}

std::shared_ptr<IGraph> PluginCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                       const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "compile");

    _logger.debug("compile start");
    auto networkDesc = _compiler->compile(model, config);
    _logger.debug("compile end");

    ze_graph_handle_t graphHandle = nullptr;

    if (_zeGraphExt) {
        // Depending on the config, we may get an error when trying to get the graph handle from the compiled network
        try {
            graphHandle = _zeGraphExt->getGraphHandle(networkDesc.compiledNetwork);
        } catch (...) {
            _logger.info("Failed to obtain the level zero graph handle. Inference requests for this model are not "
                         "allowed. Only exports are available");
        }
    }

    return std::make_shared<PluginGraph>(_zeGraphExt,
                                         _compiler,
                                         _zeroInitStruct,
                                         graphHandle,
                                         std::move(networkDesc.metadata),
                                         std::move(networkDesc.compiledNetwork),
                                         config);
}
std::vector<std::shared_ptr<IGraph>> PluginCompilerAdapter::compileWS(const std::shared_ptr<ov::Model>& model,
                                                                      const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "compileWS");

    _logger.debug("compile start");
    const std::vector<std::shared_ptr<NetworkDescription>> initMainNetworkDescriptions =
        _compiler->compileWS(model, config);
    _logger.debug("compile end");

    auto initNetworkDescription = initMainNetworkDescriptions[0];
    auto mainNetworkDescription = initMainNetworkDescriptions[1];

    ze_graph_handle_t initGraphHandle = nullptr;
    ze_graph_handle_t mainGraphHandle = nullptr;

    if (_zeGraphExt) {
        // Depending on the config, we may get an error when trying to get the graph handle from the compiled network
        try {
            initGraphHandle = _zeGraphExt->getGraphHandle(initNetworkDescription->compiledNetwork);
            mainGraphHandle = _zeGraphExt->getGraphHandle(mainNetworkDescription->compiledNetwork);
        } catch (...) {
            _logger.info("Failed to obtain the level zero graph handle. Inference requests for this model are not "
                         "allowed. Only exports are available");
        }
    }

    auto initPluginGraph = std::make_shared<PluginGraph>(_zeGraphExt,
                                                         _compiler,
                                                         _zeroInitStruct,
                                                         initGraphHandle,
                                                         std::move(initNetworkDescription->metadata),
                                                         std::move(initNetworkDescription->compiledNetwork),
                                                         config);
    auto mainPluginGraph = std::make_shared<PluginGraph>(_zeGraphExt,
                                                         _compiler,
                                                         _zeroInitStruct,
                                                         mainGraphHandle,
                                                         std::move(mainNetworkDescription->metadata),
                                                         std::move(mainNetworkDescription->compiledNetwork),
                                                         config);

    return {initPluginGraph, mainPluginGraph};
}

std::shared_ptr<IGraph> PluginCompilerAdapter::parse(std::vector<uint8_t> network, const Config& config) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "parse");

    _logger.debug("parse start");
    auto networkMeta = _compiler->parse(network, config);
    _logger.debug("parse end");

    ze_graph_handle_t graphHandle = nullptr;

    if (_zeGraphExt) {
        graphHandle = _zeGraphExt->getGraphHandle(network);
    }

    return std::make_shared<PluginGraph>(_zeGraphExt,
                                         _compiler,
                                         _zeroInitStruct,
                                         graphHandle,
                                         std::move(networkMeta),
                                         std::move(network),
                                         config);
}

ov::SupportedOpsMap PluginCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const Config& config) const {
    OV_ITT_TASK_CHAIN(QUERY_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "query");

    return _compiler->query(model, config);
}

}  // namespace intel_npu
