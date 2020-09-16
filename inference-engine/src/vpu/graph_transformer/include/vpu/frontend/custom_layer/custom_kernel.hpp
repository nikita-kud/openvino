// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pugixml.hpp>
#include <ie_common.h>

#include <vpu/utils/enums.hpp>
#include <vpu/utils/small_vector.hpp>

namespace vpu {

namespace ie = InferenceEngine;

VPU_DECLARE_ENUM(CustomParamType,
    Input,
    Output,
    Data,
    LocalData,
    InputBuffer,
    OutputBuffer,
    Int,
    Float)

VPU_DECLARE_ENUM(CustomDataFormat,
                 BYXF = 0,  // NHWC used in most software layers
                 BFYX = 1,  // NCHW used if HW module is enabled
                 YXF = 2,   // HWC used in most software layers
                 FYX = 3,   // CHW used if HW module is enabled
                 BF = 4,    // NC layout
                 Any = 5,   // doesn't really matter
                 None = 6)

VPU_DECLARE_ENUM(CustomDimSource, Input, Output)

class CustomCppKernel;
class CustomClKernel;

class CustomKernelVisitor {
public:
    virtual void visitCpp(const CustomCppKernel& kernel) = 0;
    virtual void visitCL(const CustomClKernel& kernel) = 0;
};

class CustomKernel {
public:
    using SPtr = std::shared_ptr<CustomKernel>;

    struct KernelParam final {
        CustomParamType type = CustomParamType::Input;
        CustomDataFormat format = CustomDataFormat::Any;
        std::string argName;
        int portIndex = -1;
        std::string irSource;
        std::string bufferSizeRule;
        CustomDimSource dimSource;
        int dimIdx = -1;
    };

protected:
    std::string _kernelBinary;
    SmallVector<KernelParam> _kernelParams;
    SmallVector<std::string> _parameters;

    CustomDimSource _wgDimSource = CustomDimSource::Input;
    int _wgDimIdx = -1;

    int _maxShaves = 0;
    int _inputDataCount = 0;

public:
    const std::string& kernelBinary() const { return _kernelBinary; }
    SmallVector<KernelParam> bindings() const { return _kernelParams; }
    SmallVector<std::string> parameters() const { return _parameters; }

    CustomDimSource dimSource() const { return _wgDimSource; }
    int dimSourceIndex() const { return _wgDimIdx; }

    int maxShaves() const { return _maxShaves; }
    int inputDataCount() const { return _inputDataCount; }

    virtual void accept(CustomKernelVisitor& validator) const = 0;

protected:
    std::string loadKernelBinary(const pugi::xml_node& node, std::string configDir);
    void processParametersNode(const pugi::xml_node& node);
    std::pair<CustomDimSource, int> parseDimSource(const std::string& dims);
    CustomDataFormat formatFromString(const std::string& str);
    SmallVector<std::string> parseSizeRule(const std::string& size);
};

class CustomCppKernel final : public CustomKernel {
public:
    explicit CustomCppKernel(const pugi::xml_node& node, std::string configDir);

    void accept(CustomKernelVisitor& validator) const override;

protected:
    void processWorkSizesNode(const pugi::xml_node& node);
};

class CustomClKernel final : public CustomKernel {
private:
    SmallVector<std::string> _globalGridSizeRules;
    SmallVector<std::string> _localGridSizeRules;
    int _kernelId = 0;

public:
    explicit CustomClKernel(const pugi::xml_node& node, std::string configDir);

    void accept(CustomKernelVisitor& validator) const override;

    SmallVector<std::string> globalGridSizeRules() const { return _globalGridSizeRules; }
    SmallVector<std::string> localGridSizeRules() const { return _localGridSizeRules; }
    int kernelId() const { return _kernelId; }

private:
    void processWorkSizesNode(const pugi::xml_node& node);
};

} // namespace vpu
