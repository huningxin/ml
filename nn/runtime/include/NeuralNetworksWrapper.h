/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Provides C++ classes to more easily use the Neural Networks API.

#ifndef ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_WRAPPER_H
#define ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_WRAPPER_H

#include "NeuralNetworks.h"

#include <math.h>
#include <vector>

namespace android {
namespace nn {
namespace wrapper {

enum class Type {
    OEM = ANEURALNETWORKS_OEM,
    FLOAT32 = ANEURALNETWORKS_FLOAT32,
    INT32 = ANEURALNETWORKS_INT32,
    UINT32 = ANEURALNETWORKS_UINT32,
    TENSOR_OEM_BYTE = ANEURALNETWORKS_TENSOR_OEM_BYTE,
    TENSOR_FLOAT32 = ANEURALNETWORKS_TENSOR_FLOAT32,
    TENSOR_INT32 = ANEURALNETWORKS_TENSOR_INT32,
    TENSOR_QUANT8_ASYMM = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
};

enum class ExecutePreference {
    PREFER_LOW_POWER = ANEURALNETWORKS_PREFER_LOW_POWER,
    PREFER_FAST_SINGLE_ANSWER = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER,
    PREFER_SUSTAINED_SPEED = ANEURALNETWORKS_PREFER_SUSTAINED_SPEED
};

enum class Result {
    NO_ERROR = ANEURALNETWORKS_NO_ERROR,
    OUT_OF_MEMORY = ANEURALNETWORKS_OUT_OF_MEMORY,
    INCOMPLETE = ANEURALNETWORKS_INCOMPLETE,
    UNEXPECTED_NULL = ANEURALNETWORKS_UNEXPECTED_NULL,
    BAD_DATA = ANEURALNETWORKS_BAD_DATA,
};

struct OperandType {
    ANeuralNetworksOperandType operandType;
    // uint32_t type;
    std::vector<uint32_t> dimensions;

    OperandType(Type type, const std::vector<uint32_t>& d) : dimensions(d) {
        operandType.type = static_cast<uint32_t>(type);
        operandType.scale = 0.0f;
        operandType.offset = 0;

        operandType.dimensions.count = static_cast<uint32_t>(dimensions.size());
        operandType.dimensions.data = dimensions.data();
    }

    OperandType(Type type, float scale, const std::vector<uint32_t>& d) : OperandType(type, d) {
        operandType.scale = scale;
    }

    OperandType(Type type, float f_min, float f_max, const std::vector<uint32_t>& d)
        : OperandType(type, d) {
        uint8_t q_min = std::numeric_limits<uint8_t>::min();
        uint8_t q_max = std::numeric_limits<uint8_t>::max();
        float range = q_max - q_min;
        float scale = (f_max - f_min) / range;
        int32_t offset =
                    fmin(q_max, fmax(q_min, static_cast<uint8_t>(round(q_min - f_min / scale))));

        operandType.scale = scale;
        operandType.offset = offset;
    }
};

inline Result Initialize() {
    return static_cast<Result>(ANeuralNetworksInitialize());
}

inline void Shutdown() {
    ANeuralNetworksShutdown();
}

class Memory {
public:

    Memory(size_t size, int protect, int fd, size_t offset) {
        mValid = ANeuralNetworksMemory_createFromFd(size, protect, fd, offset, &mMemory) ==
                         ANEURALNETWORKS_NO_ERROR;
    }

    ~Memory() { ANeuralNetworksMemory_free(mMemory); }

    // Disallow copy semantics to ensure the runtime object can only be freed
    // once. Copy semantics could be enabled if some sort of reference counting
    // or deep-copy system for runtime objects is added later.
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    // Move semantics to remove access to the runtime object from the wrapper
    // object that is being moved. This ensures the runtime object will be
    // freed only once.
    Memory(Memory&& other) {
        *this = std::move(other);
    }
    Memory& operator=(Memory&& other) {
        if (this != &other) {
            mMemory = other.mMemory;
            mValid = other.mValid;
            other.mMemory = nullptr;
            other.mValid = false;
        }
        return *this;
    }

    ANeuralNetworksMemory* get() const { return mMemory; }
    bool isValid() const { return mValid; }

private:
    ANeuralNetworksMemory* mMemory = nullptr;
    bool mValid = true;
};

class Model {
public:
    Model() {
        // TODO handle the value returned by this call
        ANeuralNetworksModel_create(&mModel);
    }
    ~Model() { ANeuralNetworksModel_free(mModel); }

    // Disallow copy semantics to ensure the runtime object can only be freed
    // once. Copy semantics could be enabled if some sort of reference counting
    // or deep-copy system for runtime objects is added later.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Move semantics to remove access to the runtime object from the wrapper
    // object that is being moved. This ensures the runtime object will be
    // freed only once.
    Model(Model&& other) {
        *this = std::move(other);
    }
    Model& operator=(Model&& other) {
        if (this != &other) {
            mModel = other.mModel;
            mNextOperandId = other.mNextOperandId;
            mValid = other.mValid;
            other.mModel = nullptr;
            other.mNextOperandId = 0;
            other.mValid = false;
        }
        return *this;
    }

    int finish() { return ANeuralNetworksModel_finish(mModel); }

    uint32_t addOperand(const OperandType* type) {
        if (ANeuralNetworksModel_addOperand(mModel, &(type->operandType)) !=
            ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
        return mNextOperandId++;
    }

    void setOperandValue(uint32_t index, const void* buffer, size_t length) {
        if (ANeuralNetworksModel_setOperandValue(mModel, index, buffer, length) !=
            ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
    }

    void setOperandValueFromMemory(uint32_t index, const Memory* memory, uint32_t offset,
                                   size_t length) {
        if (ANeuralNetworksModel_setOperandValueFromMemory(mModel, index, memory->get(), offset,
                                                           length) != ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
    }

    void addOperation(ANeuralNetworksOperationType type, const std::vector<uint32_t>& inputs,
                      const std::vector<uint32_t>& outputs) {
        ANeuralNetworksIntList in, out;
        Set(&in, inputs);
        Set(&out, outputs);
        if (ANeuralNetworksModel_addOperation(mModel, type, &in, &out) !=
            ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
    }
    void setInputsAndOutputs(const std::vector<uint32_t>& inputs,
                             const std::vector<uint32_t>& outputs) {
        ANeuralNetworksIntList in, out;
        Set(&in, inputs);
        Set(&out, outputs);
        if (ANeuralNetworksModel_setInputsAndOutputs(mModel, &in, &out) !=
            ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
    }
    ANeuralNetworksModel* getHandle() const { return mModel; }
    bool isValid() const { return mValid; }

private:
    /**
     * WARNING list won't be valid once vec is destroyed or modified.
     */
    void Set(ANeuralNetworksIntList* list, const std::vector<uint32_t>& vec) {
        list->count = static_cast<uint32_t>(vec.size());
        list->data = vec.data();
    }

    ANeuralNetworksModel* mModel = nullptr;
    // We keep track of the operand ID as a convenience to the caller.
    uint32_t mNextOperandId = 0;
    bool mValid = true;
};

class Compilation {
public:
    Compilation(const Model* model) {
        int result = ANeuralNetworksCompilation_create(model->getHandle(), &mCompilation);
        if (result != 0) {
            // TODO Handle the error
        }
    }

    ~Compilation() { ANeuralNetworksCompilation_free(mCompilation); }

    Compilation(const Compilation&) = delete;
    Compilation& operator=(const Compilation &) = delete;

    Compilation(Compilation&& other) {
        *this = std::move(other);
    }
    Compilation& operator=(Compilation&& other) {
        if (this != &other) {
            mCompilation = other.mCompilation;
            other.mCompilation = nullptr;
        }
        return *this;
    }

    Result setPreference(ExecutePreference preference) {
        return static_cast<Result>(ANeuralNetworksCompilation_setPreference(
                    mCompilation, static_cast<uint32_t>(preference)));
    }

    // TODO startCompile

    Result compile() {
        Result result = static_cast<Result>(ANeuralNetworksCompilation_start(mCompilation));
        if (result != Result::NO_ERROR) {
            return result;
        }
        // TODO how to manage the lifetime of compilations when multiple waiters
        // is not clear.
        return static_cast<Result>(ANeuralNetworksCompilation_wait(mCompilation));
    }

    ANeuralNetworksCompilation* getHandle() const { return mCompilation; }

private:
    ANeuralNetworksCompilation* mCompilation = nullptr;
};

class Request {
public:
    Request(const Compilation* compilation) {
        int result = ANeuralNetworksRequest_create(compilation->getHandle(), &mRequest);
        if (result != 0) {
            // TODO Handle the error
        }
    }

    ~Request() { ANeuralNetworksRequest_free(mRequest); }

    // Disallow copy semantics to ensure the runtime object can only be freed
    // once. Copy semantics could be enabled if some sort of reference counting
    // or deep-copy system for runtime objects is added later.
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;

    // Move semantics to remove access to the runtime object from the wrapper
    // object that is being moved. This ensures the runtime object will be
    // freed only once.
    Request(Request&& other) {
        *this = std::move(other);
    }
    Request& operator=(Request&& other) {
        if (this != &other) {
            mRequest = other.mRequest;
            other.mRequest = nullptr;
        }
        return *this;
    }

    Result setInput(uint32_t index, const void* buffer, size_t length,
                    const ANeuralNetworksOperandType* type = nullptr) {
        return static_cast<Result>(
                    ANeuralNetworksRequest_setInput(mRequest, index, type, buffer, length));
    }

    Result setInputFromMemory(uint32_t index, const Memory* memory, uint32_t offset,
                              uint32_t length, const ANeuralNetworksOperandType* type = nullptr) {
        return static_cast<Result>(ANeuralNetworksRequest_setInputFromMemory(
                    mRequest, index, type, memory->get(), offset, length));
    }

    Result setOutput(uint32_t index, void* buffer, size_t length,
                     const ANeuralNetworksOperandType* type = nullptr) {
        return static_cast<Result>(
                    ANeuralNetworksRequest_setOutput(mRequest, index, type, buffer, length));
    }

    Result setOutputFromMemory(uint32_t index, const Memory* memory, uint32_t offset,
                               uint32_t length, const ANeuralNetworksOperandType* type = nullptr) {
        return static_cast<Result>(ANeuralNetworksRequest_setOutputFromMemory(
                    mRequest, index, type, memory->get(), offset, length));
    }

    Result startCompute() {
        Result result = static_cast<Result>(ANeuralNetworksRequest_startCompute(mRequest));
        return result;
    }

    Result wait() {
        return static_cast<Result>(ANeuralNetworksRequest_wait(mRequest));
    }

    Result compute() {
        Result result = static_cast<Result>(ANeuralNetworksRequest_startCompute(mRequest));
        if (result != Result::NO_ERROR) {
            return result;
        }
        // TODO how to manage the lifetime of events when multiple waiters is not
        // clear.
        return static_cast<Result>(ANeuralNetworksRequest_wait(mRequest));
    }

private:
    ANeuralNetworksRequest* mRequest = nullptr;
};

}  // namespace wrapper
}  // namespace nn
}  // namespace android

#endif  //  ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_WRAPPER_H
