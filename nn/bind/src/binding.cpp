#include <emscripten/bind.h>

#include "Operations.h"
#include "OperationsUtils.h"

using namespace emscripten;
using namespace android::nn;

namespace binding_utils {
  int32_t getShapeType(const Shape& shape) {
    return (int32_t)shape.type;
  }

  void setShapeType(Shape& shape, int32_t type) {
    shape.type = (OperandType)type;
  }

  val getShapeDimensions(const Shape& shape) {
    emscripten::val js_dims = emscripten::val::array();
    for (int i = 0; i < shape.dimensions.size(); i++) {
      js_dims.call<void>("push", shape.dimensions[i]);
    }
    return js_dims;
  }

  void setShapeDimensions(Shape& shape, val js_dims) {
    shape.dimensions = vecFromJSArray<uint32_t>(js_dims);
  }

  // Operation helper wrappers.
  bool addMulPrepareWrapper(const Shape& in1, const Shape& in2, Shape& out1) {
    return addMulPrepare(in1, in2, &out1);
  }

  bool floorPrepareWrapper(const Shape& input, Shape& output) {
    return floorPrepare(input, &output);
  }

  bool dequantizePrepareWrapper(const Shape& input, Shape& output) {
    return dequantizePrepare(input, &output);
  }

  bool depthwiseConvPrepareWrapper(const Shape& input, const Shape& filter, const Shape& bias,
                                   int32_t padding_left, int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
                                   int32_t stride_width, int32_t stride_height, Shape& output) {
    return depthwiseConvPrepare(input, filter, bias, padding_left, padding_right, padding_top, padding_bottom, stride_width, stride_height, &output);
  }

  bool convPrepareWrapper(const Shape& input, const Shape& filter, const Shape& bias,
                          int32_t padding_left, int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height, Shape& output) {
    return convPrepare(input, filter, bias, padding_left, padding_right, padding_top, padding_bottom, stride_width, stride_height, &output);
  }

  // Operation wrappers.
  bool addFloat32Wrapper(const intptr_t in1, const Shape& shape1,
                         const intptr_t in2, const Shape& shape2,
                         int32_t activation, intptr_t out, const Shape& shapeOut) {
    return addFloat32((const float*)in1, shape1, (const float*)in2, shape2, activation, (float*)out, shapeOut);
  }

  bool addQuant8Wrapper(const intptr_t in1, const Shape& shape1,
                        const intptr_t in2, const Shape& shape2,
                        int32_t activation, intptr_t out, const Shape& shapeOut) {
    return addQuant8((const uint8_t*)in1, shape1, (const uint8_t*)in2, shape2, activation, (uint8_t*)out, shapeOut);
  }

  bool mulFloat32Wrapper(const intptr_t in1, const Shape& shape1,
                         const intptr_t in2, const Shape& shape2,
                         int32_t activation, intptr_t out, const Shape& shapeOut) {
    return mulFloat32((const float*)in1, shape1, (const float*)in2, shape2, activation, (float*)out, shapeOut);
  }

  bool mulQuant8Wrapper(const intptr_t in1, const Shape& shape1,
                        const intptr_t in2, const Shape& shape2,
                        int32_t activation, intptr_t out, const Shape& shapeOut) {
    return mulQuant8((const uint8_t*)in1, shape1, (const uint8_t*)in2, shape2, activation, (uint8_t*)out, shapeOut);
  }

  bool floorFloat32Wrapper(const intptr_t inputData, intptr_t outputData, const Shape& shape) {
    return floorFloat32((const float*)inputData, (float*)outputData, shape);
  }

  bool dequantizeQuant8ToFloat32Wrapper(const intptr_t inputData, intptr_t outputData, const Shape& shape) {
    return dequantizeQuant8ToFloat32((const uint8_t*)inputData, (float*)outputData, shape);
  }

  bool depthwiseConvFloat32Wrapper(const intptr_t inputData, const Shape& inputShape,
                                   const intptr_t filterData, const Shape& filterShape,
                                   const intptr_t biasData, const Shape& biasShape,
                                   int32_t padding_left, int32_t padding_right,
                                   int32_t padding_top, int32_t padding_bottom,
                                   int32_t stride_width, int32_t stride_height,
                                   int32_t depth_multiplier, int32_t activation,
                                   intptr_t outputData, const Shape& outputShape) {
    return depthwiseConvFloat32((const float*)inputData, inputShape,
                                (const float*)filterData, filterShape,
                                (const float*)biasData, biasShape,
                                padding_left, padding_right, padding_top, padding_bottom,
                                stride_width, stride_height,
                                depth_multiplier, activation,
                                (float*)outputData, outputShape);
  }

  bool depthwiseConvQuant8Wrapper(const intptr_t inputData, const Shape& inputShape,
                                  const intptr_t filterData, const Shape& filterShape,
                                  const intptr_t biasData, const Shape& biasShape,
                                  int32_t padding_left, int32_t padding_right,
                                  int32_t padding_top, int32_t padding_bottom,
                                  int32_t stride_width, int32_t stride_height,
                                  int32_t depth_multiplier, int32_t activation,
                                  intptr_t outputData, const Shape& outputShape) {
    return depthwiseConvQuant8((const uint8_t*)inputData, inputShape,
                               (const uint8_t*)filterData, filterShape,
                               (const int32_t*)biasData, biasShape,
                               padding_left, padding_right,
                               padding_top, padding_bottom,
                               stride_width, stride_height,
                               depth_multiplier, activation,
                               (uint8_t*)outputData, outputShape);
  }

  bool convFloat32Wrapper(const intptr_t inputData, const Shape& inputShape,
                          const intptr_t filterData, const Shape& filterShape,
                          const intptr_t biasData, const Shape& biasShape,
                          int32_t padding_left, int32_t padding_right,
                          int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height,
                          int32_t activation,
                          intptr_t outputData, const Shape& outputShape) {
    return convFloat32((const float*)inputData, inputShape,
                       (const float*)filterData, filterShape,
                       (const float*)biasData, biasShape,
                       padding_left, padding_right, padding_top, padding_bottom,
                       stride_width,  stride_height, activation,
                       (float*)outputData, outputShape);
  }
  bool convQuant8Wrapper(const intptr_t inputData, const Shape& inputShape,
                         const intptr_t filterData, const Shape& filterShape,
                         const intptr_t biasData, const Shape& biasShape,
                         int32_t padding_left, int32_t padding_right,
                         int32_t padding_top, int32_t padding_bottom,
                         int32_t stride_width, int32_t stride_height,
                         int32_t activation,
                         intptr_t outputData, const Shape& outputShape) {
    return convQuant8((const uint8_t*)inputData, inputShape,
                      (const uint8_t*)filterData, filterShape,
                      (const int32_t*)biasData, biasShape,
                      padding_left, padding_right,
                      padding_top, padding_bottom,
                      stride_width, stride_height, activation,
                      (uint8_t*)outputData, outputShape);
  }
}

EMSCRIPTEN_BINDINGS(nn)
{
  constant("NONE", (int32_t)FusedActivationFunc::NONE);
  constant("RELU", (int32_t)FusedActivationFunc::RELU);
  constant("RELU1", (int32_t)FusedActivationFunc::RELU1);
  constant("RELU6", (int32_t)FusedActivationFunc::RELU6);

  constant("FLOAT32", (int32_t)OperandType::FLOAT32);
  constant("INT32", (int32_t)OperandType::INT32);
  constant("UINT32", (int32_t)OperandType::UINT32);
  constant("TENSOR_FLOAT32", (int32_t)OperandType::TENSOR_FLOAT32);
  constant("TENSOR_INT32", (int32_t)OperandType::TENSOR_INT32);
  constant("TENSOR_QUANT8_ASYMM", (int32_t)OperandType::TENSOR_QUANT8_ASYMM);

  class_<Shape>("Shape")
    .constructor<>()
    .property("type", &binding_utils::getShapeType, &binding_utils::setShapeType)
    .property("dimensions", &binding_utils::getShapeDimensions, &binding_utils::setShapeDimensions)
    .property("scale", &Shape::scale)
    .property("offset", &Shape::offset)
    ;

  // Operation helpers.
  function("addMulPrepare", &binding_utils::addMulPrepareWrapper);
  function("floorPrepare", &binding_utils::floorPrepareWrapper);
  function("dequantizePrepare", &binding_utils::dequantizePrepareWrapper);
  function("depthwiseConvPrepare", &binding_utils::depthwiseConvPrepareWrapper);
  function("convPrepare", &binding_utils::convPrepareWrapper);

  // Operations.
  function("addFloat32", &binding_utils::addFloat32Wrapper, allow_raw_pointers());
  function("addQuant8", &binding_utils::addQuant8Wrapper, allow_raw_pointers());
  function("mulFloat32", &binding_utils::mulFloat32Wrapper, allow_raw_pointers());
  function("mulQuant8", &binding_utils::mulQuant8Wrapper, allow_raw_pointers());
  function("floorFloat32", &binding_utils::floorFloat32Wrapper, allow_raw_pointers());
  function("dequantizeQuant8ToFloat32", &binding_utils::dequantizeQuant8ToFloat32Wrapper, allow_raw_pointers());
  function("depthwiseConvFloat32", &binding_utils::depthwiseConvFloat32Wrapper, allow_raw_pointers());
  function("depthwiseConvQuant8", &binding_utils::depthwiseConvQuant8Wrapper, allow_raw_pointers());
  function("convFloat32", &binding_utils::convFloat32Wrapper, allow_raw_pointers());
  function("convQuant8", &binding_utils::convQuant8Wrapper, allow_raw_pointers());
  function("averagePoolFloat32", &averagePoolFloat32, allow_raw_pointers());
  function("averagePoolQuant8", &averagePoolQuant8, allow_raw_pointers());
  function("l2PoolFloat32", &l2PoolFloat32, allow_raw_pointers());
  function("maxPoolFloat32", &maxPoolFloat32, allow_raw_pointers());
  function("maxPoolQuant8", &maxPoolQuant8, allow_raw_pointers());
  function("reluFloat32", &reluFloat32, allow_raw_pointers());
  function("relu1Float32", &relu1Float32, allow_raw_pointers());
  function("relu6Float32", &relu6Float32, allow_raw_pointers());
  function("tanhFloat32", &tanhFloat32, allow_raw_pointers());
  function("logisticFloat32", &logisticFloat32, allow_raw_pointers());
  function("softmaxFloat32", &softmaxFloat32, allow_raw_pointers());
  function("reluQuant8", &reluQuant8, allow_raw_pointers());
  function("relu1Quant8", &relu1Quant8, allow_raw_pointers());
  function("relu6Quant8", &relu6Quant8, allow_raw_pointers());
  function("logisticQuant8", &logisticQuant8, allow_raw_pointers());
  function("softmaxQuant8", &softmaxQuant8, allow_raw_pointers());
  function("fullyConnectedFloat32", &fullyConnectedFloat32, allow_raw_pointers());
  function("fullyConnectedQuant8", &fullyConnectedQuant8, allow_raw_pointers());
  function("concatenationFloat32", &concatenationFloat32, allow_raw_pointers());
  function("concatenationQuant8", &concatenationQuant8, allow_raw_pointers());
  function("l2normFloat32", &l2normFloat32, allow_raw_pointers());
  function("l2normQuant8", &l2normQuant8, allow_raw_pointers());
  function("localResponseNormFloat32", &localResponseNormFloat32, allow_raw_pointers());
  function("reshapeGeneric", &reshapeGeneric, allow_raw_pointers());
  function("resizeBilinearFloat32", &resizeBilinearFloat32, allow_raw_pointers());
  function("depthToSpaceGeneric", &depthToSpaceGeneric, allow_raw_pointers());
  function("spaceToDepthGeneric", &spaceToDepthGeneric, allow_raw_pointers());
}