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

  bool addFloat32Wrapper(intptr_t in1, const Shape& shape1, intptr_t in2, const Shape& shape2, int32_t activation, intptr_t out, const Shape& shapeOut) {
    return addFloat32((const float*)in1, shape1, (const float*)in2, shape2, activation, (float*)out, shapeOut);
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

  function("addFloat32", &binding_utils::addFloat32Wrapper, allow_raw_pointers());
  function("addQuant8", &addQuant8, allow_raw_pointers());
  function("mulFloat32", &mulFloat32, allow_raw_pointers());
  function("mulQuant8", &mulQuant8, allow_raw_pointers());
  function("floorFloat32", &floorFloat32, allow_raw_pointers());
  function("dequantizeQuant8ToFloat32", &dequantizeQuant8ToFloat32, allow_raw_pointers());
  function("depthwiseConvFloat32", &depthwiseConvFloat32, allow_raw_pointers());
  function("depthwiseConvQuant8", &depthwiseConvQuant8, allow_raw_pointers());
  function("convFloat32", &convFloat32, allow_raw_pointers());
  function("convQuant8", &convQuant8, allow_raw_pointers());
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