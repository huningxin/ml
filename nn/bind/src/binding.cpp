#include <emscripten/bind.h>

#include "Operations.h"
#include "OperationsUtils.h"

using namespace emscripten;
using namespace android::nn;

namespace binding_utils {
  Shape* createShape(OperandType type, val js_dims, float scale, int32_t offset) {
    Shape* shape = new Shape;
    shape->type = type;
    shape->dimensions = vecFromJSArray<uint32_t>(js_dims);
    shape->scale = scale;
    shape->offset = offset;
    return shape;
  }

  val getShapeDimensions(const Shape& shape) {
    emscripten::val js_dims = emscripten::val::array();
    for (int i = 0; i < shape.dimensions.size(); i++) {
      js_dims.call<void>("push", shape.dimensions[i]);
    }
    return js_dims;
  }
}

EMSCRIPTEN_BINDINGS(nn)
{
  enum_<FusedActivationFunc>("FusedActivationFunc")
    .value("NONE", FusedActivationFunc::NONE)
    .value("RELU", FusedActivationFunc::RELU)
    .value("RELU1", FusedActivationFunc::RELU1)
    .value("RELU6", FusedActivationFunc::RELU6)
    ;

  enum_<OperandType>("OperandType")
    .value("FLOAT32", OperandType::FLOAT32)
    .value("INT32", OperandType::INT32)
    .value("UINT32", OperandType::UINT32)
    .value("TENSOR_FLOAT32", OperandType::TENSOR_FLOAT32)
    .value("TENSOR_INT32", OperandType::TENSOR_INT32)
    .value("TENSOR_QUANT8_ASYMM", OperandType::TENSOR_QUANT8_ASYMM)
    ;

  class_<Shape>("Shape")
    .constructor(&binding_utils::createShape, allow_raw_pointers())
    .property("type", &Shape::type)
    .property("dimensions", &binding_utils::getShapeDimensions)
    .property("scale", &Shape::scale)
    .property("offset", &Shape::offset)
    ;

  function("addFloat32", &addFloat32, allow_raw_pointers());
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