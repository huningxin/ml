describe('Operations Test', function() {
  const assert = chai.assert;

  function almostEqual(a, b) {
    const FLOAT_EPISILON = 1e-6;
    let delta = Math.abs(a - b);
    return delta < FLOAT_EPISILON;
  }

  it('addFloat32', function() {
    let tensor1Data = new Float32Array([0.5, 0.5, 0.5, 0.5]);
    let tensor1ByteLength = tensor1Data.length * tensor1Data.BYTES_PER_ELEMENT;
    let tensor1 = Module._malloc(tensor1ByteLength);
    let dataHeap = new Uint8Array(Module.HEAPU8.buffer, tensor1, tensor1ByteLength);
    dataHeap.set( new Uint8Array(tensor1Data.buffer) );
    let shape1 = new Module.Shape;
    shape1.type = Module.TENSOR_FLOAT32;
    shape1.dimensions = [5];

    let tensor2Data = new Float32Array([0.1, 0.1, 0.1, 0.1]);
    let tensor2ByteLength = tensor2Data.length * tensor2Data.BYTES_PER_ELEMENT;
    let tensor2 = Module._malloc(tensor2ByteLength);
    dataHeap = new Uint8Array(Module.HEAPU8.buffer, tensor2, tensor2ByteLength);
    dataHeap.set( new Uint8Array(tensor2Data.buffer) );
    let shape2 = new Module.Shape;
    shape2.type = Module.TENSOR_FLOAT32;
    shape2.dimensions = [5];

    let tensor3Data = new Float32Array(4);
    tensor3Data.fill(0);
    let tensor3ByteLength = tensor3Data.length * tensor3Data.BYTES_PER_ELEMENT;
    let tensor3 = Module._malloc(tensor2ByteLength);
    let shape3 = new Module.Shape;
    shape3.type = Module.TENSOR_FLOAT32;
    shape3.dimensions = [5];
    Module.addFloat32(tensor1, shape1, tensor2, shape2, Module.NONE, tensor3, shape3);
    
    dataHeap = new Uint8Array(Module.HEAPU8.buffer, tensor3, tensor3ByteLength);
    tensor3Buffer = new Uint8Array(tensor3Data.buffer);
    tensor3Buffer.set(dataHeap);

    for (let i = 0; i < tensor1Data.length; ++i) {
      assert.isTrue(almostEqual(tensor3Data[i], tensor1Data[i] + tensor2Data[i]));
    }
  });
});