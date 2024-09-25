import onnx
import numpy
from typing import Union

class ONNXTensorDescriptor:
    def __init__(self, onnx_tensor: onnx.TensorProto):
        self.onnx_tensor: onnx.TensorProto = onnx_tensor
        
        # Declare variables for layer properties
        self.name: str = self.onnx_tensor.name
        self.data_type: int = self.onnx_tensor.data_type
        self.in_features: Union[None,int] = None
        self.out_features: Union[None,int] = None
        
        # Get all dimensions as a list (rearrange them in reverse order)
        self._dims: list[int] = []
        tensor_count: int = len(self.onnx_tensor.dims)
        for i in range(tensor_count):
            self._dims.append(self.onnx_tensor.dims[i])
        self._dims.reverse()
        
        self.in_features = self._dims[0]
        
        if tensor_count == 2:
            self.out_features = self._dims[1]
            
        # Check the validation
        if tensor_count > 2:
            raise NotImplementedError(f"layers with structures greater than 3-D are not yet supported.")
    
    def __del__(self):
        pass
    
    def convert_tensor_to_numpy(self) -> numpy.ndarray:
        return onnx.numpy_helper.to_array(self.onnx_tensor)
