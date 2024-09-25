import onnx
from typing import Any, Union

class ONNXNodeDescriptor:
    def __init__(self, onnx_node: onnx.NodeProto):
        # Declare variables for layer properties
        self.name: str = onnx_node.name
        self.op_type: str = onnx_node.op_type
        self.input_name: Union[None,str] = None
        self.output_name: Union[None,str] = None
        self.weight_name: Union[None,str] = None
        self.bias_name: Union[None,str] = None
        self.axis: Union[None,int] = None
        self.alpha: float = 1.0
        self.beta: float = 1.0
        self.is_transpose_weight: bool = False    # No use, automatically handled by onnx.numpy_helper.to_array()
        
        # Get all inputs as a list
        self._inputs: list[str] = []
        for item in onnx_node.input:
            self._inputs.append(item)
        self.input_name = self._inputs[0]

        # Get all outputs as a list
        self._outputs: list[str] = []
        for item in onnx_node.output:
            self._outputs.append(item)
        self.output_name = self._outputs[0]
        
        # Get all attributes as a dict
        self._attributes: dict = {}
        for item in onnx_node.attribute:
            if item.name in self._attributes:
                raise NotImplementedError(f"attribute '{item.name}' already exists. (Multiple attributes are not supported yet)")
            if item.type == 1:  # AttributeType FLOAT
                self._attributes[item.name] = float(item.f)
            elif item.type == 2:  # AttributeType INT
                self._attributes[item.name] = int(item.i)
            else:
                raise NotImplementedError(f"AttributeType '{item.type}' is not supported yet.")
    
        # Define the processing method according to the layer type
        self._init_layer_properties()
    
    def __del__(self):
        pass
    
    def _get_attribute(self, keyword: str) -> Any:
        if keyword not in self._attributes:
            raise KeyError(f"keyword '{keyword}' does not exist.")
        return self._attributes[keyword]
        
    def _init_layer_properties(self) -> None:
        # Define the processing method according to the layer type
        if self.op_type == "Flatten":
            self.axis = self._get_attribute("axis")
            
        elif self.op_type == "Gemm":
            self.weight_name = self._inputs[1]
            if len(self._inputs) >= 3:
                self.bias_name = self._inputs[2]
            self.alpha = float(self._get_attribute("alpha"))
            self.beta = float(self._get_attribute("beta"))
            self.is_transpose_weight = bool(self._get_attribute("transB"))
            
        elif self.op_type == "Relu":
            pass
