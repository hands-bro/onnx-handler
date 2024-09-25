import enum
import io
import numpy
import onnx
import onnxoptimizer, onnxsim
import onnxruntime, onnxruntime.quantization
import onnx2pytorch
import onnx2torch       # Recommended
import torch
from typing import Optional, Union
from multipledispatch import dispatch

class QuantType(enum.Enum):
    INT8 = 0
    UINT8 = 1
    INT16 = 3
    UINT16 = 4
    INT4 = 5
    UINT4 = 6

class ONNXHandler:
    def __init__(self):
        pass
    
    def __del__(self):
        pass
    
    @staticmethod
    def load_onnx_model(filename: str) -> onnx.ModelProto:
        onnx_model: onnx.ModelProto = onnx.load(filename)
        return onnx_model
    
    @staticmethod
    @dispatch(onnx.ModelProto, str)
    def save_onnx_model(model: onnx.ModelProto, filename: str) -> None:
        onnx.save(model, filename)
    
    @staticmethod
    @dispatch(torch.nn.Module, str, torch.device, int, int, int, int)
    def save_onnx_model(torch_model: torch.nn.Module, filename: str, device: torch.device, 
        batch_size: int, channels: int, height: int, width: int) -> None:
        # Generate a dummy data
        onnx_input = torch.randn(batch_size, channels, height, width).to(device)
        
        # Convert the PyTorch model to ONNX (dynamic) model
        torch.onnx.export(
                model=torch_model,                          # model being run
                args=onnx_input,                            # model input (or a tuple for multiple inputs)
                f=filename,                                 # where to save the model (can be a file or file-like object)
                verbose=False,                              # Whether to display the model structure
                opset_version=11,                           # the ONNX version to export the model to
                do_constant_folding=True,                   # whether to execute constant folding for optimization
                input_names=['input'],                      # the model's input names
                output_names=['output'],                    # the model's output names
                dynamic_axes={'input' : {0:'batch_size'},   # variable length axes (for using dynamic batch)
                              'output' : {0:'batch_size'}}
        )
    
    @staticmethod
    @dispatch(object)
    def save_onnx_model(model: Union[onnx.ModelProto, torch.nn.Module], *args) -> None:
        raise Exception("unsupported type")
    
    @staticmethod
    def save_model_as_numpy(onnx_model: onnx.ModelProto, npz_filename: str) -> None:
        numpy_model = ONNXHandler.convert_model_onnx2numpy(onnx_model)
        numpy.savez(npz_filename, **numpy_model)
    
    @staticmethod
    def convert_model_torch2onnx(torch_model: torch.nn.Module, device: torch.device, 
        batch_size: int, channels: int, height: int, width: int) -> onnx.ModelProto:
        
        # Create a buffer
        memory_buffer = io.BytesIO()
        
        # Generate a dummy data
        onnx_input = torch.randn(batch_size, channels, height, width).to(device)
        
        # Convert the PyTorch model to ONNX (dynamic) model
        torch.onnx.export(
                model=torch_model,                          # model being run
                args=onnx_input,                            # model input (or a tuple for multiple inputs)
                f=memory_buffer,                            # where to save the model (can be a file or file-like object)
                verbose=False,                              # Whether to display the model structure
                opset_version=11,                           # the ONNX version to export the model to
                do_constant_folding=True,                   # whether to execute constant folding for optimization
                input_names=['input'],                      # the model's input names
                output_names=['output'],                    # the model's output names
                dynamic_axes={'input' : {0:'batch_size'},   # variable length axes (for using dynamic batch)
                              'output' : {0:'batch_size'}}
        )
        
        # Get an ONNX model from the buffer
        onnx_model = onnx.load_model_from_string(memory_buffer.getvalue())
        
        return onnx_model

    @staticmethod
    def convert_model_onnx2numpy(onnx_model: onnx.ModelProto) -> dict[str, numpy.ndarray]:
        numpy_model = {init.name: onnx.numpy_helper.to_array(init) for init in onnx_model.graph.initializer}
        return numpy_model

    @staticmethod
    def convert_model_onnx2pytorch(onnx_model: onnx.ModelProto) -> torch.nn.Module:
        """
        Convert an ONNX model to the PyTorch model using onnx2pytorch package. (Not recommended)
        
        Note
        --------
        Input with larger batch size than 1 not supported yet.
        That is, you need to separately declare a data loader with batch size 1.
        """
        return onnx2pytorch.ConvertModel(onnx_model)
    
    @staticmethod
    def convert_model_onnx2torch(onnx_model: onnx.ModelProto) -> torch.nn.Module:
        """
        Convert an ONNX model to the PyTorch model using onnx2torch package.
        
        Note
        --------
        Input with larger batch size than 1 not supported yet.
        """
        return onnx2torch.convert(onnx_model)
    
    @staticmethod
    def optimize_onnx_model(model: onnx.ModelProto) -> onnx.ModelProto:
        passes = ["eliminate_deadend", "fuse_bn_into_conv", "fuse_add_bias_into_conv"]  # options
        return onnxoptimizer.optimize(model, passes)
    
    @staticmethod
    def simplify_onnx_model(model: onnx.ModelProto) -> onnx.ModelProto:
        simplified_model, check = onnxsim.simplify(model)
        assert check, "Simplified ONNX model could not be validated"
        return simplified_model

    @staticmethod
    def quantize_onnx_model(
        model: onnx.ModelProto,
        preprocessed_model_filename: str,
        quantized_model_filename: str,
        quant_type: QuantType = QuantType.UINT8) -> None:
        
        # Set the target quant type
        if quant_type == QuantType.INT8:
            raise Exception("unsupported yet")
        elif quant_type == QuantType.UINT8:
            quant_type = onnxruntime.quantization.QuantType.QUInt8
        elif quant_type == QuantType.INT16:
            raise Exception("unsupported yet")
        elif quant_type == QuantType.UINT16:
            raise Exception("unsupported yet")
        elif quant_type == QuantType.INT4:
            raise Exception("unsupported yet")
        elif quant_type == QuantType.UINT4:
            raise Exception("unsupported yet")
        else:
            raise Exception("unknown quant type")
        
        # Preprocessing
        onnxruntime.quantization.quant_pre_process(model, preprocessed_model_filename, skip_simbolic_shape=False)
        
        # ONNX Runtime Quantization
        prefix = ["MatMul", "Add", "Relu"]
        names = {node.name for node in model.graph.node}
        linear_names = [v for v in names if v.split("_")[0] in prefix]
        onnxruntime.quantization.quantize_dynamic(preprocessed_model_filename,
                                          quantized_model_filename,
                                          weight_type=quant_type,
                                          #nodes_to_quantize=linear_names,
                                          extra_options={"MatMulConstBOnly":True}
                                          )
