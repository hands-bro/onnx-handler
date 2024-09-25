import onnx
from onnx_node_descriptor import ONNXNodeDescriptor
from onnx_tensor_descriptor import ONNXTensorDescriptor
import torch

class ONNXModelConverter:
    def __init__(self, onnx_model: onnx.ModelProto):
        self.onnx_model = onnx_model
        
        # Convert nodes to descriptors
        self.onnx_nodes: list[ONNXNodeDescriptor] = []
        for item in self.onnx_model.graph.node:
            # Analyze the node and append it to list
            self.onnx_nodes.append(ONNXNodeDescriptor(item))
            
        # Convert tensors to descriptors
        self.onnx_tensors: dict[str,ONNXNodeDescriptor] = {}
        for item in self.onnx_model.graph.initializer:
            if item.name in self.onnx_tensors:
                raise NotImplementedError(f"tensor '{item.name}' already exists. (Duplicate names are not supported yet)")
            
            # Analyze the tensor and append it to dict
            self.onnx_tensors[item.name] = ONNXTensorDescriptor(item)
    
    def __del__(self):
        pass
    
    def _get_tensor_descriptor(self, keyword: str) -> ONNXTensorDescriptor:
        if keyword not in self.onnx_tensors:
            raise KeyError(f"keyword '{keyword}' does not exist.")
        return self.onnx_tensors[keyword]
    
    def convert_to_torch_model(self, include_weight_and_bias: bool = True) -> torch.nn.Module:
        layers: list[torch.nn.Module] = []
        
        for node in self.onnx_nodes:            
            if node.op_type == "Flatten":
                layers.append(torch.nn.Flatten(start_dim = node.axis))
                
            elif node.op_type == "Gemm":
                weight = self._get_tensor_descriptor(node.weight_name)
                use_bias = False if node.bias_name == None else True
                layer = torch.nn.Linear(
                    in_features = weight.in_features,
                    out_features = weight.out_features,
                    bias = use_bias)
                if include_weight_and_bias is False:
                    layers.append(layer)
                    continue
                layer.weight.data = torch.nn.Parameter(torch.from_numpy(weight.convert_tensor_to_numpy().copy()))
                if use_bias:
                    bias = self._get_tensor_descriptor(node.bias_name)
                    layer.bias.data = torch.nn.Parameter(torch.from_numpy(bias.convert_tensor_to_numpy().copy()))
                layers.append(layer)
                
            elif node.op_type == "Relu":
                layers.append(torch.nn.ReLU())
        
        # Define a custom sequential model
        class SequentialModel(torch.nn.Module):
            def __init__(self, layers):
                super(SequentialModel, self).__init__()
                self.model = torch.nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)
        
        return SequentialModel(layers)

################################################################
# Test cases
################################################################

if __name__ == "__main__":
    import numpy
    import torch
    import torch.utils.data
    import torchvision
    import onnx

    # Basic variables
    batch_size = 64

    # Dataset & DataLoader
    transform = torchvision.transforms.ToTensor()
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Target device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-trained model created in advance using the SimpleMNISTModel above
    onnx_filename = f"./models/onnx_model.onnx"

    # Load an ONNX model
    onnx_model: onnx.ModelProto = onnx.load(onnx_filename)

    # Print out
    print("ONNX Model >>>")
    print(onnx_model)
    print()

    # Convert to PyTorch model
    pytorch_model = ONNXModelConverter(onnx_model).convert_to_torch_model()

    print("PyTorch Model (converted from the ONNX model) >>>")
    print(pytorch_model)
    print()

    # Set the target device
    pytorch_model = pytorch_model.to(device)

################################################################
