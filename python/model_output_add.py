
import openvino as ov
import numpy as np
from openvino import opset8 as opset
from openvino import Core, Model, Type, Shape, op
import utils.common as common_utils
import argparse
import os

def my_model():
    input = opset.parameter([1, 256, 32, 32], Type.f32, name='input')

    weight_arr = np.random.uniform(low=-1, high=1.0, size=[1024,256,1,1]).astype(np.float32)
    weight = opset.constant(weight_arr, Type.f32, name='weight')

    strides = [1, 1]
    pads_begin = [1, 1]
    pads_end = [1, 1]
    dilations = [1, 1]
    conv = opset.convolution(input, weight, strides, pads_begin, pads_end, dilations)

    add = opset.add(conv, np.random.uniform(low=-1, high=1.0, size=[1,1024,1,1]).astype(np.float32), name='op_add')

    op_gelu = opset.gelu(add, approximation_mode="ERF")
 
    Result = opset.result(op_gelu, name='output')
    return Model([Result], [input], 'model_gelu')

def add_new_output(ov_model:ov.Model, name):
    found_node_output = None
    
    print(f"\nSearching for node with name: '{name}'")
    available_nodes = []
    
    for op in ov_model.get_ordered_ops():
        available_nodes.append(op.get_friendly_name())
        if op.get_friendly_name() == name:
            # Assuming op has one output, take its first output port
            found_node_output = op.output(0)
            print(f"Found target node: '{name}' with type '{op.get_type_name()}'")
            break

    new_output_name = name + "_output"
    if found_node_output:
        # Create a new Result node connected to the found output
        new_result = ov.opset12.result(found_node_output)
        new_result.set_friendly_name(new_output_name) # Set a friendly name
        new_result.output(0).set_names({new_output_name}) # Set tensor names

        # Add the new result node to the model's outputs
        ov_model.add_results([new_result])
        print(f"== Added new output: {new_output_name}")
    else:
        print(f"== Error: Could not find node '{name}' to add as a new output.")
        print(f"Available nodes: {available_nodes[:10]}...")  # Show first 10 available nodes
        if len(available_nodes) > 10:
            print(f"... and {len(available_nodes) - 10} more nodes")

    return ov_model

def test():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Add new output to OpenVINO model')
    parser.add_argument('-i', '--input', type=str, help='Path to input model file (.xml)')
    parser.add_argument('-o', '--output_node', type=str, default='op_add', 
                        help='Name of the node to add as new output (default: op_add)')
    parser.add_argument('-d', '--device', type=str, default='CPU', 
                        help='Device to run inference on (default: CPU)')
    
    args = parser.parse_args()
    
    # Decide which model to use based on whether model path is provided
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Model file '{args.input}' does not exist.")
            return
        
        print(f"Loading model from: {args.input}")
        model = ov.read_model(args.input)
        print("Model loaded successfully!")
    else:
        print("No input model specified, using default generated model...")
        model = my_model()

    # Display original model information
    print("\n=== Original Model Info ===")
    common_utils.print_model_info(model)

    # Add new output
    new_result_name = args.output_node
    model = add_new_output(model, new_result_name)
    
    # Display modified model information
    print("\n=== Modified Model Info ===")
    common_utils.print_model_info(model)

    # Compile model
    print(f"\n=== Compiling model for device: {args.device}")
    cm = ov.compile_model(model, args.device)

    # Prepare input data
    input_shape = model.input(0).get_shape()
    print(f"\n=== Preparing input data with shape: {input_shape}")
    input_data = np.random.uniform(low=0, high=1.0, size=input_shape).astype(np.float32)

    # Run inference
    print("\n=== Running inference...")
    output = cm(input_data)
    
    # Display output results
    print(f"\n=== Inference results:")
    print(f"=== Number of outputs: {len(output)}")
    
    for i, (key, value) in enumerate(output.items()):
        print(f"===\t Output {i}: key='{key}', shape={value.shape}")
        # Display first 10 values of each output
        flat_values = value.flatten()
        num_values_to_show = min(10, len(flat_values))
        print(f"===\t First {num_values_to_show} values: {flat_values[:num_values_to_show]}")
    
if __name__ ==  "__main__":
    test()