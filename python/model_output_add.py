
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

def compare_cpu_gpu_outputs(ov_model: ov.Model, input_data, tolerance=1e-5):
    """Compare outputs of all layers between CPU and GPU devices"""
    print("\n=== Comparing CPU vs GPU outputs for all layers ===")
    
    # Get all operation nodes that can be used as outputs
    all_ops = ov_model.get_ordered_ops()
    
    # Create a copy of the model to add multiple outputs
    model_with_all_outputs = ov_model
    
    # Add all intermediate layers as outputs for comparison
    added_outputs = []
    for i, op in enumerate(all_ops):
        if op.get_output_size() > 0 and op.get_type_name() not in ['Parameter', 'Constant', 'Result']:
            try:
                output_name = f"{op.get_friendly_name()}_debug_output"
                new_result = ov.opset12.result(op.output(0))
                new_result.set_friendly_name(output_name)
                new_result.output(0).set_names({output_name})
                model_with_all_outputs.add_results([new_result])
                added_outputs.append((op.get_friendly_name(), output_name, op.get_type_name()))
                print(f"  Added debug output for: {op.get_friendly_name()} (type: {op.get_type_name()})")
            except Exception as e:
                print(f"  Warning: Could not add output for {op.get_friendly_name()}: {e}")
    
    print(f"\nTotal debug outputs added: {len(added_outputs)}")
    
    # Compile for CPU and GPU
    print("\nCompiling models...")
    try:
        cm_cpu = ov.compile_model(model_with_all_outputs, "CPU")
        print("✓ CPU compilation successful")
    except Exception as e:
        print(f"✗ CPU compilation failed: {e}")
        return
    
    try:
        cm_gpu = ov.compile_model(model_with_all_outputs, "GPU")
        print("✓ GPU compilation successful")
    except Exception as e:
        print(f"✗ GPU compilation failed: {e}")
        print("Note: GPU might not be available on this system")
        return
    
    # Run inference on both devices
    print("\nRunning inference on both devices...")
    cpu_outputs = cm_cpu(input_data)
    gpu_outputs = cm_gpu(input_data)
    
    # Compare outputs
    print(f"\n=== Layer-by-layer comparison (tolerance: {tolerance}) ===")
    mismatched_layers = []
    
    for layer_name, output_name, layer_type in added_outputs:
        if output_name in cpu_outputs and output_name in gpu_outputs:
            cpu_out = cpu_outputs[output_name]
            gpu_out = gpu_outputs[output_name]
            
            # Calculate difference
            diff = np.abs(cpu_out - gpu_out)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            # Check if difference exceeds tolerance
            is_mismatch = max_diff > tolerance
            
            status = "❌ MISMATCH" if is_mismatch else "✅ MATCH"
            print(f"{status} {layer_name:20s} ({layer_type:15s}) | Max diff: {max_diff:.2e} | Mean diff: {mean_diff:.2e}")
            
            if is_mismatch:
                mismatched_layers.append({
                    'name': layer_name,
                    'type': layer_type,
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'cpu_shape': cpu_out.shape,
                    'gpu_shape': gpu_out.shape
                })
                
                # Show some sample values for mismatched layers
                print(f"    CPU sample values: {cpu_out.flatten()[:5]}")
                print(f"    GPU sample values: {gpu_out.flatten()[:5]}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Total layers compared: {len(added_outputs)}")
    print(f"Mismatched layers: {len(mismatched_layers)}")
    
    if mismatched_layers:
        print("\nLayers with significant differences:")
        for layer in mismatched_layers:
            print(f"  - {layer['name']} ({layer['type']}): max_diff={layer['max_diff']:.2e}")
    else:
        print("All layers match within tolerance!")
    
    return mismatched_layers

def create_simple_test_input(input_shape):
    """Create simple test input data for debugging"""
    print(f"\nCreating simple test input with shape: {input_shape}")
    
    # Option 1: All ones
    # input_data = np.ones(input_shape, dtype=np.float32)
    
    # Option 2: Sequential values (more likely to reveal differences)
    total_elements = np.prod(input_shape)
    input_data = np.arange(total_elements, dtype=np.float32).reshape(input_shape)
    input_data = input_data / total_elements  # Normalize to [0, 1] range
    
    # Option 3: Small random values with fixed seed for reproducibility
    # np.random.seed(42)
    # input_data = np.random.uniform(0.1, 0.9, input_shape).astype(np.float32)
    
    print(f"Input data statistics: min={np.min(input_data):.6f}, max={np.max(input_data):.6f}, mean={np.mean(input_data):.6f}")
    return input_data

def test():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Add new output to OpenVINO model')
    parser.add_argument('-i', '--input', type=str, help='Path to input model file (.xml)')
    parser.add_argument('-o', '--output_node', type=str, default='op_add', 
                        help='Name of the node to add as new output (default: op_add)')
    parser.add_argument('-d', '--device', type=str, default='CPU', 
                        help='Device to run inference on (default: CPU)')
    parser.add_argument('--compare', action='store_true', 
                        help='Compare CPU vs GPU outputs for all layers')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help='Tolerance for CPU vs GPU comparison (default: 1e-5)')
    parser.add_argument('--simple-input', action='store_true',
                        help='Use simple sequential input data instead of random')
    
    args = parser.parse_args()
    core = ov.Core()
    print(f"Using OpenVINO version: {ov.get_version()}")
    # Decide which model to use based on whether model path is provided
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Model file '{args.input}' does not exist.")
            return
        
        print(f"Loading model from: {args.input}")
        model = core.read_model(args.input)
        print("Model loaded successfully!")
    else:
        print("No input model specified, using default generated model...")
        model = my_model()

    # Display original model information
    print("\n=== Original Model Info ===")
    # common_utils.print_model_info(model)

    # If compare mode is enabled, run CPU vs GPU comparison
    if args.compare:
        input_shape = model.input(0).get_shape()
        print(f"\n=== Preparing test input for comparison ===")
        
        if args.simple_input:
            input_data = create_simple_test_input(input_shape)
        else:
            # Use simple sequential data for better reproducibility
            input_data = create_simple_test_input(input_shape)
        
        # Run comparison
        mismatched_layers = compare_cpu_gpu_outputs(model, input_data, args.tolerance)
        
        # Exit after comparison unless user wants to continue
        return

    # Compile model
    # print(f"\n=== Compiling model for device: {args.device}")
    # cm = ov.compile_model(model, args.device)

    # # Prepare input data
    # input_shape = model.input(0).get_shape()
    # print(f"\n=== Preparing input data with shape: {input_shape}")
    
    # if args.simple_input:
    #     input_data = create_simple_test_input(input_shape)
    # else:
    #     input_data = np.random.uniform(low=0, high=1.0, size=input_shape).astype(np.float32)

    # Run inference
    # print("\n=== Running inference...")
    # output = cm(input_data)
    
    # # Display output results
    # print(f"\n=== Inference results:")
    # print(f"=== Number of outputs: {len(output)}")
    
    # for i, (key, value) in enumerate(output.items()):
    #     print(f"===\t Output {i}: key='{key}', shape={value.shape}")
    #     # Display first 10 values of each output
    #     flat_values = value.flatten()
    #     num_values_to_show = min(10, len(flat_values))
    #     print(f"===\t First {num_values_to_show} values: {flat_values[:num_values_to_show]}")
    
if __name__ ==  "__main__":
    test()