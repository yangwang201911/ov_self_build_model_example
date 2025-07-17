import openvino as ov
import numpy as np
from openvino import opset8 as opset
from openvino import Core, Model, Type, Shape, op
import utils.common as common_utils
import argparse
import os

def my_model():
    input = opset.parameter([1, 256, 32, 32], Type.f32, name='input')

    weight_arr = np.full([1024,256,1,1], 1.5, dtype=np.float32)
    weight = opset.constant(weight_arr, Type.f32, name='weight')

    strides = [1, 1]
    pads_begin = [1, 1]
    pads_end = [1, 1]
    dilations = [1, 1]
    conv = opset.convolution(input, weight, strides, pads_begin, pads_end, dilations)

    add = opset.add(conv, np.full([1,1024,1,1], 1, dtype=np.float32), name='op_add')

    op_gelu = opset.gelu(add, approximation_mode="ERF")
 
    Result = opset.result(op_gelu, name='output')
    Result.output(0).set_names({'output'})
    return Model([Result], [input], 'model_add')

def add_new_output(ov_model: ov.Model, name_list):
    """
    Add new outputs to the model for specified nodes.
    
    Args:
        ov_model: OpenVINO model
        name_list: List of node names to add as outputs, or single string name
    
    Returns:
        Modified model with new outputs
    """
    # Convert single string to list for uniform processing
    if isinstance(name_list, str):
        name_list = [name_list]
    
    if not name_list:
        print("No node names provided, skipping output addition.")
        return ov_model
    
    print(f"\nSearching for nodes with names: {name_list}")
    available_nodes = []
    added_count = 0
    
    # Get all available nodes for error reporting
    all_ops = ov_model.get_ordered_ops()
    for op in all_ops:
        available_nodes.append(op.get_friendly_name())
    
    # Search for each requested node
    for name in name_list:
        found_node_output = None
        
        for op in all_ops:
            if op.get_friendly_name() == name:
                # Assuming op has one output, take its first output port
                found_node_output = op.output(0)
                print(f"Found target node: '{name}' with type '{op.get_type_name()}'")
                break

        new_output_name = name + "_output"
        if found_node_output:
            try:
                # Create a new Result node connected to the found output
                new_result = ov.opset12.result(found_node_output)
                new_result.set_friendly_name(new_output_name) # Set a friendly name
                new_result.output(0).set_names({new_output_name}) # Set tensor names

                # Add the new result node to the model's outputs
                ov_model.add_results([new_result])
                print(f"== Added new output: {new_output_name}")
                added_count += 1
            except Exception as e:
                print(f"== Error: Failed to add output for '{name}': {e}")
        else:
            print(f"== Error: Could not find node '{name}' to add as a new output.")
    
    if added_count == 0:
        print(f"Available nodes: {available_nodes[:10]}...")  # Show first 10 available nodes
        if len(available_nodes) > 10:
            print(f"... and {len(available_nodes) - 10} more nodes")
    else:
        print(f"Successfully added {added_count} new outputs from {len(name_list)} requested nodes.")

    return ov_model

def compare_cpu_gpu_outputs(ov_model: ov.Model, input_data, tolerance=1e-5, precision="FP32"):
    """Compare outputs of all layers between CPU and GPU devices"""
    print("\n=== Comparing CPU vs GPU outputs for all layers ===")
    print(f"Using precision: {precision}")
    
    # Collect all model outputs for comparison
    outputs_info = []
    print(f"=== Model Outputs ===")
    for i, output in enumerate(ov_model.outputs):
        output_name = list(output.get_names())[0] if output.get_names() else output.get_any_name()
        source_op = output.get_node()
        outputs_info.append((source_op.get_friendly_name(), output_name, source_op.get_type_name()))
        print(f"  Output {i}: {output_name} (from {source_op.get_friendly_name()}, type: {source_op.get_type_name()})")
    
    # Move the first element to the last position
    if len(outputs_info) > 1:
        first_element = outputs_info.pop(0)
        outputs_info.append(first_element)
        print(f"\nReordered outputs: moved first output to last position for final model output comparison")
    
    print(f"\nTotal outputs for comparison: {len(outputs_info)}")

    # Compile for CPU and GPU
    print("\nCompiling models...")
    try:
        cm_cpu = ov.compile_model(ov_model, "CPU", {"INFERENCE_PRECISION_HINT": precision})
        print("✓ CPU compilation successful")
    except Exception as e:
        print(f"✗ CPU compilation failed: {e}")
        return
    
    try:
        cm_gpu = ov.compile_model(ov_model, "GPU", {"INFERENCE_PRECISION_HINT": precision})
        print("✓ GPU compilation successful")
    except Exception as e:
        print(f"✗ GPU compilation failed: {e}")
        print("Note: GPU might not be available on this system")
        return
    
    # Run inference on both devices
    print("\nRunning inference on both devices...")
    cpu_outputs = cm_cpu(input_data)
    gpu_outputs = cm_gpu(input_data)
    
    print(f"\n=== Debug: Expected output names from outputs_info ===")
    for layer_name, output_name, layer_type in outputs_info:
        print(f"  Expected: '{output_name}' (from {layer_name}, type: {layer_type})")
    
    # Compare outputs
    print(f"\n=== Output Comparison (tolerance: {tolerance}) ===")
    mismatched_layers = []
    final_output_matches = True  # Track if the final output matches
    
    for idx, (layer_name, output_name, layer_type) in enumerate(outputs_info, 1):
        is_final_output = (idx == len(outputs_info))  # Check if this is the final output
        
        # Extract node order from output name if it's a debug output
        node_order_info = ""
        if "_debug_output_node" in output_name:
            try:
                node_num = output_name.split("_debug_output_node")[1]
                node_order_info = f" (Model Node #{node_num})"
            except:
                pass
        
        if is_final_output:
            print(f"\n =====   Model output name: {output_name}{node_order_info} ======")
        else:
            print(f"\n =====   Layer output name: {output_name}{node_order_info} ======")
        if output_name in cpu_outputs and output_name in gpu_outputs:
            cpu_out = cpu_outputs[output_name]
            gpu_out = gpu_outputs[output_name]
            
            # Check for NaN or Inf values first
            cpu_has_nan = np.isnan(cpu_out).any()
            cpu_has_inf = np.isinf(cpu_out).any()
            gpu_has_nan = np.isnan(gpu_out).any()
            gpu_has_inf = np.isinf(gpu_out).any()
            
            # Calculate difference (handle NaN case)
            diff = np.abs(cpu_out - gpu_out)
            max_diff = np.max(diff) if not np.isnan(diff).any() else float('inf')
            mean_diff = np.mean(diff) if not np.isnan(diff).any() else float('inf')
            
            # Check if difference exceeds tolerance or if there are NaN/Inf values
            has_invalid_values = cpu_has_nan or cpu_has_inf or gpu_has_nan or gpu_has_inf
            is_mismatch = max_diff > tolerance or has_invalid_values
            
            # Track final output match status
            if is_final_output:
                final_output_matches = not is_mismatch
            
            # Use unified formatting for all outputs
            status = "❌ MISMATCH" if is_mismatch else "✅ MATCH"
            
            # Add warning indicators for invalid values
            warning_indicators = []
            if cpu_has_nan or gpu_has_nan:
                warning_indicators.append("NaN")
            if cpu_has_inf or gpu_has_inf:
                warning_indicators.append("Inf")
            warning_str = f" [{'/'.join(warning_indicators)}]" if warning_indicators else ""
            
            print(f"[{idx:3d}] {status} {layer_name:20s} ({layer_type:15s}){node_order_info} | Max diff: {max_diff:.2e} | Mean diff: {mean_diff:.2e}{warning_str}")
            
            # Show detailed invalid value information
            if has_invalid_values:
                print(f"    ⚠️  Invalid values detected:")
                if cpu_has_nan:
                    cpu_nan_count = np.isnan(cpu_out).sum()
                    print(f"        CPU NaN count: {cpu_nan_count}")
                if cpu_has_inf:
                    cpu_inf_count = np.isinf(cpu_out).sum()
                    print(f"        CPU Inf count: {cpu_inf_count}")
                if gpu_has_nan:
                    gpu_nan_count = np.isnan(gpu_out).sum()
                    print(f"        GPU NaN count: {gpu_nan_count}")
                if gpu_has_inf:
                    gpu_inf_count = np.isinf(gpu_out).sum()
                    print(f"        GPU Inf count: {gpu_inf_count}")
                
                # For NaN cases, show CPU sample values for comparison
                if cpu_has_nan or gpu_has_nan:
                    print(f"        CPU sample values: {cpu_out.flatten()[:10]}")
                    if not gpu_has_nan:  # Only show GPU values if they don't contain NaN
                        print(f"        GPU sample values: {gpu_out.flatten()[:10]}")
                    else:
                        # When GPU has NaN, show CPU values at NaN positions
                        flat_cpu = cpu_out.flatten()
                        flat_gpu = gpu_out.flatten()
                        nan_indices = np.where(np.isnan(flat_gpu))[0]
                        
                        if len(nan_indices) > 0:
                            # Show first 10 NaN positions and corresponding CPU values
                            show_count = min(10, len(nan_indices))
                            print(f"        GPU NaN positions and corresponding CPU values:")
                            for i in range(show_count):
                                idx = nan_indices[i]
                                cpu_val = flat_cpu[idx]
                                print(f"          Index {idx}: GPU=nan, CPU={cpu_val:.6f}")
                            
                            if len(nan_indices) > 10:
                                print(f"          ... and {len(nan_indices) - 10} more NaN positions")
                elif cpu_has_inf or gpu_has_inf:
                    # For Inf-only cases, show both CPU and GPU sample values
                    print(f"        CPU sample values: {cpu_out.flatten()[:10]}")
                    print(f"        GPU sample values: {gpu_out.flatten()[:10]}")
            
            # Track mismatches
            if is_mismatch:
                mismatch_info = {
                    'name': layer_name,
                    'output_name': output_name,
                    'type': layer_type,
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'cpu_shape': cpu_out.shape,
                    'gpu_shape': gpu_out.shape,
                    'is_final': is_final_output,
                    'node_order': node_order_info,
                    'has_nan': cpu_has_nan or gpu_has_nan,
                    'has_inf': cpu_has_inf or gpu_has_inf
                }
                mismatched_layers.append(mismatch_info)
                
                # Only show detailed analysis if values are not NaN/Inf
                if not has_invalid_values:
                    # Find and show the location and values with largest deviations
                    flat_cpu = cpu_out.flatten()
                    flat_gpu = gpu_out.flatten()
                    flat_diff = np.abs(flat_cpu - flat_gpu)
                    
                    # Find indices of top 3 largest differences
                    top_diff_indices = np.argsort(flat_diff)[-3:][::-1]  # Get top 3, reverse to largest first
                    
                    print(f"    Top {len(top_diff_indices)} largest deviations:")
                    for i, idx in enumerate(top_diff_indices, 1):
                        deviation = flat_diff[idx]
                        cpu_val = flat_cpu[idx]
                        gpu_val = flat_gpu[idx]
                        
                        # Show neighboring values (±2 around the index)
                        start_idx = max(0, idx - 2)
                        end_idx = min(len(flat_cpu), idx + 3)
                        
                        cpu_neighbors = flat_cpu[start_idx:end_idx]
                        gpu_neighbors = flat_gpu[start_idx:end_idx]
                        neighbor_indices = list(range(start_idx, end_idx))
                        
                        print(f"      [{i}] Index {idx}: diff={deviation:.6e}, CPU={cpu_val:.6f}, GPU={gpu_val:.6f}")
                        print(f"          Neighboring values (indices {start_idx}-{end_idx-1}):")
                        print(f"          CPU: {[f'{v:.6f}' for v in cpu_neighbors]}")
                        print(f"          GPU: {[f'{v:.6f}' for v in gpu_neighbors]}")
                        print(f"          Idx: {neighbor_indices}")
                # Note: For NaN/Inf cases, sample values are not shown to avoid confusion
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Total outputs compared: {len(outputs_info)}")
    print(f"Total mismatched outputs: {len(mismatched_layers)}")
    
    # Count different types of issues
    nan_layers = [x for x in mismatched_layers if x.get('has_nan', False)]
    inf_layers = [x for x in mismatched_layers if x.get('has_inf', False)]
    numerical_diff_layers = [x for x in mismatched_layers if not x.get('has_nan', False) and not x.get('has_inf', False)]
    
    if nan_layers:
        print(f"Outputs with NaN values: {len(nan_layers)}")
    if inf_layers:
        print(f"Outputs with Inf values: {len(inf_layers)}")
    if numerical_diff_layers:
        print(f"Outputs with numerical differences: {len(numerical_diff_layers)}")
    
    # Check final output status and display overall result
    final_output_mismatch = [x for x in mismatched_layers if x.get('is_final', False)]

    print(f"\n=== Final Model Output Comparison on precision '{precision}' ===")
    if final_output_matches:
        print("🎉 Model final outputs match between CPU and GPU")
        print("✅ Comparison Result: PASS")
    else:
        print("❌ Model final outputs differ between CPU and GPU")
        print("❌ Comparison Result: FAIL")
        if final_output_mismatch:
            layer = final_output_mismatch[0]
            failure_reason = []
            if layer.get('has_nan', False):
                failure_reason.append("NaN values")
            if layer.get('has_inf', False):
                failure_reason.append("Inf values")
            if not failure_reason:
                failure_reason.append(f"numerical difference (max_diff: {layer['max_diff']:.2e})")
            
            print(f"    Final output mismatch details: {layer['name']} - {', '.join(failure_reason)}")
    
    debug_mismatches = [x for x in mismatched_layers if x['type'] != "Result" and not x.get('is_final', False)]
    if debug_mismatches:
        print("\n⚠️  Debug outputs with differences:")
        for layer in debug_mismatches:
            issues = []
            if layer.get('has_nan', False):
                issues.append("NaN")
            if layer.get('has_inf', False):
                issues.append("Inf")
            if not issues:
                issues.append(f"max_diff={layer['max_diff']:.2e}")
            
            print(f"  - {layer['name']} ({layer['type']}): {', '.join(issues)}")
    
    if not mismatched_layers:
        print("✅ All outputs match within tolerance!")
    
    return mismatched_layers

def add_all_debug_outputs(ov_model: ov.Model, percentage=100):
    """Add intermediate layers as debug outputs to the model
    
    Args:
        ov_model: OpenVINO model
        percentage: Percentage of eligible nodes to add as outputs (1-100)
                   Selects the first N% of nodes in model execution order
    
    Note:
        Skips Parameter, Constant, Result, and all Convolution-related nodes (nodes with type starting with 'conv')
    """
    print(f"\n=== Adding Debug Outputs to Model (first {percentage}% of eligible nodes) ===")
    
    # Get all operation nodes that can be used as outputs
    all_ops = ov_model.get_ordered_ops()
    eligible_ops = []
    
    # Filter eligible operations and record their original order
    for idx, op in enumerate(all_ops, 1):
        if (op.get_output_size() > 0 and op.get_type_name() not in ['Parameter', 'Constant', 'Result', 'Convolution', 'MatMul', 'GroupConvolution', 'Concat', 'Add', 'Convert', 'Slice', 'ShapeOf', 'Interpolate']):
            eligible_ops.append((op, idx))  # Store both op and its order number
    
    print(f"Found {len(eligible_ops)} eligible nodes for debug outputs")
    
    # Calculate how many nodes to add based on percentage
    if percentage <= 0 or percentage > 100:
        print(f"Warning: Invalid percentage {percentage}%. Using 100%")
        percentage = 100
    
    num_to_add = max(1, int(len(eligible_ops) * percentage / 100))
    print(f"Will add debug outputs for the first {num_to_add} nodes ({percentage}%)")
    
    # Select nodes to add (first N% of nodes)
    if percentage >= 100:
        selected_ops = eligible_ops
    else:
        # Select the first N% of nodes based on their order in the model
        selected_ops = eligible_ops[:num_to_add]
    
    # Add debug outputs for selected nodes
    added_count = 0
    for idx, (op, node_order) in enumerate(selected_ops, 1):
        try:
            output_name = f"{op.get_friendly_name()}_debug_output_node{node_order}"
            new_result = ov.opset12.result(op.output(0))
            new_result.set_friendly_name(output_name)
            new_result.output(0).set_names({output_name})
            ov_model.add_results([new_result])
            added_count += 1
            print(f"  [{idx:2d}] Added debug output for: {op.get_friendly_name()} (type: {op.get_type_name()}, model node #{node_order})")
        except Exception as e:
            print(f"  [{idx:2d}] Warning: Could not add output for {op.get_friendly_name()}: {e}")
    
    print(f"Successfully added {added_count} debug outputs to the model")
    return ov_model

def create_simple_test_input(input_shape):
    """Create simple test input data for debugging"""
    print(f"\nCreating simple test input with shape: {input_shape}")
    
    # Option 1: All ones
    input_data = np.ones(input_shape, dtype=np.float32)
    
    # Option 2: Sequential values (more likely to reveal differences)
    # total_elements = np.prod(input_shape)
    # input_data = np.arange(total_elements, dtype=np.float32).reshape(input_shape)
    # input_data = input_data / total_elements  # Normalize to [0, 1] range
    
    # Option 3: Small random values with fixed seed for reproducibility
    # np.random.seed(42)
    # input_data = np.random.uniform(0.1, 0.9, input_shape).astype(np.float32)
    
    print(f"Input data statistics: min={np.min(input_data):.6f}, max={np.max(input_data):.6f}, mean={np.mean(input_data):.6f}")
    return input_data

def load_input_data_from_file(file_path, expected_shape):
    """Load input data from a .npy file and validate its shape"""
    print(f"\nLoading input data from file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input data file '{file_path}' does not exist.")
    
    if not file_path.endswith('.npy'):
        raise ValueError(f"Input data file must be a .npy file, got: {file_path}")
    
    try:
        input_data = np.load(file_path)
        print(f"Loaded data shape: {input_data.shape}")
        print(f"Expected shape: {expected_shape}")
        
        # Convert to float32 if needed
        if input_data.dtype != np.float32:
            print(f"Converting data from {input_data.dtype} to float32")
            input_data = input_data.astype(np.float32)
        
        # Validate shape
        if input_data.shape != tuple(expected_shape):
            print(f"Warning: Loaded data shape {input_data.shape} doesn't match expected shape {expected_shape}")
            print("Attempting to reshape the data...")
            
            # Try to reshape if total elements match
            if np.prod(input_data.shape) == np.prod(expected_shape):
                input_data = input_data.reshape(expected_shape)
                print(f"Successfully reshaped data to {input_data.shape}")
            else:
                raise ValueError(f"Cannot reshape data: total elements {np.prod(input_data.shape)} != {np.prod(expected_shape)}")
        
        print(f"Input data statistics: min={np.min(input_data):.6f}, max={np.max(input_data):.6f}, mean={np.mean(input_data):.6f}")
        return input_data
        
    except Exception as e:
        raise RuntimeError(f"Failed to load input data from {file_path}: {e}")

def test():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Add new output to OpenVINO model')
    parser.add_argument('-m', '--model', type=str, help='Path to input model file (.xml)')
    parser.add_argument('-o', '--output_node', type=str, nargs='*', default=[], 
                        help='Name(s) of the node(s) to add as new output. Can specify multiple nodes separated by spaces.')
    parser.add_argument('-d', '--device', type=str, default='CPU', 
                        help='Device to run inference on (default: CPU)')
    parser.add_argument('--compare', action='store_true', 
                        help='Compare CPU vs GPU outputs for all layers')
    parser.add_argument('--debug-percentage', type=float, default=None,
                        help='Percentage of eligible nodes to add as debug outputs (1-100). Selects the first N%% of nodes in execution order. If not specified, no debug outputs will be added.')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='Tolerance for CPU vs GPU comparison (default: 1e-5)')
    parser.add_argument('--simple-input', action='store_true',
                        help='Use simple sequential input data instead of random')
    parser.add_argument('-i', '--input_data', type=str, default=None,
                        help='Path to .npy file containing input data. If specified, loads data from file instead of generating it.')
    parser.add_argument('--precision', type=str, choices=['FP32', 'FP16', 'INT8'], default='FP32',
                        help='Inference precision hint (default: FP32). Options: FP32, FP16, INT8')
    
    args = parser.parse_args()
    core = ov.Core()
    print(f"Using OpenVINO version: {ov.get_version()}")
    
    # Decide which model to use based on whether model path is provided
    if args.model:
        if not os.path.exists(args.model):
            print(f"Error: Model file '{args.model}' does not exist.")
            return

        print(f"Loading model from: {args.model}")
        model = core.read_model(args.model)
        print("Model loaded successfully!")
    else:
        print("No input model specified, using default generated model...")
        model = my_model()


    # Add the specified output nodes if provided
    if args.output_node:
        print(f"Adding specified output nodes: {args.output_node}")
        model = add_new_output(model, args.output_node)

    # Display original model information
    print("\n=== Original Model Info ===")
    common_utils.print_model_info(model)

    # If compare mode is enabled, prepare model with debug outputs and run comparison
    if args.compare:
        print("\n=== Preparing model for CPU vs GPU comparison ===")
        
        # Add debug outputs only if percentage is specified
        if args.debug_percentage is not None:
            model = add_all_debug_outputs(model, args.debug_percentage)
        else:
            print("No debug percentage specified, skipping debug output addition.")
        
        # Prepare test input
        input_shape = model.input(0).get_shape()
        print(f"\n=== Preparing test input for comparison ===")
        
        if args.input_data:
            # Load input data from file
            input_data = load_input_data_from_file(args.input_data, input_shape)
        elif args.simple_input:
            input_data = create_simple_test_input(input_shape)
        else:
            # Use simple sequential data for better reproducibility
            input_data = create_simple_test_input(input_shape)
        
        # Run comparison
        mismatched_layers = compare_cpu_gpu_outputs(model, input_data, args.tolerance, args.precision)
        
        # Exit after comparison
        return
    
    # Compile model
    print(f"\n=== Compiling model for device: {args.device} with precision: {args.precision} ===")
    cm = ov.compile_model(model, args.device, {"INFERENCE_PRECISION_HINT": args.precision})

    # Prepare input data
    input_shape = model.input(0).get_shape()
    print(f"\n=== Preparing input data with shape: {input_shape}")
    
    if args.input_data:
        # Load input data from file
        input_data = load_input_data_from_file(args.input_data, input_shape)
    elif args.simple_input:
        input_data = create_simple_test_input(input_shape)
    else:
        input_data = np.random.uniform(low=0, high=255, size=input_shape).astype(np.float32)

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