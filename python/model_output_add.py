import openvino as ov
import numpy as np
from openvino import opset8 as opset
from openvino import Core, Model, Type, Shape, op
import utils.common as common_utils
import argparse
import os
import time
import sys


def get_timestamp():
    """Generate timestamp string for file naming"""
    return time.strftime("%Y%m%d_%H%M%S")


def create_simple_model():
    input = opset.parameter([1, 256, 32, 32], Type.f32, name="input")

    weight_arr = np.full([1024, 256, 1, 1], 1.5, dtype=np.float32)
    weight = opset.constant(weight_arr, Type.f32, name="weight")

    strides = [1, 1]
    pads_begin = [1, 1]
    pads_end = [1, 1]
    dilations = [1, 1]
    conv = opset.convolution(input, weight, strides, pads_begin, pads_end, dilations)

    add = opset.add(conv, np.full([1, 1024, 1, 1], 1, dtype=np.float32), name="op_add")

    op_gelu = opset.gelu(add, approximation_mode="ERF")

    Result = opset.result(op_gelu, name="output")
    Result.output(0).set_names({"output"})
    return Model([Result], [input], "model_add")


def add_debug_output_by_name(ov_model: ov.Model, name_list):
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
                new_result.set_friendly_name(new_output_name)  # Set a friendly name
                new_result.output(0).set_names({new_output_name})  # Set tensor names

                # Add the new result node to the model's outputs
                ov_model.add_results([new_result])
                print(f"== Added new output: {new_output_name}")
                added_count += 1
            except Exception as e:
                print(f"== Error: Failed to add output for '{name}': {e}")
        else:
            print(f"== Error: Could not find node '{name}' to add as a new output.")

    if added_count == 0:
        print(
            f"Available nodes: {available_nodes[:10]}..."
        )  # Show first 10 available nodes
        if len(available_nodes) > 10:
            print(f"... and {len(available_nodes) - 10} more nodes")
    else:
        print(
            f"Successfully added {added_count} new outputs from {len(name_list)} requested nodes."
        )

    return ov_model


def compare_cpu_gpu_outputs(
    ov_model: ov.Model, input_data, tolerance=1e-5, precision="FP32", need_dump=False
):
    """Compare outputs of all layers between CPU and GPU devices"""
    print("\n=== Comparing CPU vs GPU outputs for all layers ===")
    print(f"Using precision: {precision}")

    if need_dump:
        # Create output directory for saving data
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"debug_outputs_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Collect all model outputs for comparison
    outputs_info = []
    print(f"=== Model Outputs ===")
    for i, output in enumerate(ov_model.outputs):
        output_name = (
            list(output.get_names())[0] if output.get_names() else output.get_any_name()
        )
        source_op = output.get_node()
        outputs_info.append(
            (source_op.get_friendly_name(), output_name, source_op.get_type_name())
        )
        print(
            f"  Output {i}: {output_name} (from {source_op.get_friendly_name()}, type: {source_op.get_type_name()})"
        )

    # Move the first element to the last position
    if len(outputs_info) > 1:
        first_element = outputs_info.pop(0)
        outputs_info.append(first_element)
        print(
            f"\nReordered outputs: moved first output to last position for final model output comparison"
        )

    print(f"\nTotal outputs for comparison: {len(outputs_info)}")

    # Compile for CPU and GPU
    print("\nCompiling models...")
    try:
        cm_cpu = ov.compile_model(
            ov_model, "CPU", {"INFERENCE_PRECISION_HINT": precision}
        )
        print("✓ CPU compilation successful")
    except Exception as e:
        print(f"✗ CPU compilation failed: {e}")
        return

    try:
        cm_gpu = ov.compile_model(
            ov_model, "GPU", {"INFERENCE_PRECISION_HINT": precision}
        )
        print("✓ GPU compilation successful")
    except Exception as e:
        print(f"✗ GPU compilation failed: {e}")
        print("Note: GPU might not be available on this system")
        return

    # Run inference on both devices
    print("\nRunning inference on both devices...")
    
    # Measure CPU inference time
    print("Running CPU inference...")
    cpu_start_time = time.time()
    cpu_outputs = cm_cpu(input_data)
    cpu_end_time = time.time()
    cpu_inference_time = cpu_end_time - cpu_start_time
    
    # Measure GPU inference time
    print("Running GPU inference...")
    gpu_start_time = time.time()
    gpu_outputs = cm_gpu(input_data)
    gpu_end_time = time.time()
    gpu_inference_time = gpu_end_time - gpu_start_time
    
    # Display performance information
    print(f"\n=== Performance Information ===")
    print(f"CPU inference time: {cpu_inference_time:.4f} seconds ({cpu_inference_time*1000:.2f} ms)")
    print(f"GPU inference time: {gpu_inference_time:.4f} seconds ({gpu_inference_time*1000:.2f} ms)")
    
    if gpu_inference_time > 0:
        speedup_ratio = cpu_inference_time / gpu_inference_time
        if speedup_ratio > 1:
            print(f"GPU is {speedup_ratio:.2f}x faster than CPU")
        else:
            print(f"CPU is {1/speedup_ratio:.2f}x faster than GPU")
    
    print(f"Inference time difference: {abs(cpu_inference_time - gpu_inference_time):.4f} seconds")

    print(f"\n=== Debug: Expected output names from outputs_info ===")
    for layer_name, output_name, layer_type in outputs_info:
        print(f"  Expected: '{output_name}' (from {layer_name}, type: {layer_type})")

    # Compare outputs
    print(f"\n=== Output Comparison (tolerance: {tolerance}) ===")
    mismatched_layers = []
    final_output_matches = True  # Track if the final output matches

    for idx, (layer_name, output_name, layer_type) in enumerate(outputs_info, 1):
        is_final_output = idx == len(outputs_info)  # Check if this is the final output

        # Extract node order from output name if it's a debug output
        node_order_info = ""
        if "_debug_output_node" in output_name:
            try:
                node_num = output_name.split("_debug_output_node")[1]
                node_order_info = f" (Model Node #{node_num})"
            except:
                pass

        if is_final_output:
            print(
                f"\n =====   Model output name: {output_name}{node_order_info} ======"
            )
        else:
            print(
                f"\n =====   Layer output name: {output_name}{node_order_info} ======"
            )
        if output_name in cpu_outputs and output_name in gpu_outputs:
            cpu_out = cpu_outputs[output_name]
            gpu_out = gpu_outputs[output_name]

            # Check for NaN or Inf values first
            cpu_has_nan = np.isnan(cpu_out).any()
            cpu_has_inf = np.isinf(cpu_out).any()
            gpu_has_nan = np.isnan(gpu_out).any()
            gpu_has_inf = np.isinf(gpu_out).any()

            # Calculate min/max values for CPU and GPU outputs
            cpu_min = (
                np.min(cpu_out)
                if not (cpu_has_nan or cpu_has_inf)
                else (
                    np.min(cpu_out[np.isfinite(cpu_out)])
                    if np.any(np.isfinite(cpu_out))
                    else float("nan")
                )
            )
            cpu_max = (
                np.max(cpu_out)
                if not (cpu_has_nan or cpu_has_inf)
                else (
                    np.max(cpu_out[np.isfinite(cpu_out)])
                    if np.any(np.isfinite(cpu_out))
                    else float("nan")
                )
            )
            gpu_min = (
                np.min(gpu_out)
                if not (gpu_has_nan or gpu_has_inf)
                else (
                    np.min(gpu_out[np.isfinite(gpu_out)])
                    if np.any(np.isfinite(gpu_out))
                    else float("nan")
                )
            )
            gpu_max = (
                np.max(gpu_out)
                if not (gpu_has_nan or gpu_has_inf)
                else (
                    np.max(gpu_out[np.isfinite(gpu_out)])
                    if np.any(np.isfinite(gpu_out))
                    else float("nan")
                )
            )

            # Calculate difference (handle NaN case)
            diff = np.abs(cpu_out - gpu_out)
            max_diff = np.max(diff) if not np.isnan(diff).any() else float("inf")
            mean_diff = np.mean(diff) if not np.isnan(diff).any() else float("inf")

            # Check if difference exceeds tolerance or if there are NaN/Inf values
            has_invalid_values = (
                cpu_has_nan or cpu_has_inf or gpu_has_nan or gpu_has_inf
            )
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
            warning_str = (
                f" [{'/'.join(warning_indicators)}]" if warning_indicators else ""
            )

            print(
                f"[{idx:3d}] {status} {layer_name:20s} ({layer_type:15s}){node_order_info} | Max diff: {max_diff:.2e} | Mean diff: {mean_diff:.2e}{warning_str}"
            )
            print(
                f"      CPU: min={cpu_min:.6e}, max={cpu_max:.6e} | GPU: min={gpu_min:.6e}, max={gpu_max:.6e}"
            )

            # Save output data to files for important nodes (final output, mismatched, or nodes with invalid values)
            should_save = is_final_output or is_mismatch or has_invalid_values
            if should_save and need_dump:
                try:
                    # Create safe filename from output name
                    safe_name = "".join(
                        c for c in output_name if c.isalnum() or c in ("_", "-")
                    ).rstrip()
                    if not safe_name:
                        safe_name = f"output_{idx}"

                    # Add shape information to filename
                    shape_str = "x".join(map(str, cpu_out.shape))
                    safe_name_with_shape = f"{safe_name}_shape_{shape_str}"

                    # Save CPU output
                    cpu_file = os.path.join(
                        output_dir, f"{safe_name_with_shape}_cpu.npy"
                    )
                    np.save(cpu_file, cpu_out)

                    # Save GPU output
                    gpu_file = os.path.join(
                        output_dir, f"{safe_name_with_shape}_gpu.npy"
                    )
                    np.save(gpu_file, gpu_out)

                    print(
                        f"      💾 Saved outputs to: {safe_name_with_shape}_{{cpu,gpu,diff}}.npy"
                    )

                except Exception as e:
                    print(f"      ⚠️  Failed to save output data: {e}")

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
                    if (
                        not gpu_has_nan
                    ):  # Only show GPU values if they don't contain NaN
                        print(f"        GPU sample values: {gpu_out.flatten()[:10]}")
                    else:
                        # When GPU has NaN, show CPU values at NaN positions
                        flat_cpu = cpu_out.flatten()
                        flat_gpu = gpu_out.flatten()
                        nan_indices = np.where(np.isnan(flat_gpu))[0]

                        if len(nan_indices) > 0:
                            # Show first 10 NaN positions and corresponding CPU values
                            show_count = min(10, len(nan_indices))
                            print(
                                f"        GPU NaN positions and corresponding CPU values:"
                            )
                            for i in range(show_count):
                                idx = nan_indices[i]
                                cpu_val = flat_cpu[idx]
                                print(
                                    f"          Index {idx}: GPU=nan, CPU={cpu_val:.6f}"
                                )

                            if len(nan_indices) > 10:
                                print(
                                    f"          ... and {len(nan_indices) - 10} more NaN positions"
                                )
                elif cpu_has_inf or gpu_has_inf:
                    # For Inf-only cases, show both CPU and GPU sample values
                    print(f"        CPU sample values: {cpu_out.flatten()[:10]}")
                    print(f"        GPU sample values: {gpu_out.flatten()[:10]}")

            # Track mismatches
            if is_mismatch:
                mismatch_info = {
                    "name": layer_name,
                    "output_name": output_name,
                    "type": layer_type,
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "cpu_shape": cpu_out.shape,
                    "gpu_shape": gpu_out.shape,
                    "is_final": is_final_output,
                    "node_order": node_order_info,
                    "has_nan": cpu_has_nan or gpu_has_nan,
                    "has_inf": cpu_has_inf or gpu_has_inf,
                }
                mismatched_layers.append(mismatch_info)

                # Only show detailed analysis if values are not NaN/Inf
                if not has_invalid_values:
                    # Find and show the location and values with largest deviations
                    flat_cpu = cpu_out.flatten()
                    flat_gpu = gpu_out.flatten()
                    flat_diff = np.abs(flat_cpu - flat_gpu)

                    # Find indices of top 3 largest differences
                    top_diff_indices = np.argsort(flat_diff)[-3:][
                        ::-1
                    ]  # Get top 3, reverse to largest first

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

                        print(
                            f"      [{i}] Index {idx}: diff={deviation:.6e}, CPU={cpu_val:.6f}, GPU={gpu_val:.6f}"
                        )
                        print(
                            f"          Neighboring values (indices {start_idx}-{end_idx-1}):"
                        )
                        print(f"          CPU: {[f'{v:.6f}' for v in cpu_neighbors]}")
                        print(f"          GPU: {[f'{v:.6f}' for v in gpu_neighbors]}")
                        print(f"          Idx: {neighbor_indices}")
                # Note: For NaN/Inf cases, sample values are not shown to avoid confusion

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total outputs compared: {len(outputs_info)}")
    print(f"Total mismatched outputs: {len(mismatched_layers)}")
    
    # Performance summary
    print(f"\n=== Performance Summary ===")
    print(f"CPU inference time: {cpu_inference_time:.4f}s ({cpu_inference_time*1000:.2f}ms)")
    print(f"GPU inference time: {gpu_inference_time:.4f}s ({gpu_inference_time*1000:.2f}ms)")

    # Count different types of issues
    nan_layers = [x for x in mismatched_layers if x.get("has_nan", False)]
    inf_layers = [x for x in mismatched_layers if x.get("has_inf", False)]
    numerical_diff_layers = [
        x
        for x in mismatched_layers
        if not x.get("has_nan", False) and not x.get("has_inf", False)
    ]

    if nan_layers:
        print(f"Outputs with NaN values: {len(nan_layers)}")
    if inf_layers:
        print(f"Outputs with Inf values: {len(inf_layers)}")
    if numerical_diff_layers:
        print(f"Outputs with numerical differences: {len(numerical_diff_layers)}")

    # Check final output status and display overall result
    final_output_mismatch = [x for x in mismatched_layers if x.get("is_final", False)]

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
            if layer.get("has_nan", False):
                failure_reason.append("NaN values")
            if layer.get("has_inf", False):
                failure_reason.append("Inf values")
            if not failure_reason:
                failure_reason.append(
                    f"numerical difference (max_diff: {layer['max_diff']:.2e})"
                )

            print(
                f"    Final output mismatch details: {layer['name']} - {', '.join(failure_reason)}"
            )

    debug_mismatches = [
        x
        for x in mismatched_layers
        if x["type"] != "Result" and not x.get("is_final", False)
    ]
    if debug_mismatches:
        print("\n⚠️  Debug outputs with differences:")
        for layer in debug_mismatches:
            issues = []
            if layer.get("has_nan", False):
                issues.append("NaN")
            if layer.get("has_inf", False):
                issues.append("Inf")
            if not issues:
                issues.append(f"max_diff={layer['max_diff']:.2e}")

            print(f"  - {layer['name']} ({layer['type']}): {', '.join(issues)}")

    if not mismatched_layers:
        print("✅ All outputs match within tolerance!")

    if need_dump:
        # Print summary of saved files
        try:
            saved_files = [f for f in os.listdir(output_dir) if f.endswith(".npy")]
            if saved_files:
                print(f"\n📁 Output Data Files Summary:")
                print(f"   Directory: {output_dir}")
                print(f"   Total files saved: {len(saved_files)}")

                # Group files by node
                node_groups = {}
                for filename in saved_files:
                    base_name = filename.rsplit("_", 1)[
                        0
                    ]  # Remove _cpu/_gpu/_diff suffix
                    if base_name not in node_groups:
                        node_groups[base_name] = []
                    node_groups[base_name].append(filename)

                print(f"   Nodes with saved outputs: {len(node_groups)}")
                for node_name, files in sorted(node_groups.items()):
                    print(f"     - {node_name}: {', '.join(sorted(files))}")
            else:
                print(
                    f"\n📁 No output files were saved (all outputs matched within tolerance)"
                )
        except Exception as e:
            print(f"\n⚠️  Error accessing output directory: {e}")

    return mismatched_layers


def add_debug_outputs_by_percentage(ov_model: ov.Model, percentage=100):
    """Add intermediate layers as debug outputs to the model

    Args:
        ov_model: OpenVINO model
        percentage: Percentage of eligible nodes to add as outputs (1-100)
                   Selects the first N% of nodes in model execution order

    Note:
        Skips Parameter, Constant, Result, and all Convolution-related nodes (nodes with type starting with 'conv')
    """
    print(
        f"\n=== Adding Debug Outputs to Model (first {percentage}% of eligible nodes) ==="
    )

    # Get all operation nodes that can be used as outputs
    all_ops = ov_model.get_ordered_ops()
    eligible_ops = []

    # Filter eligible operations and record their original order
    for idx, op in enumerate(all_ops, 1):
        if op.get_output_size() > 0 and op.get_type_name() not in [
            "Parameter",
            "Constant",
            "Result",
            "Concat",
            "Add",
            "Convert",
            "Slice",
            "ShapeOf",
            "Interpolate",
        ]:
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
            print(
                f"  [{idx:2d}] Added debug output for: {op.get_friendly_name()} (type: {op.get_type_name()}, model node #{node_order})"
            )
        except Exception as e:
            print(
                f"  [{idx:2d}] Warning: Could not add output for {op.get_friendly_name()}: {e}"
            )

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

    print(
        f"Input data statistics: min={np.min(input_data):.6f}, max={np.max(input_data):.6f}, mean={np.mean(input_data):.6f}"
    )
    return input_data


def load_input_data_from_file(file_path, expected_shape):
    """Load input data from a .npy file and validate its shape"""
    print(f"\nLoading input data from file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input data file '{file_path}' does not exist.")

    if not file_path.endswith(".npy"):
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
            print(
                f"Warning: Loaded data shape {input_data.shape} doesn't match expected shape {expected_shape}"
            )
            print("Attempting to reshape the data...")

            # Try to reshape if total elements match
            if np.prod(input_data.shape) == np.prod(expected_shape):
                input_data = input_data.reshape(expected_shape)
                print(f"Successfully reshaped data to {input_data.shape}")
            else:
                raise ValueError(
                    f"Cannot reshape data: total elements {np.prod(input_data.shape)} != {np.prod(expected_shape)}"
                )

        print(
            f"Input data statistics: min={np.min(input_data):.6f}, max={np.max(input_data):.6f}, mean={np.mean(input_data):.6f}"
        )
        return input_data

    except Exception as e:
        raise RuntimeError(f"Failed to load input data from {file_path}: {e}")


def inspect_node_by_number(ov_model: ov.Model, node_number):
    """
    Inspect a specific node by its number and return information about its inputs and outputs.
    If the node number is out of range, return the original model unchanged.

    Args:
        ov_model: OpenVINO model
        node_number: The sequential number of the node to inspect (1-based indexing)

    Returns:
        tuple: (model, node_info) where:
            - model: The original model (unchanged)
            - node_info: Dictionary containing node information, or None if node not found
    """
    print(f"\n=== Inspecting Node #{node_number} ===")

    # Get all operation nodes
    all_ops = ov_model.get_ordered_ops()

    # Check if node number is valid (1-based indexing)
    if node_number < 1 or node_number > len(all_ops):
        print(
            f"Error: Node number {node_number} is out of range. Valid range: 1-{len(all_ops)}"
        )
        return ov_model, None

    # Get the target node (convert to 0-based indexing)
    target_node = all_ops[node_number - 1]

    print(
        f"Found Node #{node_number}: '{target_node.get_friendly_name()}' (type: {target_node.get_type_name()})"
    )

    # Collect node information
    node_info = {
        "node_number": node_number,
        "friendly_name": target_node.get_friendly_name(),
        "type_name": target_node.get_type_name(),
        "inputs": [],
        "outputs": [],
    }

    # Analyze inputs
    print(f"\n--- Node #{node_number} Inputs ---")
    if target_node.get_input_size() == 0:
        print("  No inputs (this is likely a Parameter or Constant node)")
        node_info["inputs"] = []
    else:
        for i in range(target_node.get_input_size()):
            input_port = target_node.input(i)
            source_output = input_port.get_source_output()
            source_node = source_output.get_node()

            input_info = {
                "input_index": i,
                "shape": list(input_port.get_shape()),
                "element_type": str(input_port.get_element_type()),
                "source_node_name": source_node.get_friendly_name(),
                "source_node_type": source_node.get_type_name(),
                "source_output_index": source_output.get_index(),
            }

            node_info["inputs"].append(input_info)

            print(
                f"  Input {i}: shape={input_info['shape']}, type={input_info['element_type']}"
            )
            print(
                f"    ↳ Source: {input_info['source_node_name']} ({input_info['source_node_type']}) output[{input_info['source_output_index']}]"
            )

    # Analyze outputs
    print(f"\n--- Node #{node_number} Outputs ---")
    if target_node.get_output_size() == 0:
        print("  No outputs (this should not happen for valid nodes)")
        node_info["outputs"] = []
    else:
        for i in range(target_node.get_output_size()):
            output_port = target_node.output(i)

            output_info = {
                "output_index": i,
                "shape": list(output_port.get_shape()),
                "element_type": str(output_port.get_element_type()),
                "target_inputs": [],
            }

            # Find where this output is consumed
            for target_input in output_port.get_target_inputs():
                target_node_consumer = target_input.get_node()
                consumer_info = {
                    "consumer_node_name": target_node_consumer.get_friendly_name(),
                    "consumer_node_type": target_node_consumer.get_type_name(),
                    "consumer_input_index": target_input.get_index(),
                }
                output_info["target_inputs"].append(consumer_info)

            node_info["outputs"].append(output_info)

            print(
                f"  Output {i}: shape={output_info['shape']}, type={output_info['element_type']}"
            )

            if output_info["target_inputs"]:
                print(f"    ↳ Consumed by:")
                for consumer in output_info["target_inputs"]:
                    print(
                        f"      - {consumer['consumer_node_name']} ({consumer['consumer_node_type']}) input[{consumer['consumer_input_index']}]"
                    )
            else:
                print(f"    ↳ Not consumed by any node (might be a model output)")

    print(f"\n=== Node #{node_number} Inspection Complete ===")

    return ov_model, node_info


def add_debug_outputs_with_dependencies_by_number(
    ov_model: ov.Model, node_numbers, dependency_depth=None
):
    """
    Add debug outputs for specified nodes and all their input dependency nodes.

    Args:
        ov_model: OpenVINO model
        node_numbers: List of node numbers (1-based indexing) or single integer
        dependency_depth: Maximum depth of dependency collection. None for unlimited depth.
                         - depth=0: Only direct input nodes (1 layer)
                         - depth=1: Direct inputs + their inputs (2 layers)
                         - depth=N: N+1 layers of dependencies

    Returns:
        Modified model with debug outputs added for all specified nodes and their input dependencies
    """
    # Convert single integer to list for uniform processing
    if isinstance(node_numbers, int):
        node_numbers = [node_numbers]

    if not node_numbers:
        print("No node numbers provided, skipping debug output addition.")
        return ov_model

    print(f"Target node numbers: {node_numbers}")

    # Get all operation nodes
    all_ops = ov_model.get_ordered_ops()
    total_nodes = len(all_ops)

    # Validate all node numbers first
    invalid_nodes = [num for num in node_numbers if num < 1 or num > total_nodes]
    if invalid_nodes:
        print(
            f"Error: Invalid node numbers {invalid_nodes}. Valid range: 1-{total_nodes}"
        )
        return ov_model

    # Collect all nodes that need debug outputs (target nodes + their dependencies)
    nodes_to_debug = set()

    # Process each target node
    for node_num in node_numbers:
        target_node = all_ops[node_num - 1]  # Convert to 0-based indexing

        print(
            f"\nProcessing Node #{node_num}: '{target_node.get_friendly_name()}' (type: {target_node.get_type_name()})"
        )

        # Add the target node itself
        nodes_to_debug.add(node_num)
        print(f"  Added target node #{node_num} to debug list")

        # Recursively collect all input dependency nodes
        dependency_nodes = set()
        _collect_input_dependencies(
            target_node, all_ops, dependency_nodes, dependency_depth
        )

        if dependency_nodes:
            depth_desc = (
                f" (max depth: {dependency_depth})"
                if dependency_depth is not None
                else " (unlimited depth)"
            )
            print(
                f"  Found {len(dependency_nodes)} input dependency nodes{depth_desc}:"
            )
            for dep_num in sorted(dependency_nodes):
                dep_node = all_ops[dep_num - 1]
                print(
                    f"    Node #{dep_num}: '{dep_node.get_friendly_name()}' ({dep_node.get_type_name()})"
                )
                nodes_to_debug.add(dep_num)
        else:
            print(f"  No input dependency nodes found")

    # Remove nodes that shouldn't have debug outputs
    eligible_nodes = set()
    skipped_nodes = []

    for node_num in nodes_to_debug:
        node = all_ops[node_num - 1]

        # Skip certain node types that typically don't need debug outputs
        if node.get_type_name() in ["Parameter", "Constant", "Result"]:
            skipped_nodes.append(
                (node_num, node.get_friendly_name(), node.get_type_name())
            )
            continue

        # Check if node has outputs
        if node.get_output_size() == 0:
            skipped_nodes.append((node_num, node.get_friendly_name(), "No outputs"))
            continue

        eligible_nodes.add(node_num)

    if skipped_nodes:
        print(f"\nSkipped {len(skipped_nodes)} nodes (not eligible for debug outputs):")
        for node_num, name, reason in skipped_nodes:
            print(f"  Node #{node_num}: '{name}' - {reason}")

    print(f"\nAdding debug outputs for {len(eligible_nodes)} eligible nodes...")

    # Add debug outputs for all eligible nodes
    added_count = 0
    failed_count = 0

    for node_num in sorted(eligible_nodes):
        node = all_ops[node_num - 1]

        try:
            # Create debug output name with node number for easier identification
            output_name = f"{node.get_friendly_name()}_debug_output_node{node_num}"

            # Create new Result node
            new_result = ov.opset12.result(node.output(0))
            new_result.set_friendly_name(output_name)
            new_result.output(0).set_names({output_name})

            # Add to model
            ov_model.add_results([new_result])
            added_count += 1

            print(
                f"  ✓ Added debug output for Node #{node_num}: '{node.get_friendly_name()}' ({node.get_type_name()})"
            )

        except Exception as e:
            failed_count += 1
            print(
                f"  ✗ Failed to add debug output for Node #{node_num}: '{node.get_friendly_name()}' - {e}"
            )

    print(f"\n=== Debug Output Addition Summary ===")
    print(f"Target nodes requested: {len(node_numbers)}")
    print(f"Total dependency nodes found: {len(nodes_to_debug) - len(node_numbers)}")
    print(f"Eligible nodes for debug outputs: {len(eligible_nodes)}")
    print(f"Successfully added debug outputs: {added_count}")
    print(f"Failed to add debug outputs: {failed_count}")
    print(f"Skipped nodes: {len(skipped_nodes)}")

    return ov_model


def _collect_input_dependencies(
    node, all_ops, dependency_nodes, depth=None, visited=None
):
    """
    Recursively collect all input dependency nodes for a given node.

    Args:
        node: The node to analyze
        all_ops: List of all operations in the model (for node number lookup)
        dependency_nodes: Set to store dependency node numbers (1-based indexing)
        depth: Maximum depth of dependency collection. None for unlimited depth.
               - depth=0: Only direct input nodes (1 layer)
               - depth=1: Direct inputs + their inputs (2 layers)
               - depth=N: N+1 layers of dependencies
        visited: Set of already visited nodes to avoid infinite loops
    """
    if visited is None:
        visited = set()

    # If depth is 0, we've reached the maximum depth, stop recursion
    if depth is not None and depth < 0:
        return

    # Avoid infinite loops
    node_id = id(node)
    if node_id in visited:
        return
    visited.add(node_id)

    # Process all input nodes
    for i in range(node.get_input_size()):
        input_port = node.input(i)
        source_output = input_port.get_source_output()
        source_node = source_output.get_node()

        # Find the node number (1-based indexing) in the ordered ops list
        source_node_number = None
        for idx, op in enumerate(all_ops, 1):
            if op is source_node:
                source_node_number = idx
                break

        if source_node_number:
            dependency_nodes.add(source_node_number)

            # Recursively collect dependencies of this input node if depth allows
            if depth is None:
                # Unlimited depth (original behavior)
                _collect_input_dependencies(
                    source_node, all_ops, dependency_nodes, depth, visited
                )
            elif depth > 0:
                # Reduce depth by 1 for next level
                _collect_input_dependencies(
                    source_node, all_ops, dependency_nodes, depth - 1, visited
                )
            # If depth == 0, we don't recurse further (only collect direct inputs)


def add_debug_output_by_number(ov_model: ov.Model, node_number):
    """
    Add debug output for a specific node by its number.

    Args:
        ov_model: OpenVINO model
        node_number: The sequential number of the node to add debug output for (1-based indexing)

    Returns:
        Modified model with debug output added, or original model if node not found/invalid
    """
    print(f"\n=== Adding Debug Output for Node #{node_number} ===")

    # Get all operation nodes
    all_ops = ov_model.get_ordered_ops()

    # Check if node number is valid (1-based indexing)
    if node_number < 1 or node_number > len(all_ops):
        print(
            f"Error: Node number {node_number} is out of range. Valid range: 1-{len(all_ops)}"
        )
        return ov_model

    # Get the target node (convert to 0-based indexing)
    target_node = all_ops[node_number - 1]

    print(
        f"Target Node #{node_number}: '{target_node.get_friendly_name()}' (type: {target_node.get_type_name()})"
    )

    # Check if node is eligible for debug output
    if target_node.get_output_size() == 0:
        print(f"Warning: Node #{node_number} has no outputs, cannot add debug output")
        return ov_model

    if target_node.get_type_name() in ["Parameter", "Constant", "Result"]:
        print(
            f"Warning: Node #{node_number} is a {target_node.get_type_name()} node, typically not used for debug outputs"
        )
        return ov_model

    try:
        # Create debug output name with node number for easier identification
        output_name = (
            f"{target_node.get_friendly_name()}_debug_output_node{node_number}"
        )

        # Create new Result node
        new_result = ov.opset12.result(target_node.output(0))
        new_result.set_friendly_name(output_name)
        new_result.output(0).set_names({output_name})

        # Add to model
        ov_model.add_results([new_result])

        print(f"✓ Successfully added debug output: {output_name}")

    except Exception as e:
        print(f"✗ Failed to add debug output for Node #{node_number}: {e}")

    return ov_model


def test_matmul_operation(
    input1_file,
    input2_file,
    device_name,
    precision,
    exp_input_shape1=(1, 32, 256, 32),
    exp_input_shape2=(1, 32, 256, 33),
):
    """
    Test MatMul operation with specified inputs and check for inf values.
    Transpose a is true and transpose b is false.

    Args:
        input1_file: Path to .npy file containing first input data
        input2_file: Path to .npy file containing second input data
        device_name: Device to run inference on ('CPU' or 'GPU')
        precision: Inference precision ('FP16' or 'FP32')
        exp_input_shape1: Expected shape for input 1. Default (1, 32, 256, 32)
        exp_input_shape2: Expected shape for input 2. Default (1, 32, 256, 33)

    Returns:
        tuple: (has_inf, output_data, inf_count, output_stats)
    """
    print(f"\n=== Testing MatMul Operation ===")
    print(f"Device: {device_name}, Precision: {precision}")
    print(f"Input 1 file: {input1_file}")
    print(f"Input 2 file: {input2_file}")

    # Load input data
    try:
        input1_data = np.load(input1_file)
        input2_data = np.load(input2_file)
        print(f"✓ Loaded input data successfully")
        print(f"  Input 1 shape: {input1_data.shape}, dtype: {input1_data.dtype}")
        print(f"  Input 2 shape: {input2_data.shape}, dtype: {input2_data.dtype}")
    except Exception as e:
        print(f"✗ Failed to load input data: {e}")
        return None, None, 0, None

    if input1_data.shape != exp_input_shape1:
        print(
            f"✗ Input 1 shape mismatch: expected {exp_input_shape1}, got {input1_data.shape}"
        )
        return None, None, 0, None

    if input2_data.shape != exp_input_shape2:
        print(
            f"✗ Input 2 shape mismatch: expected {exp_input_shape2}, got {input2_data.shape}"
        )
        return None, None, 0, None

    # Convert to float32 if needed
    if input1_data.dtype != np.float32:
        input1_data = input1_data.astype(np.float32)
        print(f"  Converted input 1 to float32")

    if input2_data.dtype != np.float32:
        input2_data = input2_data.astype(np.float32)
        print(f"  Converted input 2 to float32")

    # Check input data for inf/nan values
    input1_has_inf = np.isinf(input1_data).any()
    input1_has_nan = np.isnan(input1_data).any()
    input2_has_inf = np.isinf(input2_data).any()
    input2_has_nan = np.isnan(input2_data).any()

    print(
        f"  Input 1 statistics: min={np.min(input1_data):.6e}, max={np.max(input1_data):.6e}, mean={np.mean(input1_data):.6e}"
    )
    print(
        f"  Input 2 statistics: min={np.min(input2_data):.6e}, max={np.max(input2_data):.6e}, mean={np.mean(input2_data):.6e}"
    )

    if input1_has_inf or input1_has_nan:
        print(
            f"⚠️  Input 1 contains invalid values: inf={input1_has_inf}, nan={input1_has_nan}"
        )
    if input2_has_inf or input2_has_nan:
        print(
            f"⚠️  Input 2 contains invalid values: inf={input2_has_inf}, nan={input2_has_nan}"
        )

    # Create a simple MatMul model
    print(f"\n--- Creating MatMul Model ---")
    try:
        # Create input parameters
        input1_param = ov.opset12.parameter(
            exp_input_shape1, ov.Type.f32, name="input1"
        )
        input2_param = ov.opset12.parameter(
            exp_input_shape2, ov.Type.f32, name="input2"
        )

        # Create MatMul operation with transpose parameters
        matmul_op = ov.opset12.matmul(
            input1_param, input2_param, transpose_a=True, transpose_b=False
        )
        matmul_op.set_friendly_name("test_matmul")

        # Create result
        result = ov.opset12.result(matmul_op)
        result.set_friendly_name("matmul_output")
        result.output(0).set_names({"matmul_output"})

        # Create model
        model = ov.Model([result], [input1_param, input2_param], "test_matmul_model")
        print(f"✓ Created MatMul model successfully")

        # Calculate expected output shape
        # MatMul with transpose_a=True, transpose_b=False:
        # Input1 [1,32,256,32] transposed -> [1,32,32,256]
        # Input2 [1,32,256,33] (no transpose)
        # Result: [1,32,32,256] × [1,32,256,33] -> [1,32,32,33]
        print(f"  Expected output shape: [1,32,32,33]")
        print(f"  MatMul configuration: transpose_a=True, transpose_b=False")

    except Exception as e:
        print(f"✗ Failed to create MatMul model: {e}")
        return None, None, 0, None

    # Compile model
    print(f"\n--- Compiling Model ---")
    try:
        core = ov.Core()

        # Set precision hint
        config = {"INFERENCE_PRECISION_HINT": precision}

        # Additional GPU-specific settings if using GPU
        if device_name.upper() == "GPU":
            # You can add GPU-specific settings here if needed
            pass

        compiled_model = core.compile_model(model, device_name, config)
        print(f"✓ Model compiled successfully for {device_name} with {precision}")

    except Exception as e:
        print(f"✗ Failed to compile model for {device_name}: {e}")
        return None, None, 0, None

    # Run inference
    print(f"\n--- Running Inference ---")
    try:
        # Prepare input dictionary
        input_dict = {"input1": input1_data, "input2": input2_data}

        # Run inference
        output = compiled_model(input_dict)
        output_data = output["matmul_output"]

        print(f"✓ Inference completed successfully")
        print(f"  Output shape: {output_data.shape}")
        print(f"  Output dtype: {output_data.dtype}")

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return None, None, 0, None

    # Analyze output for inf/nan values
    print(f"\n--- Analyzing Output ---")

    # Check for inf values
    has_inf = np.isinf(output_data).any()
    inf_count = np.isinf(output_data).sum()

    # Check for nan values
    has_nan = np.isnan(output_data).any()
    nan_count = np.isnan(output_data).sum()

    # Calculate output statistics
    finite_mask = np.isfinite(output_data)
    if np.any(finite_mask):
        output_min = np.min(output_data[finite_mask])
        output_max = np.max(output_data[finite_mask])
        output_mean = np.mean(output_data[finite_mask])
    else:
        output_min = output_max = output_mean = float("nan")

    output_stats = {
        "min": output_min,
        "max": output_max,
        "mean": output_mean,
        "finite_count": np.sum(finite_mask),
        "total_count": output_data.size,
    }

    print(f"  Output statistics (finite values only):")
    print(f"    Min: {output_min:.6e}")
    print(f"    Max: {output_max:.6e}")
    print(f"    Mean: {output_mean:.6e}")
    print(
        f"    Finite values: {output_stats['finite_count']}/{output_stats['total_count']}"
    )

    # Report inf/nan status
    if has_inf:
        print(f"❌ FOUND INF VALUES: {inf_count} inf values detected in output")

        # Show some inf positions
        inf_indices = np.where(np.isinf(output_data))
        print(f"   Sample inf positions (first 5):")
        for i in range(min(5, len(inf_indices[0]))):
            pos = tuple(idx[i] for idx in inf_indices)
            print(f"     Position {pos}: {output_data[pos]}")
    else:
        print(f"✅ NO INF VALUES: Output contains no inf values")

    if has_nan:
        print(f"❌ FOUND NAN VALUES: {nan_count} nan values detected in output")

        # Show some nan positions
        nan_indices = np.where(np.isnan(output_data))
        print(f"   Sample nan positions (first 5):")
        for i in range(min(5, len(nan_indices[0]))):
            pos = tuple(idx[i] for idx in nan_indices)
            print(f"     Position {pos}: {output_data[pos]}")
    else:
        print(f"✅ NO NAN VALUES: Output contains no nan values")

    print(f"\n=== MatMul Test Summary ===")
    print(f"Device: {device_name}, Precision: {precision}")
    print(f"Inf values found: {has_inf} (count: {inf_count})")
    print(f"NaN values found: {has_nan} (count: {nan_count})")

    return has_inf or has_nan, output_data, inf_count + nan_count, output_stats


def test():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Add new output to OpenVINO model",
        epilog="""
    Examples:
        Compare CPU vs GPU outputs with debug information:
            python model_output_add.py --compare -m /mnt/ywang2/models/mobilesamv2_openvino_model/quantized_image_encoder_ov_model.xml -i /mnt/ywang2/input_image.npy --precision FP16 --debug-percentage 20
    
        Inspect a specific node:
            python model_output_add.py --inspect-node 25 -m model.xml
    
        Compare CPU vs GPU outputs with specified node:
            python model_output_add.py --compare -m /mnt/ywang2/models/mobilesamv2_openvino_model/quantized_image_encoder_ov_model.xml -i /mnt/ywang2/input_image.npy --precision FP16 --add-debug-nodes-deps 784 --dependency-depth 0 -m model.xml

        Compare with MatMul compression disabled:
            python model_output_add.py --compare -m model.xml -i input.npy --precision FP16 --disable-matmul-compression --debug-percentage 20

        Test MatMul operation:
            python model_output_add.py --test-matmul --matmul-input1 input1.npy --matmul-input1-shape [1,256,33,32] --matmul-input2 input2.npy --matmul-input2-shape [1,256,33,32] --matmul-device GPU --matmul-precision FP16
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model", type=str, help="Path to input model file (.xml)"
    )
    parser.add_argument(
        "-o",
        "--output_node",
        type=str,
        nargs="*",
        default=[],
        help="Name(s) of the node(s) to add as new output. Can specify multiple nodes separated by spaces.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="CPU",
        help="Device to run inference on (default: CPU)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare CPU vs GPU outputs for all layers",
    )
    parser.add_argument(
        "--debug-percentage",
        type=float,
        default=None,
        help="Percentage of eligible nodes to add as debug outputs (1-100). Selects the first N%% of nodes in execution order. If not specified, no debug outputs will be added.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for CPU vs GPU comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--simple-input",
        action="store_true",
        help="Use simple sequential input data instead of random",
    )
    parser.add_argument(
        "-i",
        "--input_data",
        type=str,
        default=None,
        help="Path to .npy file containing input data. If specified, loads data from file instead of generating it.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["FP32", "FP16", "INT8"],
        default="FP32",
        help="Inference precision hint (default: FP32). Options: FP32, FP16, INT8",
    )
    parser.add_argument(
        "--inspect-node",
        type=int,
        default=None,
        help="Inspect a specific node by its number (1-based indexing). Shows input/output information for the specified node.",
    )
    parser.add_argument(
        "--add-debug-node",
        type=int,
        default=None,
        help="Add debug output for a specific node by its number (1-based indexing).",
    )
    parser.add_argument(
        "--add-debug-nodes-deps",
        type=int,
        nargs="*",
        default=None,
        help="Add debug outputs for specified node numbers and all their input dependencies (1-based indexing). Can specify multiple nodes separated by spaces.",
    )
    parser.add_argument(
        "--dependency-depth",
        type=int,
        default=0,
        help="Maximum depth for dependency collection (default: 0) when using --add-debug-nodes-deps. 0=only direct inputs (1 layer), 1=2 layers, etc.",
    )
    parser.add_argument(
        "--test-matmul",
        action="store_true",
        help="Test MatMul operation with specified input files and check for inf values",
    )
    parser.add_argument(
        "--matmul-input1",
        type=str,
        default=None,
        help="Path to .npy file containing first MatMul input data with shape [1,32,256,32]",
    )
    parser.add_argument(
        "--matmul-input2",
        type=str,
        default=None,
        help="Path to .npy file containing second MatMul input data with shape [1,32,256,33]",
    )
    parser.add_argument(
        "--matmul-device",
        type=str,
        choices=["CPU", "GPU"],
        default="CPU",
        help="Device for MatMul testing (default: CPU)",
    )
    parser.add_argument(
        "--matmul-precision",
        type=str,
        choices=["FP16", "FP32"],
        default="FP32",
        help="Precision for MatMul testing (default: FP32)",
    )
    parser.add_argument(
        "--convert-model", action="store_true", help="Force to prevent overflow"
    )
    parser.add_argument(
        "--matmul-input1-shape",
        type=str,
        default="[1,32,256,32]",
        help="Expected shape for input 1",
    )
    parser.add_argument(
        "--matmul-input2-shape",
        type=str,
        default="[1,32,256,33]",
        help="Expected shape for input 2",
    )
    parser.add_argument(
        "--disable-matmul-compression",
        action="store_true",
        help="Disable compression for MatMul operations by setting rt_info['precise_0']",
    )

    args = parser.parse_args()
    core = ov.Core()
    print(f"Using OpenVINO version: {ov.get_version()}")

    # Check if MatMul test mode is requested
    if args.test_matmul:
        print(f"\n=== MatMul Test Mode ===")

        # Validate required arguments
        if not args.matmul_input1:
            print("Error: --matmul-input1 is required for MatMul testing")
            return
        if not args.matmul_input2:
            print("Error: --matmul-input2 is required for MatMul testing")
            return

        # Check if input files exist
        if not os.path.exists(args.matmul_input1):
            print(f"Error: MatMul input 1 file '{args.matmul_input1}' does not exist")
            return
        if not os.path.exists(args.matmul_input2):
            print(f"Error: MatMul input 2 file '{args.matmul_input2}' does not exist")
            return

        # Parse shape strings to tuples
        try:
            import ast

            input1_shape = tuple(ast.literal_eval(args.matmul_input1_shape))
            input2_shape = tuple(ast.literal_eval(args.matmul_input2_shape))
            print(f"Parsed input1 shape: {input1_shape}")
            print(f"Parsed input2 shape: {input2_shape}")
        except Exception as e:
            print(f"Error: Failed to parse shape strings: {e}")
            print(f"Input1 shape string: {args.matmul_input1_shape}")
            print(f"Input2 shape string: {args.matmul_input2_shape}")
            return

        # Run MatMul test
        has_invalid, output_data, invalid_count, output_stats = test_matmul_operation(
            args.matmul_input1,
            args.matmul_input2,
            args.matmul_device,
            args.matmul_precision,
            input1_shape,
            input2_shape,
        )

        # Exit after MatMul test
        if has_invalid:
            print(
                f"\n❌ MatMul test FAILED: Found {invalid_count} invalid values (inf/nan)"
            )
            sys.exit(1)
        else:
            print(f"\n✅ MatMul test PASSED: No invalid values found")
            sys.exit(0)

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
        model = create_simple_model()

    # Display original model information
    print("\n=== Original Model Info ===")
    common_utils.print_model_info(model)

    # Conditionally disable compression for MatMul operations
    if args.disable_matmul_compression:
        print(f"\n=== Disable compression for MatMul operations ===")
        matmul_count = 0
        for idx, op in enumerate(model.get_ordered_ops(), 1):
            if op.get_type_name() == "MatMul":
                # add runtime info for MatMul operations
                op.rt_info['precise_0'] = ''
                matmul_count += 1
                print(f"clear precise_0 for MatMul operation #{idx}: {op.get_friendly_name()}")
        
        if matmul_count == 0:
            print("No MatMul operations found in the model")
        else:
            print(f"Disabled compression for {matmul_count} MatMul operations")
    else:
        print(f"\n=== MatMul compression settings unchanged ===")
        matmul_count = sum(1 for op in model.get_ordered_ops() if op.get_type_name() == "MatMul")
        if matmul_count > 0:
            print(f"Found {matmul_count} MatMul operations (compression settings not modified)")
        else:
            print("No MatMul operations found in the model")

    if args.convert_model:
        from nncf import compress_weights, CompressWeightsMode
        import copy

        print("\n=== Converting model to OpenVINO IR format ===")
        # Convert model to OpenVINO IR format
        model_int4 = compress_weights(
            copy.deepcopy(model), mode=CompressWeightsMode.INT4_ASYM
        )
        ov.save_model(model_int4, "converted_model_int4.xml")
        print("Model converted successfully!")
        return
    # Check if node inspection is requested
    if args.inspect_node is not None:
        print(f"\n=== Node Inspection Mode ===")
        model, node_info = inspect_node_by_number(model, args.inspect_node)
        if node_info:
            print(f"\nNode inspection completed successfully.")
            print(
                f"You can use --add-debug-node {args.inspect_node} to add debug output for this node."
            )
        return

    # Check if debug output for specific node is requested
    if args.add_debug_node is not None:
        print(f"\n=== Adding Debug Output for Specific Node ===")
        model = add_debug_output_by_number(model, args.add_debug_node)

    # Check if debug outputs for nodes and dependencies are requested
    if args.add_debug_nodes_deps is not None:
        print(f"\n=== Adding Debug Outputs for Nodes and Dependencies ===")
        model = add_debug_outputs_with_dependencies_by_number(
            model, args.add_debug_nodes_deps, args.dependency_depth
        )

    # Add the specified output nodes if provided
    if args.output_node:
        print(f"Adding specified output nodes: {args.output_node}")
        model = add_debug_output_by_name(model, args.output_node)

    # If compare mode is enabled, prepare model with debug outputs and run comparison
    if args.compare:
        print("\n=== Preparing model for CPU vs GPU comparison ===")

        # Add debug outputs only if percentage is specified
        if args.debug_percentage is not None:
            model = add_debug_outputs_by_percentage(model, args.debug_percentage)
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
        compare_cpu_gpu_outputs(
            model,
            input_data,
            args.tolerance,
            args.precision,
            args.add_debug_node is not None or args.add_debug_nodes_deps is not None,
        )

        # Exit after comparison
        return

    # Compile model
    print(
        f"\n=== Compiling model for device: {args.device} with precision: {args.precision} ==="
    )
    cm = ov.compile_model(
        model, args.device, {"INFERENCE_PRECISION_HINT": args.precision}
    )

    # Prepare input data
    input_shape = model.input(0).get_shape()
    print(f"\n=== Preparing input data with shape: {input_shape}")

    if args.input_data:
        # Load input data from file
        input_data = load_input_data_from_file(args.input_data, input_shape)
    elif args.simple_input:
        input_data = create_simple_test_input(input_shape)
    else:
        input_data = np.random.uniform(low=0, high=255, size=input_shape).astype(
            np.float32
        )

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
        print(
            f"===\t First {num_values_to_show} values: {flat_values[:num_values_to_show]}"
        )


if __name__ == "__main__":
    test()
