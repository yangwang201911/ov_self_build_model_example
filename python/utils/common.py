import openvino as ov
from openvino import opset8 as opset
from openvino import Core, Model, Type, Shape, op


def print_model_info(ov_model:ov.Model):
    print(f"== Model name: {ov_model.get_friendly_name()}")
    print(f"== Model inputs info: {[(input.get_node().get_friendly_name(), input.get_shape()) for input in ov_model.inputs]}")
    print(f"== Model outputs info: {[(output.get_node().get_friendly_name(), output.get_shape()) for output in ov_model.outputs]}")
    
    for idx, op in enumerate(ov_model.get_ordered_ops(), 1):
        try:
            # Get all input shapes
            input_shapes = []
            for i in range(op.get_input_size()):
                input_shapes.append(str(op.input(i).get_shape()))
            input_info = f"{op.get_input_size()} inputs: {input_shapes}" if op.get_input_size() > 0 else "No input"
            
            # Get all output shapes
            output_shapes = []
            for i in range(op.get_output_size()):
                output_shapes.append(str(op.output(i).get_shape()))
            output_info = f"{op.get_output_size()} outputs: {output_shapes}" if op.get_output_size() > 0 else "No output"
            
            # Show all inputs and outputs if multiple exist
            if op.get_input_size() > 1 or op.get_output_size() > 1:
                print(f"==\t [{idx:3d}] Found op: '{op.get_friendly_name()}' with type '{op.get_type_name()}'. {input_info}, {output_info}")
            else:
                # For single input/output operations, use the original display for brevity
                input_shape = op.input(0).get_shape() if op.get_input_size() > 0 else "No input"
                output_shape = op.output(0).get_shape() if op.get_output_size() > 0 else "No output"
                print(f"==\t [{idx:3d}] Found op: '{op.get_friendly_name()}' with type '{op.get_type_name()}'. Input shape: {input_shape}, Output shape: {output_shape}")
        except Exception as e:
            print(f"==\t [{idx:3d}] Found op: '{op.get_friendly_name()}' with type '{op.get_type_name()}'. Error getting shapes: {e}")
    print(f"== Total number of operations: {len(ov_model.get_ordered_ops())}")