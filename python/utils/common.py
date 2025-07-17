import openvino as ov
from openvino import opset8 as opset
from openvino import Core, Model, Type, Shape, op


def print_model_info(ov_model:ov.Model):
    print(f"== Model name: {ov_model.get_friendly_name()}")
    print(f"== Model inputs info: {[(input.get_node().get_friendly_name(), input.get_shape()) for input in ov_model.inputs]}")
    print(f"== Model outputs info: {[(output.get_node().get_friendly_name(), output.get_shape()) for output in ov_model.outputs]}")
    
    for idx, op in enumerate(ov_model.get_ordered_ops(), 1):
        try:
            input_shape = op.input(0).get_shape() if op.get_input_size() > 0 else "No input"
            output_shape = op.output(0).get_shape() if op.get_output_size() > 0 else "No output"
            print(f"==\t [{idx:3d}] Found op: '{op.get_friendly_name()}' with type '{op.get_type_name()}'. Input shape: {input_shape}, Output shape: {output_shape}")
        except Exception as e:
            print(f"==\t [{idx:3d}] Found op: '{op.get_friendly_name()}' with type '{op.get_type_name()}'. Error getting shapes: {e}")
    print(f"== Total number of operations: {len(ov_model.get_ordered_ops())}")