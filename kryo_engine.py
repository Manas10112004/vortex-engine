import os
import ctypes
import numpy as np
from ctypes import c_char_p, c_void_p, c_bool, POINTER, c_uint8, c_float, c_int

class KryoEngine:
    def __init__(self, model_filename: str):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.project_dir, "models", model_filename)
        self.dll_path = os.path.join(self.project_dir, "core", "kryo_core.dll")
        self.core_ptr = None
        
        if not os.path.exists(self.dll_path):
            raise FileNotFoundError(f"Missing core binaries at {self.dll_path}")

        os.add_dll_directory(os.path.dirname(self.dll_path))
        self.lib = ctypes.CDLL(self.dll_path, winmode=0)
        
        # Link function typings
        self.lib.create_kryo_instance.argtypes = [c_char_p]
        self.lib.create_kryo_instance.restype = c_void_p
        
        self.lib.execute_core_mapping.argtypes = [c_void_p]
        self.lib.execute_core_mapping.restype = c_bool
        
        self.lib.execute_4bit_unpack.argtypes = [c_void_p, POINTER(c_uint8), POINTER(c_float), c_int, c_float]
        self.lib.execute_4bit_unpack.restype = None
        
        self.lib.free_kryo_instance.argtypes = [c_void_p]
        self.lib.free_kryo_instance.restype = None

    def initialize_system(self):
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, "wb") as f:
                f.write(b"\x42" * 1024 * 1024 * 25) # 25MB sample file
                
        self.core_ptr = self.lib.create_kryo_instance(self.model_path.encode('utf-8'))
        return self.lib.execute_core_mapping(self.core_ptr)

    def process_quantized_weights(self, packed_bytes: bytes, scale: float):
        """Passes a pointer to a compressed python byte array down to raw C++ bits loops"""
        packed_len = len(packed_bytes)
        output_len = packed_len * 2
        
        # Create continuous fast buffers via numpy layouts
        np_input = np.frombuffer(packed_bytes, dtype=np.uint8)
        np_output = np.zeros(output_len, dtype=np.float32)
        
        # Pull underlying pointers
        input_ptr = np_input.ctypes.data_as(POINTER(c_uint8))
        output_ptr = np_output.ctypes.data_as(POINTER(c_float))
        
        # Call naked C++ engine execution loop
        self.lib.execute_4bit_unpack(self.core_ptr, input_ptr, output_ptr, packed_len, scale)
        return np_output

    def terminate_system(self):
        if self.core_ptr:
            self.lib.free_kryo_instance(self.core_ptr)
            self.core_ptr = None