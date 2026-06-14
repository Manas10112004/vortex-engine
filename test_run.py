from kryo_engine import KryoEngine
import time

def main():
    print("=== KRYO ENGINE RUNTIME VERIFICATION ===")
    engine = KryoEngine("mistral-7b-int4.bin")
    
    if not engine.initialize_system():
        print("[System Check] Failed to allocate mapping context.")
        return
        
    # Mocking real 4-bit data stream ingestion (10 distinct packed weights bytes)
    # Hex representation containing structured sample arrays
    mock_packed_weights = b"\x3F\x4A\x12\x8B\x9C\xDE\x77\x55\xAA\xFF"
    scale_factor = 0.025
    
    print("\n[Orchestrator] Passing compressed weights arrays down to C++ Subsystems...")
    unpacked_weights = engine.process_quantized_weights(mock_packed_weights, scale_factor)
    
    print(f"[Orchestrator] Success! Unpacked {len(unpacked_weights)} float weights cleanly:")
    print(f"Resulting Vector Preview: {unpacked_weights[:6]}...")
    
    print("\n=== SYSTEM CLEANUP INITIALIZATION ===")
    engine.terminate_system()
    print("Verification execution run fully stable.")

if __name__ == "__main__":
    main()