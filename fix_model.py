#!/usr/bin/env python3
"""
Fix model configuration to prevent verbose thinking output
"""

import subprocess
import json
import sys

def create_modelfile():
    """Create a modelfile to fix the qwen model behavior"""
    
    modelfile_content = """FROM qwen3:32b

# System message to prevent thinking output
SYSTEM You are Ultimate Local AI. Respond directly and concisely to user queries without showing any internal thinking or reasoning process. Be helpful and accurate.

# Parameters to control output
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
PARAMETER num_ctx 4096
"""

    # Write modelfile
    with open('Modelfile', 'w') as f:
        f.write(modelfile_content)
    
    print("✅ Created Modelfile")
    
    # Create the new model
    print("🔄 Creating optimized model...")
    result = subprocess.run(['ollama', 'create', 'ultimate-ai:32b', '-f', 'Modelfile'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Model created successfully!")
        print("📋 New model: ultimate-ai:32b")
    else:
        print(f"❌ Error creating model: {result.stderr}")
        
    # Clean up
    subprocess.run(['rm', 'Modelfile'])

def test_model():
    """Test the model with a simple query"""
    print("\n🧪 Testing model...")
    
    result = subprocess.run([
        'ollama', 'run', 'ultimate-ai:32b', 
        'Hello! Just say hi back without any thinking.'
    ], capture_output=True, text=True)
    
    print(f"Response: {result.stdout}")

if __name__ == "__main__":
    create_modelfile()
    test_model()
    
    print("\n✅ To use the fixed model, update your CLI to use 'ultimate-ai:32b' instead of 'qwen3:32b'")