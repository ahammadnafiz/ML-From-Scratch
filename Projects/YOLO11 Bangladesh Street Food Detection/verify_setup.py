#!/usr/bin/env python3
"""
Bangladesh Street Food Detection - Setup Verification Script
This script verifies that your .env configuration is correct
"""

import os
from dotenv import load_dotenv

def verify_setup():
    """Verify that all required environment variables are properly configured"""
    print("🔍 Verifying Bangladesh Street Food Detection Setup...")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Required variables
    required_vars = {
        'ROBOFLOW_API_KEY': 'Private API key for dataset access',
        'ROBOFLOW_WORKSPACE': 'Roboflow workspace name',
        'ROBOFLOW_PROJECT': 'Roboflow project name',
        'ROBOFLOW_VERSION': 'Dataset version number'
    }
    
    # Optional variables
    optional_vars = {
        'ROBOFLOW_PUBLISHABLE_KEY': 'Publishable key for client-side inference',
        'WANDB_API_KEY': 'Weights & Biases API key for experiment tracking',
        'TRAINING_EPOCHS': 'Number of training epochs',
        'BATCH_SIZE': 'Training batch size',
        'IMAGE_SIZE': 'Input image size',
        'CLASS_1': 'First selected class',
        'CLASS_2': 'Second selected class',
        'CLASS_3': 'Third selected class'
    }
    
    all_good = True
    
    print("✅ REQUIRED CONFIGURATION:")
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            if 'API_KEY' in var:
                masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
                print(f"   ✅ {var}: {masked_value} ({description})")
            else:
                print(f"   ✅ {var}: {value} ({description})")
        else:
            print(f"   ❌ {var}: NOT SET ({description})")
            all_good = False
    
    print(f"\n⚙️ OPTIONAL CONFIGURATION:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            if 'API_KEY' in var:
                masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
                print(f"   ✅ {var}: {masked_value} ({description})")
            else:
                print(f"   ✅ {var}: {value} ({description})")
        else:
            print(f"   ⚪ {var}: Using default ({description})")
    
    print("=" * 60)
    
    if all_good:
        print("🎉 SETUP VERIFICATION SUCCESSFUL!")
        print("Your configuration is ready for training.")
        print("\nNext steps:")
        print("1. Ensure you have GPU access (Google Colab Pro recommended)")
        print("2. Install required packages:")
        print("   pip install ultralytics>=8.3.0 roboflow supervision wandb python-dotenv")
        print("3. Run the Bangladesh_Street_Food_Detection.py script")
    else:
        print("❌ SETUP VERIFICATION FAILED!")
        print("Please check your .env file and ensure all required variables are set.")
    
    return all_good

def test_roboflow_connection():
    """Test connection to Roboflow with your credentials"""
    load_dotenv()
    
    try:
        from roboflow import Roboflow
        
        api_key = os.getenv('ROBOFLOW_API_KEY')
        workspace = os.getenv('ROBOFLOW_WORKSPACE')
        project = os.getenv('ROBOFLOW_PROJECT')
        
        if not all([api_key, workspace, project]):
            print("❌ Missing required Roboflow credentials")
            return False
        
        print(f"\n🔗 Testing Roboflow connection...")
        rf = Roboflow(api_key=api_key)
        rf_project = rf.workspace(workspace).project(project)
        
        print(f"✅ Successfully connected to Roboflow!")
        print(f"   Workspace: {workspace}")
        print(f"   Project: {project}")
        
        # Try to get project info
        try:
            versions = rf_project.versions()
            print(f"   Available versions: {[v.version for v in versions]}")
        except Exception as e:
            print(f"   ⚠️ Could not fetch versions: {e}")
        
        return True
        
    except ImportError:
        print("❌ Roboflow package not installed. Run: pip install roboflow")
        return False
    except Exception as e:
        print(f"❌ Roboflow connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Bangladesh Street Food Detection - Setup Verification")
    print("=" * 60)
    
    # Verify basic setup
    setup_ok = verify_setup()
    
    if setup_ok:
        # Test Roboflow connection
        test_roboflow_connection()
    
    print("\n" + "=" * 60)
    print("Setup verification complete!")
