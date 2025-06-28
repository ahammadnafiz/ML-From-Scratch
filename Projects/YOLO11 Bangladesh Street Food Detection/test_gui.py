#!/usr/bin/env python3
"""
Test script for YOLOv12 Bangladesh Street Food Detection GUI
============================================================

This script tests that the GUI application can be imported and initialized
without errors. It validates all dependencies are properly installed.
"""

import sys
import os

def test_imports():
    """Test that all required imports work"""
    print("🔍 Testing imports...")
    
    try:
        import customtkinter as ctk
        print("✅ CustomTkinter imported successfully")
    except ImportError:
        print("❌ CustomTkinter import failed")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError:
        print("❌ OpenCV import failed")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
    except ImportError:
        print("❌ Ultralytics import failed")
        return False
    
    try:
        from PIL import Image, ImageTk
        print("✅ PIL/Pillow imported successfully")
    except ImportError:
        print("❌ PIL/Pillow import failed")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError:
        print("❌ NumPy import failed")
        return False
    
    return True

def test_app_class():
    """Test that the main app class can be instantiated"""
    print("\n🏗️ Testing app class instantiation...")
    
    try:
        # Import the main app class
        from app import YOLOv12BangladeshFoodDetector
        print("✅ App class imported successfully")
        
        # Test that we can create the class (but don't run mainloop)
        app = YOLOv12BangladeshFoodDetector()
        print("✅ App class instantiated successfully")
        print(f"✅ Classes configured: {app.CLASSES}")
        print(f"✅ Default confidence threshold: {app.CONFIDENCE_THRESHOLD}")
        
        # Clean up
        app.root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ App class test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🍛 YOLOv12 Bangladesh Street Food Detection - GUI Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing packages:")
        print("pip install customtkinter opencv-python ultralytics Pillow numpy")
        sys.exit(1)
    
    # Test app class
    if not test_app_class():
        print("\n❌ App class tests failed.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! The GUI application is ready to run.")
    print("\nTo start the application:")
    print("python app.py")
    print("\n📋 Usage:")
    print("1. Load a YOLOv12 model (auto-loads if available)")
    print("2. Start camera for real-time detection")
    print("3. Adjust confidence/IoU thresholds as needed")
    print("4. Monitor detection statistics")

if __name__ == "__main__":
    main()
