#!/usr/bin/env python3
"""
Test script to verify Plotly installation
"""

print("Testing Plotly installation...")

try:
    import plotly
    print(f"✅ Plotly imported successfully! Version: {plotly.__version__}")
except ImportError as e:
    print(f"❌ Plotly import failed: {e}")

try:
    import plotly.express as px
    print("✅ Plotly Express imported successfully!")
except ImportError as e:
    print(f"❌ Plotly Express import failed: {e}")

try:
    import plotly.graph_objects as go
    print("✅ Plotly Graph Objects imported successfully!")
except ImportError as e:
    print(f"❌ Plotly Graph Objects import failed: {e}")

print("Test complete.")
