#!/usr/bin/env python3
"""
Simple test app to verify Plotly installation
"""

import streamlit as st

st.title("Plotly Test App")

# Test basic imports
st.write("Testing imports...")

try:
    import plotly
    st.success(f"✅ Plotly imported! Version: {plotly.__version__}")
except ImportError as e:
    st.error(f"❌ Plotly import failed: {e}")

try:
    import plotly.express as px
    st.success("✅ Plotly Express imported!")
except ImportError as e:
    st.error(f"❌ Plotly Express import failed: {e}")

try:
    import plotly.graph_objects as go
    st.success("✅ Plotly Graph Objects imported!")
except ImportError as e:
    st.error(f"❌ Plotly Graph Objects import failed: {e}")

# Test creating a simple plot
try:
    import plotly.express as px
    import pandas as pd
    
    # Create simple test data
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    })
    
    # Create plot
    fig = px.line(df, x='x', y='y', title='Test Plot')
    st.plotly_chart(fig)
    st.success("✅ Plot created successfully!")
    
except Exception as e:
    st.error(f"❌ Plot creation failed: {e}")

st.write("Test complete!")
