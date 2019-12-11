import streamlit as st
import numpy as np
import pandas as pd

# Sidebar config
st.sidebar.header("ML project")
st.sidebar.markdown("By Mark Shapiro")

mode = st.sidebar.selectbox("Options", ["Home", "Visualization", "Model Selection"])

if mode == "Home":
    st.title("This is my homepage")
    st.image("/home/shapim/Downloads/Shapiro_Mark.jpg", caption="Kennedy thinks I'm cute")

elif mode == "Visualization":
    st.title("Provided by matplotlib")
    visual = st.sidebar.selectbox("See the data", ["--", "Images", "Plot"])
    if visual == "--":
        st.write("Select a mode")
    elif visual == "Images":
        st.write("Images go here")
    else:
        st.write("Plot goes here")

else:
    st.title("Model Selection")
