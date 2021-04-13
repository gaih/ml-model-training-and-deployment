import streamlit as st
import pandas as pd
import numpy as np

# Selectbox in the Sidebar
sidebarSelectBox = st.sidebar.selectbox("How would you like to be contacted?",
                                        ("Email", "Home phone", "Mobile phone"))

st.sidebar.write(sidebarSelectBox)

# Layout & Container
with st.beta_container():
    st.write("This text is displayed within the given container!")
    st.bar_chart(np.random.randn(50, 3))

st.write("This text is displayed outside the given container!")

# Custom Container
st.subheader("Custom Container")

customContainer = st.beta_container()

# Inside
customContainer.write("This is inside the container")

# Outside
st.write("This is outside the container")

# Outisde
st.subheader("Outside Custom Container")

# Inside
customContainer.write("This is inside too")

# Columns -> Streamlit Docs
col1, col2, col3 = st.beta_columns(3)

with col1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg")

# Code Display
st.subheader("Code Display")

with st.echo():
    st.write("Code!")

# Progress and Status

import time

myBar = st.progress(0)

for percent in range(100):
    time.sleep(0.05)
    myBar.progress(percent + 1)

# Temporarily Text Display
with st.spinner('Wait for it...'):
    time.sleep(1)
st.success("Done!")

# Ballons
st.balloons()

# Timer
import time

with st.empty():
    for seconds in range(60):
        st.write(f"⏳ {seconds} seconds have passed")
        time.sleep(1)
st.write("✔️ 1 minute over!")
