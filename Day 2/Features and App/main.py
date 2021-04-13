import io

import streamlit as st

# Adding a title -> <h1> Features and App Page </h1>
st.title("Features and App Page")

# Attaching image to the website -> <img src="images/globalaihub.png" />
from PIL import Image

st.subheader("Global AI Hub Logo Down Below:")
image = Image.open("images/globalaihub.png")
st.image(image, use_column_width=True)

st.subheader("Global AI Hub Logo Down Below Black:")
image3 = Image.open("images/globalaihubbl.png")
st.image(image3, use_column_width=True)

# OpenCV
import cv2

st.subheader("Global AI Hub Logo OpenCV Below:")
image2 = cv2.imread("images/globalaihublogo.jpg")
image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
st.image(image2, use_column_width=True)

# Adding a text -> <p> This is a custom text! </p> // Supports data_frame/error/func/module/dict/obj/keras/plotly_fig...
st.write("This is a custom text!")

# Emoji Support
st.write("Hello, World! :nerd_face:")
st.write("Hello ! :wink:")

# Multiple Different Type Argument Support
st.write("5 + 5 = ", 10, "Do you like that?", {5: 5}, [1, 2, 3, 4, 5, 6], True)

# LaTex Support
st.latex(r'''
...     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
...     \sum_{k=0}^{n-1} ar^k =
...     a \left(\frac{1-r^{n}}{1-r}\right)
...     ''')

# Adding Markdown -> Jupyter Notebook Cell Markdowns / GitHub README.md etc.
st.markdown("""
## ![Welcome to my profile 🤟](https://assets.materialup.com/uploads/805362d3-e9d6-4aa7-b314-ed9dde22558b/preview.gif)

🔭 I love Data Science!

`Some codes go here`

```
Some more codes here (click to copy)
```
```
 /$$   /$$           /$$ /$$           /$$
| $$  | $$          | $$| $$          | $$
| $$  | $$  /$$$$$$ | $$| $$  /$$$$$$ | $$
| $$$$$$$$ /$$__  $$| $$| $$ /$$__  $$| $$
| $$__  $$| $$$$$$$$| $$| $$| $$  \ $$|__/
| $$  | $$| $$_____/| $$| $$| $$  | $$    
| $$  | $$|  $$$$$$$| $$| $$|  $$$$$$/ /$$
|__/  |__/ \_______/|__/|__/ \______/ |__/
```

```
st.subheader("Global AI Hub Logo Down Below:")
image = Image.open("images/globalaihub.png")
st.image(image, use_column_width=True)
```

```
while True:
    iLoveGAIH("<3")
```
""")

# Code Syntax Support
myCode = """
a = 5
b = 10

for i in range(a):
    for j in range(b):
        print("Hello, World!)      
"""
st.code(myCode, language="python")

# API and output management Interaction Widgets
# Exceptions

# Success -> Green Hued Success Output
st.success("Success!")

# Info -> Gray Hued Info Output
st.info("Info!")

# Warning -> Yellow Hued Warning Output
st.warning("Warning!")

# Error -> Red Hued Error/Fail Output
st.error("Fail/Error!")

# Exception -> Custom Exception Showcase
st.exception("Error!")

# Help -> Python Help / Function Representation
st.help(5)

# DataFrames -> Reading from a .csv file
import pandas as pd
import numpy as np

df = pd.read_csv("Machine_readable_file_bd_employ.csv")
st.dataframe(df)

# Numpy Arrays -> Custom Generated Dataframe Visualization
dataframe = np.random.rand(50, 20)
st.dataframe(dataframe)

# General Python Formattings and Syntax Usage in Streamlit
st.write("0" * 100)
st.text("0" * 100 + "GLOBAL AI HUB GLOBAL AI HUB GLOBAL AI HUB GLOBAL AI HUB ")

st.text("".join([str(i) for i in range(100)]))

# Table
import numpy as np

df = pd.DataFrame(
    np.random.rand(5, 5)
)
st.table(df)

# Chart Table
chartTable = pd.DataFrame(
    np.random.rand(25, 5),
    columns=["A", "B", "X", "Y", "Z"]
)
st.subheader("Chart Table")
st.line_chart(chartTable)

# Colored Chart Table
coloredChartTable = pd.DataFrame(
    np.random.rand(25, 3),
    columns=["A", "B", "C"]
)
st.subheader("Colored Chart")
st.area_chart(coloredChartTable)

# Bar Chart
barChart = pd.DataFrame(
    np.random.rand(25, 3),
    columns=["A", "B", "C"]
)
st.subheader("Bar Chart")
st.bar_chart(barChart)

# Displaying Plots
import matplotlib.pyplot as plt

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.subheader("Matplotlib Plot")
st.pyplot(fig)

# Displaying Altair
import altair as alt

df = pd.DataFrame(
    np.random.randn(100, 3),
    columns=["a", "b", "c"]
)

c = alt.Chart(df).mark_circle().encode(
    x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c']
)

st.subheader("Altair Chart")
st.altair_chart(c, use_container_width=True)

# More Altair Examples Can Be Found Here: https://altair-viz.github.io/gallery/

# Feature-Rich Plotly Chart
import plotly.figure_factory as ff

x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

hist_data = [x1, x2, x3]

group_labels = ['1', '2', '3']

fig = ff.create_distplot(
    hist_data, group_labels, bin_size=[.1, .25, .5])

st.subheader("Feature-Rich Plotly Chart")
st.plotly_chart(fig, use_container_width=True)

# Bokeh Chart
from bokeh.plotting import figure

x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

p = figure(
    title='simple line example',
    x_axis_label='x',
    y_axis_label='y')

p.line(x, y, legend='Trend', line_width=2)
st.subheader("Bokeh Chart")
st.bokeh_chart(p, use_container_width=True)

# Graphiz -> Path Draw
import graphviz as graphviz

graph = graphviz.Digraph()
graph.edge("1", "2")
graph.edge("2", "3")
graph.edge("2", "4")
graph.edge("3", "5")
graph.edge("4", "6")
graph.edge("5", "6")
graph.edge("6", "9")
graph.edge("7", "9")
graph.edge("8", "9")
graph.edge("4", "7")
graph.edge("4", "8")

st.subheader("Graphiz")
st.graphviz_chart(graph)

# Map -> Lattitude / Longitude
df = pd.DataFrame(
    np.random.randn(1000, 2) / [10, 10] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.subheader("Map")
st.map(df)

# Music Import
music = open('sounds/bensound-hey.mp3', 'rb')
musicBytes = music.read()

st.subheader("Custom Music / Sound")
st.text("Royalty Free Music from Bensound")
st.audio(musicBytes, format="audio/ogg")

# Video File Import
video = open('video/Pexels Videos 2282013.mp4', 'rb')
videoBytes = video.read()

st.subheader("Video Integration")
st.video(video)

# Interactive Widgets

# Button
st.subheader("Button")

if st.button('Say hello'):
    st.write("Hi, there!")
else:
    st.write("See you!")

# Checkbox
st.subheader("Check Boxes")

confirm = st.checkbox("I agree.")
decline = st.checkbox("I disagree.")

if confirm:
    st.write("Welcome.")
elif decline:
    st.write("Goodbye.")

# Radio Box
st.subheader("Radio Box")

hobbyChoice = st.radio("What is your hobby?",
                       ("Golf", "Gaming", "Learning", "Sleeping"))

if hobbyChoice == "Golf":
    st.write("I love playing Golf!")
elif hobbyChoice == "Gaming":
    st.write("I love playing Games!")
elif hobbyChoice == "Learning":
    st.write("I love Learning in my free time!")
else:
    st.write("Please, select a hobby you like :)")

# Select Box
st.subheader("Select Box")

selectBox = st.selectbox(
    'Where are you from?',
    ('Europe', 'America', 'Asia'))

st.write('You are from:', selectBox)

# Multiselect Box
st.subheader("Multiselect Box")

multiSelect = st.multiselect(
    "What colors do you like?",
    ["Blue", "Red", "Green", "Black", "White"],
    ["Red"]  # Default Selected
)

# Slider -> Realtime Data Change
st.subheader("Slider")

age = st.slider('How old are you?', 0, 130, 25)  # 25 is default value selected automatically to be changed
st.write(f"I am {age} years old!")

# Time Range Slider
st.subheader("Time Range Slider")

from datetime import time

appointment = st.slider(
    "Schedule your appointment:",
    value=(time(11, 30), time(12, 45)))

st.write(f"Your appointment starts at {appointment[0]}, and ends at {appointment[1]}")

# Text Input
st.subheader("Text Input Field")

textInput = st.text_input('What is your name?')
if textInput:
    st.text(f"Hello, {textInput.capitalize()}. Welcome to the system!")

# Number Input
st.subheader("Number Input")

numberInput = st.number_input('Insert a number')
st.write(f"You have chosen {numberInput}")

# "BGR" -> "RGB"
st.subheader("OpenCV Images")

bgrImg = cv2.imread("images/globalaihublogo.jpg")
# Default Read image with OpenCV
st.image(bgrImg, use_column_width=True, caption="Default Read image with OpenCV")
# Converted BGR image with OpenCV
st.image(bgrImg, use_column_width=True, channels="BGR", caption="Converted BGR image with OpenCV")

# Text Area -> MD (Markdown) Parser
st.subheader("Text Area")

textArea = st.text_area("Text to analyze", """""")

if textArea:
    st.write('MarkDown Parser:', f"""{textArea}""")

# File Uploader -> Preprocessing and visualizing files
uploadedFile = st.file_uploader("Upload a file")

# Default Docs require StringIO and cStringIO modules, which do not exist anymore. -> One file at a time

st.subheader("Upload and read .csv file")

if uploadedFile is not None:
    dataframe = pd.read_csv(uploadedFile)
    st.write(dataframe)

# Multiple Upload and Showcase of .csv Files

st.subheader("Multiple Upload and Showcase of .csv Files")

multipleUploadedFiles = st.file_uploader("Choose CSV files", accept_multiple_files=True)

if multipleUploadedFiles:
    for uploaded_file in multipleUploadedFiles:
        st.subheader(f"File: {uploaded_file.name}")
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

# Color Picker
colorPick = st.color_picker('Pick A Color', '#00f900')
st.write(f"Color is {colorPick}")

# Killing the current process -> st.stop()
yourName = st.text_input("Please enter your name")
if not yourName:
    st.warning("Please enter your name!")
    st.stop()

st.success(f"Welcome back, {yourName.capitalize()}!") # Not executed if name is not included