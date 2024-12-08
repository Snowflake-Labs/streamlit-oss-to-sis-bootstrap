# Interactive Features

In this chapter, we'll discover how to add user interactivity to our application using Streamlit widgets:

- [⚡ Sidebar](https://docs.streamlit.io/library/api-reference/layout/st.sidebar){:target="_blank"} for organizing widgets in a collapsible panel
- [⚡ Select box](https://docs.streamlit.io/library/api-reference/widgets/st.selectbox){:target="_blank"} for choosing categorical features
- [⚡ Slider](https://docs.streamlit.io/library/api-reference/widgets/st.slider){:target="_blank"} for adjusting numerical values
- [⚡ Radio buttons](https://docs.streamlit.io/library/api-reference/widgets/st.radio){:target="_blank"} for making single-choice selections
- [⚡ Checkbox](https://docs.streamlit.io/library/api-reference/widgets/st.checkbox){:target="_blank"} for toggle options

These widgets will allow users to dynamically modify model feature values, which will then update our model's predictions in real-time.

## Build a Sidebar

Streamlit's sidebar provides a convenient way for users to interactively filter and select input features for our machine learning model, enhancing the app's functionality and user experience. To implement this, let's update the `$TUTORIAL_HOME/streamlit_app.py` file with the following code:

```py title="streamlit_app.py" linenums="1" hl_lines="39-43 50-54 62-66 74-79 86-91 93-96"
import streamlit as st

# import pandas to read the our data file
import pandas as pd

st.title("🤖 Machine Learning App")

st.write("Welcome to world of Machine Learning with Streamlit.")

with st.expander("Data"):
    st.write("**Raw Data**")
    # read the csv file
    df = pd.read_csv("data/penguins_cleaned.csv")
    df
    # define and display
    st.write("**X**")
    X = df.drop("species", axis=1)
    X

    st.write("**y**")
    y = df.species
    y

with st.expander("Data Visualization"):
    st.scatter_chart(
        df,
        x="bill_length_mm",
        y="body_mass_g",
        color="species",
    )

# Interactivity
# Columns:
# 'species', 'island', 'bill_length_mm', 'bill_depth_mm',
# 'flipper_length_mm', 'body_mass_g', 'sex'
with st.sidebar:
    st.header("Input Features")
    # Islands
    islands = df.island.unique().astype(str)
    island = st.selectbox(
        "Island",
        islands,
    )
    # Bill Length
    min, max, mean = (
        df.bill_length_mm.min(),
        df.bill_length_mm.max(),
        df.bill_length_mm.mean().round(2),
    )
    bill_length_mm = st.slider(
        "Bill Length(mm)",
        min_value=min,
        max_value=max,
        value=mean,
    )
    # Bill Depth
    min, max, mean = (
        df.bill_depth_mm.min(),
        df.bill_depth_mm.max(),
        df.bill_depth_mm.mean().round(2),
    )
    bill_depth_mm = st.slider(
        "Bill Depth(mm)",
        min_value=min,
        max_value=max,
        value=mean,
    )
    # Filpper Length
    min, max, mean = (
        df.flipper_length_mm.min().astype(float),
        df.flipper_length_mm.max().astype(float),
        df.flipper_length_mm.mean().round(2),
    )
    flipper_length_mm = st.slider(
        "Flipper Length(mm)",
        min_value=min,
        max_value=max,
        value=mean,
    )
    # Body Mass
    min, max, mean = (
        df.body_mass_g.min().astype(float),
        df.body_mass_g.max().astype(float),
        df.body_mass_g.mean().round(2),
    )
    body_mass_g = st.slider(
        "Body Mass(g)",
        min_value=min,
        max_value=max,
        value=mean,
    )
    # Gender
    gender = st.radio(
        "Gender",
        ("Male", "Female"),
    )
```

## Feature Data Preprocessing

We will now filter the `data` DataFrame based on the selected features and use [`pd.concat`](https://pandas.pydata.org/docs/reference/api/pandas.concat.html){:target=_blank}, ensuring the new data undergoes the same preprocessing steps as our training data.

Edit and update the `$TUTORIAL_HOME/streamlit_app.py` with the following code,

```py title="streamlit_app.py" linenums="1" hl_lines="99-114"
import streamlit as st

# import pandas to read the our data file
import pandas as pd

st.title("🤖 Machine Learning App")

st.write("Welcome to world of Machine Learning with Streamlit.")

with st.expander("Data"):
    st.write("**Raw Data**")
    # read the csv file
    df = pd.read_csv("data/penguins_cleaned.csv")
    df
    # define and display
    st.write("**X**")
    X = df.drop("species", axis=1)
    X

    st.write("**y**")
    y = df.species
    y

with st.expander("Data Visualization"):
    st.scatter_chart(
        df,
        x="bill_length_mm",
        y="body_mass_g",
        color="species",
    )

# Interactivity
# Columns:
# 'species', 'island', 'bill_length_mm', 'bill_depth_mm',
# 'flipper_length_mm', 'body_mass_g', 'sex'
with st.sidebar:
    st.header("Input Features")
    # Islands
    islands = df.island.unique().astype(str)
    island = st.selectbox(
        "Island",
        islands,
    )
    # Bill Length
    min, max, mean = (
        df.bill_length_mm.min(),
        df.bill_length_mm.max(),
        df.bill_length_mm.mean().round(2),
    )
    bill_length_mm = st.slider(
        "Bill Length(mm)",
        min_value=min,
        max_value=max,
        value=mean,
    )
    # Bill Depth
    min, max, mean = (
        df.bill_depth_mm.min(),
        df.bill_depth_mm.max(),
        df.bill_depth_mm.mean().round(2),
    )
    bill_depth_mm = st.slider(
        "Bill Depth(mm)",
        min_value=min,
        max_value=max,
        value=mean,
    )
    # Filpper Length
    min, max, mean = (
        df.flipper_length_mm.min().astype(float),
        df.flipper_length_mm.max().astype(float),
        df.flipper_length_mm.mean().round(2),
    )
    flipper_length_mm = st.slider(
        "Flipper Length(mm)",
        min_value=min,
        max_value=max,
        value=mean,
    )
    # Body Mass
    min, max, mean = (
        df.body_mass_g.min().astype(float),
        df.body_mass_g.max().astype(float),
        df.body_mass_g.mean().round(2),
    )
    body_mass_g = st.slider(
        "Body Mass(g)",
        min_value=min,
        max_value=max,
        value=mean,
    )
    # Gender
    gender = st.radio(
        "Gender",
        ("male", "female"),
    )

# Dataframes for Input features
data = {
    "island": island,
    "bill_length_mm": bill_length_mm,
    "bill_depth_mm": bill_depth_mm,
    "flipper_length_mm": flipper_length_mm,
    "body_mass_g": body_mass_g,
    "sex": gender,
}
input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X], axis=0)

with st.expander("Input Features"):
    st.write("**Input Penguins**")
    input_df
    st.write("**Combined Penguins Data**")
    input_penguins
```

Now that we've prepared our input data, let's handle the categorical variables using encoding techniques. Feature encoding is crucial for converting categorical data into a format suitable for machine learning models.

Helpful resources for encoding:

- Scikit-learn encoding guide: <https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features>{:target=_blank}
- Pandas `get_dummies` documentation: <https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html>{:target=_blank}