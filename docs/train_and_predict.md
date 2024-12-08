# Training and Prediction

Now that our data is properly encoded, we'll use a classification model to predict penguin species based on their filtered physical characteristics. We'll employ scikit-learn's [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) to calculate probability scores for each species prediction.

In this chapter, we will:

- [x] Split our filtered dataset into training and testing sets
- [x] Train a Random Forest Classifier using our selected feature set  
- [x] Calculate prediction probabilities for each penguin species
- [x] Visualize these probabilities using Streamlit's interactive components

## Train and Predict

As part of displaying the predictions we will use the following Streamlit components to make the output aesthetically appealing,

- [Progress Column Config](https://docs.streamlit.io/develop/api-reference/data/st.column_config/st.column_config.progresscolumn)
- [Success Message](https://docs.streamlit.io/develop/api-reference/status/st.success)
- [Container](https://docs.streamlit.io/develop/api-reference/layout/st.container)

Update `$TUTORIAL_HOME/requirements.txt` to be like

```
streamlit>=1.26.0
scikit-learn
```

Update the local Python Virtual Environment to add the `scikit-learn` package,

```shell
pip install -r requirements.txt
```

Edit and update the `$TUTORIAL_HOME/streamlit_app.py` with the following code,

```py title="streamlit_app.py" linenums="1" hl_lines="5 147-198"
import streamlit as st

# import pandas to read the our data file
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("🤖 Machine Learning App")

st.write("Welcome to world of Machine Learning with Streamlit.")

with st.expander("Data"):
    st.write("**Raw Data**")
    # read the csv file
    df = pd.read_csv("data/penguins_cleaned.csv")
    df
    # define and display
    st.write("**X**")
    X_raw = df.drop("species", axis=1)
    X_raw

    st.write("**y**")
    y_raw = df.species
    y_raw

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
    # Flipper Length
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

# DataFrames for Input features
data = {
    "island": island,
    "bill_length_mm": bill_length_mm,
    "bill_depth_mm": bill_depth_mm,
    "flipper_length_mm": flipper_length_mm,
    "body_mass_g": body_mass_g,
    "sex": gender,
}
input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander("Input Features"):
    st.write("**Input Penguins**")
    input_df
    st.write("**Combined Penguins Data**")
    input_penguins

## Data Preparation

## Encode X
X_encode = ["island", "sex"]
df_penguins = pd.get_dummies(input_penguins, prefix=X_encode)
X = df_penguins[1:]
input_row = df_penguins[:1]

## Encode Y
target_mapper = {
    "Adelie": 0,
    "Chinstrap": 1,
    "Gentoo": 2,
}


def target_encoder(val_y: str) -> int:
    return target_mapper[val_y]


y = y_raw.apply(target_encoder)

with st.expander("Data Preparation"):
    st.write("**Encoded X (input penguins)**")
    input_row
    st.write("**Encoded y**")
    y


with st.container():
    st.subheader("**Prediction Probability**")
    ## Model Training
    rf_classifier = RandomForestClassifier()
    # Fit the model
    rf_classifier.fit(X, y)
    # predict using the model
    prediction = rf_classifier.predict(input_row)
    prediction_prob = rf_classifier.predict_proba(input_row)

    # reverse the target_mapper
    p_cols = dict((v, k) for k, v in target_mapper.items())
    df_prediction_prob = pd.DataFrame(prediction_prob)
    # set the column names
    df_prediction_prob.columns = p_cols.values()
    # set the Penguin name
    df_prediction_prob.rename(columns=p_cols)

    st.dataframe(
        df_prediction_prob,
        column_config={
            "Adelie": st.column_config.ProgressColumn(
                "Adelie",
                help="Adelie",
                format="%f",
                width="medium",
                min_value=0,
                max_value=1,
            ),
            "Chinstrap": st.column_config.ProgressColumn(
                "Chinstrap",
                help="Chinstrap",
                format="%f",
                width="medium",
                min_value=0,
                max_value=1,
            ),
            "Gentoo": st.column_config.ProgressColumn(
                "Gentoo",
                help="Gentoo",
                format="%f",
                width="medium",
                min_value=0,
                max_value=1,
            ),
        },
        hide_index=True,
    )

# display the prediction
st.subheader("Predicted Species")
st.success(p_cols[prediction[0]])
```

## Summary

After successfully creating an interactive penguin species prediction app using Streamlit's Progress bars for probability visualization, Columns for layout, and Containers for organized content, we'll now explore how to deploy this same application in Snowflake using Streamlit in Snowflake (SiS). This will allow us to leverage Snowflake's data platform while maintaining our app's interactive features.

In the next chapter, we will:

* Adapt our existing Streamlit app for Snowflake environment
* Configure necessary Snowflake connections
* Deploy and test our application using SiS
* Understand key differences between local and Snowflake deployment