# Deploying Streamlit Applications in Snowflake

Previous Chapter Recap:

In the last chapter, we developed an interactive Streamlit application for predicting penguin species using our Random Forest model. We focused on creating an intuitive user interface through:

- [x] Visualization of prediction probabilities using Streamlit's progress bars
- [x] Strategic content organization with columns and containers for enhanced layout
- [x] Implementation of success messages to display clear prediction results

Moving forward, we'll explore how to deploy this application within Snowflake's ecosystem using Streamlit in Snowflake (SiS). This integration combines our interactive prediction capabilities with Snowflake's enterprise-grade data platform features.

In this chapter, we will:

- [x] Modify our existing Streamlit application for seamless Snowflake integration
- [x] Establish and configure secure Snowflake connections
- [x] Deploy and validate our application using Streamlit in Snowflake
- [x] Explore critical distinctions between local development and Snowflake deployment

## Preparing for Deployment

### Database

!!!IMPORTANT
    It is assumed that Snowflake CLI has been installed and you have test your snowflake connection.

    Set your default connection name using 

    - `snow connection set-default <your snowflake connection>`
    - Set an environment variable named `SNOWFLAKE_DEFAULT_CONNECTION_NAME`
    
    And then for quick check try:

    ```shell
    snow connection test
    ```

## Database

For this demo all our Snowflake objects will be housed in a DB called `st_ml_app`.

Create the Database

```shell
snow object create database \
  name='st_ml_app' \
  comment='Database used for Streamlit ML App Tutorial'
```

## Schema

Let us create one schema to hold the notebooks.

```shell
# notebooks
snow object create schema \
  name='notebooks' comment='Will hold all notebooks' \
  --database='st_ml_app'
```

## Using Notebook

Here's my suggested revision to make the text clearer and more precise:

We will need a few more objects and will need to ingest the penguins data into our Snowflake database `st_ml_app`, specifically in the schema `data` and into a table named `penguins`. We will use [Snowflake Notebooks](https://www.snowflake.com/en/data-cloud/notebooks/){:target=_blank} for this purpose.

Download and import this [notebook](https://github.com/Snowflake-Labs/streamlit-oss-to-sis-bootstrap/blob/main/notebooks/sis_setup.ipynb){:target=_blank} and follow its instructions to prepare the environment for deployment.

## Deploying the App

### Create Streamlit Project

Create a Snowflake project from a template, to make things easy and clean we will create the application in `$TUTORIAL_HOME/sis`.

```shell
snow init sis --template example_streamlit
```

!!! NOTE
    The application init uses the `example_streamlit` template from  <https://github.com/snowflakedb/snowflake-cli-templates>{:target=_blank}


### Update the App 

Edit and update the `$TUTORIAL_HOME/sis/streamlit_app.py` with,

```py title="streamlit_app.py" linenums="1" hl_lines="4-7 10-24 24 34-45"
import streamlit as st
import os

from sklearn.ensemble import RandomForestClassifier
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import col
from snowflake.snowpark.types import StringType, DecimalType


def get_active_session() -> Session:
    """Create or get new Snowflake Session.
       When running locally it uses the SNOWFLAKE_CONNECTION_NAME environment variable to get the connection name and when running in SiS it uses the context connection.
    """
    conn = st.connection(
        os.getenv(
            "SNOWFLAKE_CONNECTION_NAME",
            "devrel-ent",
        ),
        type="snowflake",
    )
    return conn.session()


session = get_active_session()

st.title("🤖 Machine Learning App")

st.write("Welcome to world of Machine Learning with Streamlit.")

with st.expander("Data"):
    st.write("**Raw Data**")
    # read the data from table
    # cast the columns to right data types with right precision
    df = session.table("st_ml_app.data.penguins").select(
        col("island").cast(StringType()).alias("island"),
        col("species").cast(StringType()).alias("species"),
        col("bill_length_mm").cast(DecimalType(5, 2)).alias("bill_length_mm"),
        col("bill_depth_mm").cast(DecimalType(5, 2)).alias("bill_depth_mm"),
        col("flipper_length_mm").cast(DecimalType(5, 2)).alias("flipper_length_mm"),
        col("body_mass_g").cast(DecimalType()).alias("body_mass_g"),
        col("sex").cast(StringType()).alias("sex"),
    )
    df = df.to_pandas()
    # make the column names lower to reuse the rest of the code as is
    df.columns = df.columns.str.lower()
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
    # Filpper Length
    min, max, mean = (
        df.flipper_length_mm.min(),
        df.flipper_length_mm.max(),
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

# Dataframe for Input features
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

### Verify Python Packages

Edit and update `$TUTORIAL_HOME/sis/environment.yml` to be like, ensuring that the packages used locally and in Snowflake are same.

```yaml title="environment.yml" linenums="1"
name: sf_env
channels:
  - snowflake
dependencies:
  - streamlit=1.35.0
  - snowflake-snowpark-python
  - scikit-learn=1.3.0
  - pandas=2.0.3
  - numpy=1.24.3
```

### Verify Project Manifest

Edit and update the Project [manifest](https://docs.snowflake.com/en/developer-guide/snowflake-cli/native-apps/project-definitions){:target=_blank} `$TUTORIAL_HOME/sis/snowflake.yml` to be inline with your settings, especially 

- `name`
- `main_file`
- `query_warehouse`
- `artifacts`

!!! WARNING "IMPORTANT"
    If you have followed along the tutorial as is without any changes the only that you might need to change is `query_warehouse`. If you are using trial account then update it to `COMPUTE_WH`.

```yaml title="snowflake.yml" linenums="1" hl_lines="6-8 10-12"
definition_version: "2"
entities:
  streamlit_penguin:
    type: streamlit
    identifier:
      name: streamlit_penguin
    main_file: streamlit_app.py
    query_warehouse: kamesh_demos_s
    stage: src
    artifacts:
      - streamlit_app.py
      - environment.yml

```

Navigate to the application(**sis**) folder,

```shell
cd sis
```
Run the following command to deploy the Streamlit application to Snowflake,

!!!NOTE
    The output of the command displays the URL to access the application. In case you missed to note run the following command,
    ```shell
    # get the url to streamlit app named "streamlit_penguin"
    snow streamlit get-url streamlit_penguin
    ```

```shell
snow streamlit deploy --replace \
  --database='st_ml_app'  --schema='apps'
```

There you go we have seamlessly deployed the application to SiS with a very little effort.

## Summary
This chapter guided you through the process of transforming a locally running Streamlit application into a production-ready deployment within Snowflake. You learned the essential modifications needed for Snowflake compatibility, understood the configuration requirements, and mastered the deployment process. You now have a fully functional Streamlit application running in Snowflake's secure environment, accessible to your organization's users through Snowflake's interface.

> **Pro Tip**: You can run this entire application using Snowflake Notebook. Download and import the [environment.yml](https://github.com/Snowflake-Labs/streamlit-oss-to-sis-bootstrap/blob/main/notebooks/environment.yml){:target=_blank} and [application notebook](https://github.com/Snowflake-Labs/streamlit-oss-to-sis-bootstrap/blob/main/notebooks/streamlit_penguin_app.ipynb){:target=_blank} and experience the Snowflake Notebook magic! 🚀
