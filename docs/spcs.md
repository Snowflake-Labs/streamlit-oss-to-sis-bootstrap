# Deploying Streamlit Applications using Snowpark Container Services

In the previous chapter we learnt how to deploy Streamlit application in Snowflake with SiS. In this chapter we will explore how to containerize the application and deploy it using [Snowpark Container Services(SPCS)](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview).

In this chapter, we will:

- [x] Add a Dockerfile file containerize the application
- [x] Create Compute Pool
- [x] Create Image Repository
- [x] Build and Deploy the application

## Prerequisites

- You have completed the previous chapter on [Deploying to SiS](./snowflake_deploy.md)
- Snow CLI is installed and configured
- [Docker](https://www.docker.com/products/docker-desktop/) is installed and configured
- [Notebook](./snowflake_deploy.md#using-notebook) has been imported into your Snowflake Account and ready to for use.

## Preparing for SPCS Deployment

To be able to deploy our Penguins ML app as container onto SPCS we need to 

- Create compute pool
- Create image repository
- Build and Push the container image to image repository
- Deploy service

### Create Compute Pool

All SPCS containers run using a required size of compute, check the [documentation](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/working-with-compute-pool) for more details on available compute pool sizes.

Let us create one for this tutorial.

```shell
export ST_ML_APP_COMPUTE_POOL=st_ml_app_xs
snow spcs compute-pool create $ST_ML_APP_COMPUTE_POOL \
  --family CPU_X64_XS
```

Let us describe the created compute pool,

```shell
snow spcs compute-pool describe $ST_ML_APP_COMPUTE_POOL --format=json
```

The compute pool is another Snowflake resource, it has all the usual operations like list, describe etc., check the [cheatsheet](https://github.com/Snowflake-Labs/sf-cheatsheets/blob/main/snow-cli-spcs-cheatsheet.md){:target=_blank}.


### Creating SPCS Objects 

!!!IMPORTANT
    - SPCS containers cant be run as `ACCOUNTADMIN` hence we need to create a new role and use that role to run the containers.

Let us switch to our notebook that we used earlier and run the following cells under the section **Snowpark Container Services(SPCS)**,

- `sql_current_user_role`
- `sql_current_database`
- `sql_current_user`
- `spcs_variables`
- `spcs_objects` 

After successful execution of the cells you would have created,

- A Role named `st_ml_app` to create Snowpark Container Services
- A Schema named `images` on DB `ST_ML_APP` to hold the image repository.
- A Warehouse `st_ml_app_spcs_wh_s` which will be used to run query from services.


### Building and Pushing the Image

Get image registry URL,

```shell
export IMAGE_REGISTRY_URL=$(snow spcs image-registry url)
```

#### Create Image Repository

Create an image repository if not exists,

```shell
snow spcs image-repository create st_ml_apps \
  --database='st_ml_app' \
  --schema='images' \
  --role='st_ml_app' \
  --if-not-exists
```

Get the image repository `st_ml_apps` and store it the environment variable `$IMAGE_REPO`,

```shell
export IMAGE_REPO=$(snow spcs image-repository url st_ml_apps \
  --database='st_ml_app' \
  --schema='images' \
  --role='st_ml_app')
```

#### Update the App

Edit and update the `$TUTORIAL_HOME/sis/streamlit_app.py` with,

```py title="streamlit_app.py" linenums="1" hl_lines="12-23 38-44 50-67"
import logging
import os
import sys

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from snowflake.snowpark.functions import col
from snowflake.snowpark.session import Session
from snowflake.snowpark.types import DecimalType, StringType

# Environment variables below will be automatically populated by Snowflake.
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_HOST = os.getenv("SNOWFLAKE_HOST")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")

# Custom environment variables
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")


SERVICE_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = os.getenv("SERVER_PORT", 8080)


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(name)s [%(asctime)s] [%(levelname)s] %(message)s")
    )
    logger.addHandler(handler)
    return logger


def get_login_token():
    """
    Read the login token supplied automatically by Snowflake. These tokens
    are short lived and should always be read right before creating any new connection.
    """
    with open("/snowflake/session/token", "r") as f:
        return f.read()


logger = get_logger("penguins-ml-app")


def get_active_session() -> Session:
    """Create or get new Snowflake Session.
    When running locally it uses the SNOWFLAKE_CONNECTION_NAME environment variable to get the connection name and
    when running in SiS it uses the context connection.
    """
    token = get_login_token()
    session = Session.builder.configs(
        {
            "account": SNOWFLAKE_ACCOUNT,
            "host": SNOWFLAKE_HOST,
            "authenticator": "oauth",
            "token": token,
            "warehouse": SNOWFLAKE_WAREHOUSE,
            "database": SNOWFLAKE_DATABASE,
            "schema": SNOWFLAKE_SCHEMA,
        }
    ).getOrCreate()
    return session


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
    # Flipper Length
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

#### Building the Application Container Image

!!!IMPORTANT
    This uses the `docker` daemon, if you have not installed please install [Docker](https://www.docker.com/products/docker-desktop/) before proceeding further.


Let us login to the SPCS image registry,

```shell
snow spcs image-registry login
```

Let us build, tag and push the `$IMAGE_REPO/penguins_app`  to the `$IMAGE_REGISTRY_URL` ,

```shell
docker build --push  --platform=linux/amd64 -t $IMAGE_REPO/penguins_app .
```

Let us list all images in repository `st_ml_apps`,

```shell
snow spcs image-repository list-images st_ml_apps \
  --database='st_ml_app' \
  --schema='images' \
  --role='st_ml_app'
```

Let us export the image FQN to a variable to easy reference and substitution,

```shell
export IMAGE_REPO_NAME="$IMAGE_REPO/penguins_app"
```

### Deploy Service

Create a SPCS service specification file,

!!!NOTE
    Replace `$IMAGE_REPO_NAME` with actual value. If you are on Linux/macOS you can use [envsubst](https://www.gnu.org/software/gettext/manual/html_node/envsubst-Invocation.html) like,

    ```shell
        cat <<EOF | tee work/service-spec.yaml
    spec:
        containers:
        - name: st-ml-app
            image: $IMAGE_REPO_NAME
            env:
            SNOWFLAKE_WAREHOUSE: st_ml_app_spcs_wh_s
            readinessProbe:
                port: 8080
                path: /
        endpoint:
        - name: st-ml-app
            port: 8080
            public: true
    EOF
    ```

Create a new directory named `work` and create the following file in it,

```yaml title="service-spec.yaml" linenums="1" hl_lines="4-6"
spec:
  containers:
    - name: st-ml-app
        image: $IMAGE_REPO_NAME
        env:
          SNOWFLAKE_WAREHOUSE: st_ml_app_spcs_wh_s        
        readinessProbe:
          port: 8080
          path: /
  endpoint:
    - name: st-ml-app
      port: 8080
      public: true
```

Create a Service,

```shell
snow spcs service create st_ml_app \
  --compute-pool=$ST_ML_APP_COMPUTE_POOL \
  --spec-path=work/service-spec.yaml \
  --if-not-exists \
  --database='st_ml_app' \
  --schema='images' \
  --role='st_ml_app'
```

#### Service Status

Check service status,

!!!NOTE
    It will take few minutes for the service to be in `RUNNING` status

```shell
snow spcs service describe st_ml_app \
  --database='st_ml_app' \
  --schema='images' \
  --role='st_ml_app'\
  --format json | jq '.[0].status'
```

#### Service Endpoints

List the service endpoint for the service `st_ml_app`,

!!!NOTE
    It will take few minutes for the Endpoint URL to be ready.

```shell
snow spcs service list-endpoints st_ml_app  \
  --database='st_ml_app' \
  --schema='images' \
  --role='st_ml_app' --format=json | jq '.[0].ingress_url'
```

Open the application using the URL from the previous command, authenticate to access the Penguins ML application.

## Cleanup
To clean up the services created as part of this demo run the cell `spcs_cleanup` on the notebook.

## References 

### Quickstarts

- [Snowflake Developers::Quickstart](https://quickstarts.snowflake.com/guide/getting-started-with-snowflake-cli/#0)
- [Snowflake Developers::Getting Started With Snowflake CLI](https://youtu.be/ooyZh56NePA?si=3yV3s2z9YwPWVJc-)
- [Intro to Snowpark Container Services](https://quickstarts.snowflake.com/guide/intro_to_snowpark_container_services/index.html?index=../..index#0)
- [Build a Data App and run it on Snowpark Container Services](https://quickstarts.snowflake.com/guide/build_a_data_app_and_run_it_on_Snowpark_container_services/index.html?index=../..index#0)

### Documentation

- [Snowflake CLI](https://docs.snowflake.com/en/developer-guide/snowflake-cli-v2/index)
- [Execute Immediate Jinja Templating](https://docs.snowflake.com/en/sql-reference/sql/execute-immediate-from)
- [Snowpark Container Services](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview)
- <https://docs.snowflake.com/en/sql-reference/sql/create-compute-pool>
- <https://docs.snowflake.com/en/developer-guide/snowpark-container-services/specification-reference>

### Tutorials

- [Snowpark Container Services Tutorial](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview-tutorials)