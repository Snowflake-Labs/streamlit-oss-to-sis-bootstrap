# Setup

By the end of this chapter, you will have successfully created and set up a GitHub project using the [Stream Starter Kit](https://github.com/streamlit/app-starter-kit){:target=_blank}. This starter kit provides a solid foundation for building interactive web applications with Streamlit, a powerful Python library for creating data-driven apps.

In this chapter, we will guide you through the following steps:

- Forking the Streamlit Starter Kit repository to your GitHub account
- Cloning the forked repository to your local machine
- Configuring the project settings
- Optionally Running the starter app locally to ensure everything is set up correctly

Let's get started!

## What is required

To follow along with this tutorial and set up your project successfully, you'll need the following tools and accounts:

* [Visual Studio Code](https://code.visualstudio.com/){:target=_blank}

* [Docker for Desktop](https://www.docker.com/products/docker-desktop/){:target=_blank}

* A GitHub account: If you don't already have a GitHub account, you'll need to create one. 

* [Miniconda](https://docs.anaconda.com/miniconda/){:target=_blank}

* [Snowflake Account](https://signup.snowflake.com){:target=_blank}

* [Snowflake CLI](https://docs.snowflake.com/en/developer-guide/snowflake-cli/index){:target=_blank}

* And last but not the least latest Google Chrome browser

Please ensure you have all these tools and accounts set up before proceeding with the tutorial. In the following sections, we'll guide you through the process of setting up your GitHub project using the Streamlit Starter Kit.

## Creating a Project

Let us create a base Streamlit project from scratch using Streamlit application starter kit.

Navigate to the folder where you want to create your application,

```shell
cd <directory where you want to create your project>
```

Create the application folder for the rest of the demo we will call this as `st-ml-app` and for easy reference we will export to an environment variable named `$TUTORIAL_HOME`,

```shell
export TUTORIAL_HOME="$PWD/st-ml-app"
mkdir -p $TUTORIAL_HOME
cd $TUTORIAL_HOME
```

You can either directly use the template from the repo <https://github.com/streamlit/app-starter-kit>{:target=_blank},

![Streamlit Application Starter Kit](images/app-starter-kit.png)

Once you fork the repository clone the same to `$TUTORIAL_HOME`.

```shell
git clone <your template repo> $TUTORIAL_HOME
```

**(OR)**

We can use [GitHub CLI](https://cli.github.com/){:target=_blank} and create a remote git repository under your GitHub account using the same template,

```shell
gh repo create st-ml-app \
   --template='streamlit/app-starter-kit' \
   --public 
```

Clone the created repository to current directory,

```shell
gh repo clone st-ml-app .
```

## Choice of Development Environment

This tutorial can be run in many ways

- Locally on your machine installing all tools
- Using [DevContainer](https://code.visualstudio.com/docs/devcontainers/create-dev-container). The project is also enabled with DevContainers.
- Using [GitHub Codespaces](https://github.com/features/codespaces)

!!!NOTE
    For this demo we will do all the setup locally on our machines.

Once you have all that lets get to next chapter where we will deploy the bare bone application Streamlit community cloud.