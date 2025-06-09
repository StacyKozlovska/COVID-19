# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
import seaborn as sns
from ydata_profiling import ProfileReport


def initial_info(my_df):
    """
    Function to show initial info
    for the dataframe with the help of
    ydata_profiling ProfileReport.
    """
    print(my_df.info())
    case_profile = ProfileReport(my_df)
    case_profile.to_notebook_iframe()
    return None


def visualize_box_plot(my_series, my_series_name: str):
    """
    Function to plot outliers with
    box plots.
    """
    plt.figure(figsize=(15, 3))
    sns.boxplot(x=my_series)
    plt.title(f"Box Plot for {my_series_name}", fontsize=15)
    plt.xlabel(f"{my_series_name}", fontsize=14)
    plt.show()
    return None


def convert_series_to_float(s):
    """
    Function to convert series with
    strings like '30s' or '100s' into
    floats.
    """
    # Create a dictionary to map the string values to corresponding integer values
    age_mapping = {
        "0s": 0,
        "10s": 10,
        "20s": 20,
        "30s": 30,
        "40s": 40,
        "50s": 50,
        "60s": 60,
        "70s": 70,
        "80s": 80,
        "90s": 90,
        "100s": 100,
        "nan": np.nan,
    }

    # Use the dictionary to map the values in the series and convert them to integers
    series_int = s.map(age_mapping)

    # For the NaN values, convert them to numpy NaN
    series_int = series_int.where(~s.isna(), np.nan)

    return series_int


def get_corr_scores(data):
    """
    Function to get all the
    Pearson correlation scores
    for numeric columns.
    """
    # create a dataframe with correlations
    corr_data = data.corr(numeric_only=True, method="pearson")

    # Retain upper triangular values of correlation matrix and
    # make Lower triangular values Null
    upper_corr_data = corr_data.where(
        np.triu(np.ones(corr_data.shape), k=1).astype(bool)
    )

    # Convert to 1-D series and drop Null values
    unique_corr_pairs = upper_corr_data.unstack().dropna()

    # Sort correlation pairs
    sorted_corr_data = unique_corr_pairs.drop_duplicates()
    sorted_corr_data.sort_index(inplace=True)

    return sorted_corr_data


def get_corr_heatmap(data):
    """
    Function to visualize Pearson Correlation scores
    using an annotated heatmap.
    """
    # Plot correlations with annotations
    plt.figure(figsize=(8, 8))
    plt.title("Pearson Correlation of continuous features")
    ax = sns.heatmap(data.corr(numeric_only=True), annot=True)
    return ax


def plot_sns_jointplot(
    data, x: str, y: str, title: str, xlim=(-20, 850), ylim=(3, 5.1), my_figsize=(8, 5)
):
    """
    Function to automate seaborn
    jointplot plotting.
    """
    g = sns.JointGrid(data, x=x, y=y)
    g.plot_joint(sns.scatterplot, s=100, alpha=0.5)
    g.ax_marg_x.set_xlim(*xlim)
    g.ax_marg_y.set_ylim(*ylim)
    g.plot_marginals(sns.histplot, kde=True)
    g.fig.set_size_inches(my_figsize)
    g.fig.suptitle(title)

    g.fig.show()


def gender_stacked_barplot(my_df, index_col: str, my_col: str):
    """
    Function to plot a matplotlib
    stacked bar plot.
    """
    # Pivot the DataFrame to reshape it for stacked barplot
    time_gender_pivot = my_df.pivot(index=index_col, columns="sex", values=my_col)

    # Convert index to DatetimeIndex
    time_gender_pivot.index = pd.DatetimeIndex(time_gender_pivot.index)

    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Plot the stacked barplot
    time_gender_pivot.plot(kind="bar", stacked=True, width=0.8, color=["red", "blue"])

    # Customize the plot
    plt.xlabel("Date")
    plt.ylabel(f"Number of {my_col} cases")
    plt.title(f"COVID-19 {my_col} cases by gender")
    plt.legend()

    # Format x-axis ticks to display only months
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_minor_locator(FixedLocator(time_gender_pivot.index.day[0::30]))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Show the plot
    plt.tight_layout()
    plt.show()
