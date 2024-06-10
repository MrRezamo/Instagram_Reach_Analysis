# Visualization libraries

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Basic data handling and numerical operations

import pandas as pd
import numpy as np
import duckdb as db
import dask as dd
import polars as pl

import missingno as msno
import klib
 
# Additional settings to ignore warnings

import warnings
warnings.filterwarnings('ignore')

# Pandas display options for better data frame visualization

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

import sys

# SHAP for model explanations

import shap

# Model evaluation metrics

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    r2_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Preprocessing tools

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    Normalizer,
    FunctionTransformer,
    MinMaxScaler,
)
from sklearn.impute import KNNImputer, SimpleImputer

from scipy.stats import chi2_contingency, pointbiserialr
import statsmodels.api as sm
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# colors
COLOR_MAIN = "#69b3a2"
COLOR_CONTRAST = "#B3697A"
Rich_Purple = "#7D5A5A"
Warm_Beige = "#F3EBD3" 
Deep_Teal = "#31708E"
Soft_Yellow = "#F2E394"

custom_palette = [Rich_Purple,Warm_Beige,
                  Deep_Teal,Soft_Yellow]

def get_cmap():
    """
    Returns a matplotlib colormap with a main color and a contrast color.

    Returns:
    matplotlib.colors.LinearSegmentedColormap: The matplotlib colormap.
    """
    norm = mpl.colors.Normalize(-1, 1)
    colors = [
        [norm(-1.0), COLOR_CONTRAST],
        [norm(0.0), "#ffffff"],
        [norm(1.0), COLOR_MAIN],
    ]
    return mpl.colors.LinearSegmentedColormap.from_list("", colors)


def countplot(
    data: pd.DataFrame,
    column_name: str,
    title: str = "Countplot",
    hue: str = None,
    ax=None,
    figsize=(10, 5),
    bar_labels: bool = False,
    bar_label_kind: str = "percentage",
    horizontal: bool = False,
):
    """
    Generate a countplot for a given column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to plot.
        title (str, optional): The title of the countplot. Defaults to "Countplot".
        hue (str, optional): The column name to use for grouping the countplot. Defaults to None.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created. Defaults to None.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).
        bar_labels (bool, optional): Whether to add labels to the bars. Defaults to False.
        bar_label_kind (str, optional): The kind of labels to add to the bars. Can be "percentage" or "count". Defaults to "percentage".

    Returns:
        matplotlib.axes.Axes: The axis object containing the countplot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=(10, 5)) if ax is None else (plt.gcf(), ax)
    palette = custom_palette if hue else COLOR_MAIN

    if hue:
        if horizontal:
            sns.countplot(
                data=data,
                y=column_name,
                ax=ax,
                color=COLOR_MAIN,
                palette=palette,
                hue=hue,
            )
        else:
            sns.countplot(
                data=data,
                x=column_name,
                ax=ax,
                color=COLOR_MAIN,
                palette=palette,
                hue=hue,
            )
    else:
        if horizontal:
            sns.countplot(data=data, y=column_name, ax=ax, color=palette)
        else:
            sns.countplot(data=data, x=column_name, ax=ax, color=palette)

    ## Add bar labels
    if bar_labels:
        for container in ax.containers:
            if bar_label_kind == "percentage":
                ax.bar_label(container, fmt=lambda x: f" {x / len(data):.1%}")
            else:
                ax.bar_label(container, fmt=lambda x: f" {x}")

    ## Add title
    ax.set_title(label=title, fontsize=16)
    return ax


def boxplot(
    data: pd.DataFrame,
    column_name: str,
    title: str = "Boxplot",
    ax=None,
    figsize=(10, 5),
):
    """
    Create a boxplot for a given column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to create the boxplot for.
        title (str, optional): The title of the boxplot. Defaults to "Boxplot".
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).

    Returns:
        matplotlib.axes.Axes: The axis object containing the boxplot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    ## Create plot
    sns.boxplot(
        data=data,
        y=column_name,
        ax=ax,
        color=COLOR_MAIN,
    )

    ## Add title
    ax.set_title(label=title, fontsize=16)
    return ax


def histplot(
    data: pd.DataFrame,
    column_name: str,
    hue: str = None,
    title: str = "Histogram",
    ax=None,
    figsize=(10, 5),
    kde: bool = False,
    palette=[COLOR_MAIN, COLOR_CONTRAST],
):
    """
    Plot a histogram of a specified column in a pandas DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to plot.
        title (str, optional): The title of the histogram. Defaults to "Histogram".
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).

    Returns:
        matplotlib.axes.Axes: The axis object containing the histogram plot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    ## Create plot
    if hue:
        sns.histplot(
            data=data,
            x=column_name,
            ax=ax,
            color=COLOR_MAIN,
            palette=palette,
            hue=hue,
            kde=kde,
        )
    else:
        sns.histplot(data=data, x=column_name, ax=ax, color=COLOR_MAIN, kde=kde)

    ## Add title
    ax.set_title(label=title, fontsize=16)
    return ax


def plot_distribution_and_box(
    data,
    column_name: str,
    title: str = "Count and Boxplot",
    ax=None,
    figsize=(10, 5),
    width_ratios=[3, 1.25],
):
    """
    Plots the distribution and boxplot of a numerical column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the numerical column to plot.
        title (str, optional): The title of the plot. Defaults to "Count and Boxplot".
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        figsize (tuple, optional): The figure size. Defaults to (10, 5).
        width_ratios (list, optional): The width ratios of the subplots. Defaults to [3, 1.25].
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)
    assert column_name in data.select_dtypes(include=np.number).columns

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(
        figsize=figsize, ncols=2, gridspec_kw={"width_ratios": width_ratios}
    )
    histplot(
        data=data,
        column_name=column_name,
        title="",
        ax=ax[0],
    )
    boxplot(
        data=data,
        column_name=column_name,
        title="",
        ax=ax[1],
    )
    fig.suptitle(title, fontsize=16)


def plot_distribution_and_ratio(
    data,
    ratio: pd.Series,
    column_name: str,
    hue: str,
    title: str = "Distribution and Ratio",
    ax=None,
    figsize=(10, 5),
    width_ratios=[3, 1.25],
    horizontal: bool = False,
    label_rotation: int = 0,
):
    """
    Plot the distribution and ratio of a categorical variable.

    Parameters:
    - data: The DataFrame containing the data.
    - ratio: The ratio of the categories.
    - column_name: The name of the categorical column.
    - hue: The column to use for grouping the data.
    - title: The title of the plot (default: "Distribution and Ratio").
    - ax: The matplotlib axes object to plot on (default: None).
    - figsize: The figure size (default: (10, 5)).
    - width_ratios: The width ratios of the subplots (default: [3, 1.25]).
    - horizontal: Whether to plot the bars horizontally (default: False).
    - label_rotation: The rotation angle of the tick labels (default: 0).
    """
    fig, ax = plt.subplots(
        figsize=figsize, nrows=1, ncols=2, gridspec_kw={"width_ratios": width_ratios}
    )
    countplot(
        data=data,
        column_name=column_name,
        hue=hue,
        title="Distribution",
        bar_labels=True,
        ax=ax.flatten()[0],
        horizontal=horizontal,
    )
    if horizontal:
        sns.barplot(
            y=ratio.index,
            x=ratio.values,
            color=COLOR_MAIN,
            ax=ax.flatten()[1],
        )
    else:
        sns.barplot(
            x=ratio.index,
            y=ratio.values,
            color=COLOR_MAIN,
            ax=ax.flatten()[1],
        )
    ax[1].set_title("Ratio")

    if label_rotation:
        if horizontal:
            for t1, t2 in zip(ax[0].get_yticklabels(), ax[1].get_yticklabels()):
                t1.set_rotation(45)
                t2.set_rotation(45)
        else:
            for t1, t2 in zip(ax[0].get_xticklabels(), ax[1].get_xticklabels()):
                t1.set_rotation(45)
                t2.set_rotation(45)


def plot_correlation_heatmap(df, title='Correlation Heatmap', figsize=(15, 10), font_scale=0.8, 
                             cmap_start=230, cmap_end=20, annot=True, fmt=".2f", vmin=-1, vmax=1, cbar_shrink=0.5):
    
    """
    Plots a correlation heatmap for the given DataFrame.

    Parameters:
    - df : pandas.DataFrame
        The DataFrame for which to compute and plot the correlation matrix.
    - title : str, optional
        The title of the heatmap. Defaults to 'Correlation Heatmap'.
    - figsize : tuple, optional
        The size of the figure (width, height) in inches. Defaults to (15, 10).
    - font_scale : float, optional
        Font scale for the context (larger or smaller). Defaults to 0.8.
    - cmap_start, cmap_end : int, optional
        Start and end points for the colormap. Defaults to 230 and 20, respectively.
    - annot : bool, optional
        If True, write the data value in each cell. Defaults to True.
    - fmt : str, optional
        String formatting code to use when adding annotations. Defaults to ".2f".
    - vmin, vmax : float, optional
        Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments. Defaults to -1 and 1, respectively.
    - cbar_shrink : float, optional
        Shrink factor for the color bar. Defaults to 0.5.
    """
    
    # Setting the context for the plot
    sns.set_context('talk', font_scale=font_scale)
    
    # Create the matplotlib figure and axes
    f, ax = plt.subplots(figsize=figsize)
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(cmap_start, cmap_end, as_cmap=True)
    
    # Calculate the correlation matrix
    corr = df.corr()
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, annot=annot, fmt=fmt, vmin=vmin, vmax=vmax,
                square=True, linewidths=.5, cbar_kws={"shrink": cbar_shrink}, ax=ax)
    
    # Setting the title of the heatmap
    plt.title(title)
    
    plt.show()



def plot_shap_values(model, X, explainer=None, feature_names=None, plot_size=(7, 5)):
    """
    Plots the SHAP values for a given model and test data.

    Parameters:
    model (object): The trained model object.
    x_train (array-like): The training data used to train the model.
    x_test (array-like): The test data for which SHAP values will be computed and plotted.

    Returns:
    None
    """
    if explainer == "linear":
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, plot_size=plot_size, feature_names=feature_names)



def test_model(model, X_train, X_test, y_train, y_test, logger, average='binary'):
    """
    Evaluate the performance of a machine learning model on training and test data,
    log the results, and return them.

    Parameters:
        model: The trained machine learning model.
        X_train: The input features of the training data.
        X_test: The input features of the test data.
        y_train: The target labels of the training data.
        y_test: The target labels of the test data.
        logger: The logger object for logging the evaluation results.
        average: The averaging method for multi-class classification. Defaults to 'binary'.

    Returns:
        A dictionary containing model performance metrics for both training and test sets.
    """
    # Predictions
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # Training metrics
    metrics_train = {
        'Accuracy': accuracy_score(y_train, pred_train),
        'Precision': precision_score(y_train, pred_train, average=average, zero_division=0),
        'Recall': recall_score(y_train, pred_train, average=average),
        'F1 Score': f1_score(y_train, pred_train, average=average)
    }
    
    # Test metrics
    metrics_test = {
        'Accuracy': accuracy_score(y_test, pred_test),
        'Precision': precision_score(y_test, pred_test, average=average, zero_division=0),
        'Recall': recall_score(y_test, pred_test, average=average),
        'F1 Score': f1_score(y_test, pred_test, average=average)
    }
    
    # Logging
    logger.info("Training scores:")
    for metric, value in metrics_train.items():
        logger.info(f"    - {metric}: {value:.3f}")
        
    logger.info("Test scores:")
    for metric, value in metrics_test.items():
        logger.info(f"    - {metric}: {value:.3f}")

    # Return metrics for further use
    return {'train': metrics_train, 'test': metrics_test}


def get_column_types(df, verbose=True):
    """
    Segregates the columns of a given DataFrame into numerical, categorical, and other types based on their data types.

    Parameters:
    - df : pandas.DataFrame
        The DataFrame from which to segregate the columns.
    - verbose : bool, optional
        If True, prints the lists of numerical and categorical columns.

    Returns:
    - dict: A dictionary with keys 'numerical', 'categorical', and 'other', each containing a list of column names.
    """
    # Using select_dtypes to efficiently segregate column types
    numerical_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Assuming 'other' as non-numerical and non-categorical (boolean, datetime, etc.)
    other_columns = df.select_dtypes(exclude=['int64', 'float64', 'int32', 'float32', 'object', 'category']).columns.tolist()
    
    if verbose:
        print(f'Numerical columns are: {numerical_columns}')
        print(f'Categorical columns are: {categorical_columns}')
        if other_columns:
            print(f'Other columns are: {other_columns}')
    
    return {'numerical': numerical_columns, 'categorical': categorical_columns, 'other': other_columns}


def print_col(columns, chunk_size=5, delimiter=', '):
    """
    Prints elements of a list in chunks, with each chunk containing up to chunk_size elements.

    Parameters:
    - columns: List of strings to be printed.
    - chunk_size: Maximum number of items in each chunk (default is 5).
    - delimiter: String to join items in the chunk (default is ', ').
    """
    for i in range(0, len(columns), chunk_size):
        chunk = columns[i:i + chunk_size]
        print(delimiter.join(chunk))


def report_missing_values(data, column_names=None, sort_by='missing_count', ascending=False):
    """
    Reports missing values for the specified columns in a DataFrame.
    
    Parameters:
    - data : pandas.DataFrame
        The DataFrame to analyze.
    - column_names : list, optional
        The list of column names to check for missing values. If None, checks all columns.
    - sort_by : str, optional
        Specifies whether to sort the output by 'missing_count' or 'percentage'. Default is 'missing_count'.
    - ascending : bool, optional
        Determines the sort order. If False, sorts in descending order. Default is False.
        
    Returns:
    - pandas.DataFrame
        A DataFrame containing the missing value report.
    """
    if column_names is None:
        column_names = data.columns
        
    missing_data = pd.DataFrame(data[column_names].isnull().sum(), columns=['missing_count'])
    missing_data['percentage'] = (missing_data['missing_count'] / len(data)) * 100
    missing_data = missing_data[missing_data['missing_count'] > 0]
    
    if sort_by not in ['missing_count', 'percentage']:
        raise ValueError("sort_by must be 'missing_count' or 'percentage'")
    
    missing_data = missing_data.sort_values(by=sort_by, ascending=ascending).reset_index().rename(columns={'index': 'column'})
    
    return missing_data



def plot_missing_values(missing_df, threshold=50, figsize=(20, 8)):
    """
    Plots the percentage of missing values by column, highlighting columns with a high percentage of missing values.

    Parameters:
    - missing_df: DataFrame
        A DataFrame containing columns 'Column', 'Missing_count', and 'Percentage' representing missing data information.
    - threshold: float, optional
        The percentage threshold above which columns are highlighted. Default is 50%.
    - figsize: tuple, optional
        The size of the figure to be plotted. Default is (20, 8).
    """
    # Define colors based on the percentage threshold
    colors = ['#d9534f' if x > threshold else '#5bc0de' for x in missing_df['Percentage']]
    
    plt.figure(figsize=figsize)
    barplot = sns.barplot(x='Percentage', y='Column', data=missing_df, palette=colors)
    plt.title('Percentage of Missing Values by Column')
    plt.xlabel('Percentage')
    plt.ylabel('Column')
    
    # Drawing a horizontal line at the threshold percentage
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'{threshold}% Missing Values')
    plt.legend()

    # Annotate each bar with the percentage of missing values
    for p in barplot.patches:
        width = p.get_width()
        plt.text(2 + width, p.get_y() + p.get_height() / 2,
                 '{:1.2f}%'.format(width), ha='left', va='center')
    
    plt.tight_layout()
    plt.show()


def cramers_v(x, y):
    """
    Calculate Cramér's V statistic for two categorical series.
    
    Parameters:
    - x, y: pandas.Series
        Two categorical series to compute the Cramér's V statistic.
    
    Returns:
    - float
        The Cramér's V statistic.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    rcorr = r - ((r-1)**2) / (n-1)
    kcorr = k - ((k-1)**2) / (n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def compute_cramers_v_matrix(df, categorical_columns):
    """
    Compute a matrix of Cramér's V statistics for each pair of categorical columns in the DataFrame.
    
    Parameters:
    - df: pandas.DataFrame
        DataFrame containing the data.
    - categorical_columns: list
        List of column names to be considered for Cramér's V computation.
    
    Returns:
    - pandas.DataFrame
        A DataFrame representing the Cramér's V matrix.
    """
    cramers_v_matrix = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))),
                                    index=categorical_columns, columns=categorical_columns)
    
    for col1 in categorical_columns:
        for col2 in categorical_columns:
            cramers_v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
    
    return cramers_v_matrix


def plot_cramers_v_matrix(cramers_v_matrix, figsize=(12, 10)):
    """
    Plot a heatmap of Cramér's V correlation matrix.
    
    Parameters:
    - cramers_v_matrix: pandas.DataFrame
        A DataFrame containing the Cramér's V values to plot.
    - figsize: tuple, optional
        The size of the figure to be plotted. Default is (12, 10).
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cramers_v_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'shrink': .82})
    plt.title('Cramér\'s V Correlation between Categorical Variables')
    plt.show()


def calculate_pointbiserial_correlation(df, categorical_columns, target_variable):
    """
    Calculates the point-biserial correlation between binary target variable and categorical variables.
    
    Parameters:
    - df : pandas.DataFrame
        The DataFrame containing the data.
    - categorical_columns : list
        List of names of the categorical columns.
    - target_variable : str
        The name of the binary target variable in the DataFrame.
        
    Returns:
    - pandas.DataFrame
        A DataFrame with columns ['Category', 'Correlation', 'P-value'], sorted by 'Correlation'.
    """
    correlation_results = []

    for column in categorical_columns:
        # Create dummy variables for each categorical column
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
        
        # Calculate point-biserial correlation for each dummy variable
        for dummy in dummies:
            corr, p_value = pointbiserialr(df[target_variable], dummies[dummy])
            correlation_results.append((dummy, corr, p_value))
    
    # Create DataFrame from results and sort
    correlation_df = pd.DataFrame(correlation_results, columns=['Category', 'Correlation', 'P-value'])
    correlation_df = correlation_df.sort_values('Correlation', ascending=False)
    
    return correlation_df



def fit_logistic_regression_for_categoricals(df, categorical_columns, target_variable):
    """
    Fits a logistic regression model for each categorical column against a binary target variable
    and prints the model summary.
    
    Parameters:
    - df : pandas.DataFrame
        The DataFrame containing the data.
    - categorical_columns : list
        List of names of the categorical columns.
    - target_variable : str
        The name of the binary target variable in the DataFrame.
    """
    for column in categorical_columns:
        # Create dummy variables for the categorical column
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
        
        # Add a constant to the predictor variable set
        X = sm.add_constant(dummies)
        
        # Fit the logistic regression model
        logit_model = sm.Logit(df[target_variable], X).fit(disp=0)
        
        # Print the summary of the model
        print(f"Logistic Regression Model Summary for {column}:\n")
        print(logit_model.summary())
        print("\n" + "#" * 50 + "\n")  # Separator for summaries



def identify_high_corr_columns(df, threshold=0.8):
    """
    Identifies columns in the DataFrame that have a correlation higher than the specified threshold.

    Parameters:
    - df : pandas.DataFrame
        The DataFrame to analyze.
    - threshold : float, optional
        The correlation threshold to consider as 'high'. Defaults to 0.8.

    Returns:
    - list of tuples
        A list containing tuples of column pairs with correlation above the threshold, and their correlation values.
    """
    # Compute the correlation matrix and take the absolute value
    corr_matrix = df.corr().abs()
    
    # Create a boolean mask for the upper triangle of the matrix
    upper_triangle_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    
    # Apply the mask to the correlation matrix to isolate the upper triangle
    upper = corr_matrix.where(upper_triangle_mask)
    
    # Find column pairs with correlation greater than the threshold
    high_corr_pairs = [(col, row, corr_matrix.loc[col, row]) 
                       for col in upper.columns 
                       for row in upper.index 
                       if upper.loc[row, col] > threshold]
    
    print(f"There are {len(high_corr_pairs)} pairs of columns with correlation higher than {threshold}.")
    return high_corr_pairs


def identify_outliers(df, numerical_cols):
    """
    Identifies outliers in the specified numerical columns of a DataFrame based on the IQR method.

    Parameters:
    - df : pandas.DataFrame
        The DataFrame to analyze.
    - numerical_cols : list
        A list of strings representing the names of numerical columns to check for outliers.

    Returns:
    - dict
        A dictionary where each key is a column name and each value is a DataFrame containing
        the outlier rows for that column.
    """
    outliers = {}
    
    for column in numerical_cols:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_condition = (df[column] < lower_bound) | (df[column] > upper_bound)
        outliers[column] = df.loc[outliers_condition]
    
    return outliers


def print_outlier_summary(outliers_dict):
    """
    Prints a summary of outliers for each numerical column in a dataset.
    """
    for column, outliers_df in outliers_dict.items():
        if not outliers_df.empty:
            print(f'Outliers in {column}:')
            print(outliers_df.describe(), '\n')
        else:
            print(f'No outliers detected in {column}.')
            


def apply_one_class_svm(df, numerical_cols, nu=0.9, kernel='rbf', gamma='auto', outlier_column_name='OCSVM_Outlier'):
    """
    Applies One-Class SVM to detect outliers in the specified numerical columns of a DataFrame.

    Parameters:
    - df : pandas.DataFrame
        The DataFrame containing scaled numerical data.
    - numerical_cols : list
        A list of strings representing the names of the numerical columns to analyze.
    - nu : float, optional
        An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
        Must be between 0 and 1. Defaults to 0.9.
    - kernel : str, optional
        Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid',
        'precomputed', or a callable. If none is given, 'rbf' will be used. Defaults to 'rbf'.
    - gamma : str, optional
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. If 'auto', 1/n_features will be used. Defaults to 'auto'.
    - outlier_column_name : str, optional
        The name of the new column to be added to the DataFrame indicating outlier status. Defaults to 'OCSVM_Outlier'.

    Returns:
    - pandas.DataFrame
        The original DataFrame with an additional column indicating outlier status.
    """
    # Ensure numerical_cols is a list to prevent indexing errors
    if not isinstance(numerical_cols, list):
        raise ValueError("numerical_cols must be a list of column names.")

    # Fit the One-Class SVM model
    ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    ocsvm_outliers = ocsvm.fit_predict(df[numerical_cols])

    # Append the outlier detection results to the DataFrame
    df[outlier_column_name] = ocsvm_outliers
    
    return df
            


def elbow_method(data, max_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=0):
    """
    Applies the Elbow Method to determine the optimal number of clusters for K-Means clustering.

    Parameters:
    - data : array-like, shape (n_samples, n_features)
        The dataset to cluster.
    - max_clusters : int, optional
        The maximum number of clusters to try. Defaults to 10.
    - init : str, optional
        Method for initialization. Defaults to 'k-means++'.
    - max_iter : int, optional
        Maximum number of iterations of the K-Means algorithm for a single run. Defaults to 300.
    - n_init : int, optional
        Number of time the K-Means algorithm will be run with different centroid seeds. Defaults to 10.
    - random_state : int, optional
        Determines random number generation for centroid initialization. Use an int for reproducibility. Defaults to 0.
    
    Returns:
    - list
        A list containing the WCSS values for each number of clusters tested.
    """
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init=init, max_iter=max_iter, n_init=n_init, random_state=random_state)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    # Plotting the results
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='-')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()
    
    return wcss




def visualize_clusters(data, n_clusters=4, cluster_labels=None, title='Clusters of applicants', feature_names=('PCA Feature 1', 'PCA Feature 2')):
    """
    Visualizes the clusters formed by K-Means clustering on PCA-transformed data.

    Parameters:
    - data : array-like, shape (n_samples, n_features)
        The PCA-transformed dataset.
    - n_clusters : int, optional
        The number of clusters. Defaults to 4.
    - cluster_labels : array-like, shape (n_samples,), optional
        The cluster labels for each point. If None, KMeans will be used to compute this.
    - title : str, optional
        The title of the plot. Defaults to 'Clusters of applicants'.
    - feature_names : tuple of str, optional
        The names of the PCA features to label the axes. Defaults to ('PCA Feature 1', 'PCA Feature 2').
    
    """
    if cluster_labels is None:
        # Fit KMeans and predict cluster labels if not provided
        kmeans_optimal = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        cluster_labels = kmeans_optimal.fit_predict(data)
    
    # Define colors for each cluster
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange'][:n_clusters]

    plt.figure(figsize=(10, 8))
    for i, color in enumerate(colors):
        plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], s=100, c=color, label=f'Cluster {i+1}')
    
    plt.title(title)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.show()
            
            


def calculate_vif(data, numerical_columns):
    """
    Calculates Variance Inflation Factor (VIF) for each specified numerical column in the DataFrame.
    
    Parameters:
    - data : pandas.DataFrame
        The DataFrame containing the data.
    - numerical_columns : list
        A list of strings representing the names of the numerical columns to analyze.
        
    Returns:
    - pandas.DataFrame
        A DataFrame containing the VIF values for each specified numerical column.
    """
    # Ensure numerical_columns is a list to prevent indexing errors
    if not isinstance(numerical_columns, list):
        raise ValueError("numerical_columns must be a list of column names.")
    
    # Add a constant to the dataset for VIF computation
    X = add_constant(data[numerical_columns])
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    
    return vif_data




def train_and_plot_feature_importances(data, target_col, categorical_columns, test_size=0.2, n_estimators=100, random_state=42, top_n=20):
    """
    Encodes categorical variables, splits the data, trains a Random Forest classifier, and plots the top N feature importances.

    Parameters:
    - data : pandas.DataFrame
        The DataFrame containing the data.
    - target_col : str
        The name of the target variable column.
    - categorical_columns : list
        A list of names of the categorical columns to be encoded.
    - test_size : float, optional
        The proportion of the dataset to include in the test split. Defaults to 0.2.
    - n_estimators : int, optional
        The number of trees in the forest. Defaults to 100.
    - random_state : int, optional
        Controls both the randomness of the bootstrapping of the samples used when building trees 
        (if `bootstrap=True`) and the sampling of the features to consider when looking for the best split at each node 
        (if `max_features < n_features`). Defaults to 42.
    - top_n : int, optional
        The number of top features to plot. Defaults to 20.
    """
    # Encode categorical variables
    df_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Prepare the feature matrix X and target vector y
    X = df_encoded.drop([target_col], axis=1)
    y = df_encoded[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train the Random Forest classifier
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_clf.fit(X_train, y_train)
    
    # Calculate feature importances and sort them
    importances = rf_clf.feature_importances_
    feature_names = X_train.columns
    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importances_df_sorted = importances_df.sort_values(by='Importance', ascending=False)
    
    # Plot the top N feature importances
    plt.figure(figsize=(10, 8))
    plt.barh(importances_df_sorted['Feature'][:top_n], importances_df_sorted['Importance'][:top_n], color='skyblue')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title('Top 20 Feature Importances', fontsize=16)
    plt.gca().invert_yaxis()
    plt.show()
