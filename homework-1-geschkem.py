import marimo

__generated_with = "0.19.6"
app = marimo.App(width="columns")


@app.cell
def _(mo):
    mo.md(r"""
    # Homework 1 Instructions

    This homework assignment is a worksheet that each student will complete by filling in a Marimo notebook in Python. Start with the assignment template (Github Classroom link to be shared on Canvas) and follow the instructions for each question. As you go, you will typically be asked to create a Python or markdown cell with code or markdown in it that performs one or two purposes:

    - Serves as a primer/refresher on prior Python concepts, especially on building your understanding of how numpy and pandas differ from standard Python, as well as building or rebuilding familiarity with Pandas-based operations
    - Helps you become increasingly comfortable with writing Python using a Marimo notebook.

    __Due Date:__ 3 p.m. Wednesday, February 11, 2026

    __Reminder:__ Rename the file `homework-1-yourlastname.py` (for example, mine would be `homework-1-lavin.py`)

    __How to Turn It In:__ Convert Marimo to HTML, upload HTML file on Canvas

    __Notes:__

    1. The Marimo competencies are all taken from the Marimo tutorial, specifically items under the commands `marimo tutorial intro`, `marimo tutorial markdown`, `marimo tutorial plots`, `marimo tutorial for-jupyter-users `, and `marimo tutorial layout`.

    2. For this assignment, you are allowed to use Google and help sites like stackoverflow. You are not permitted get answers from another person or to work with other classmates on the assignment. (Talking about it is allowed, but directly collaborating is not.)

    3. Using AI tools such as chatGPT, Claude, etc. is permitted for this assignment, but how you use AI is restricted as follows:

    - You should make a genuine effort on your own before turning to AI for outside help.
    - You are permitted to treat AI like a consultant, not to just copy/paste the questions. Make an effort to understand and scrutinize any and all solutions suggested by AI. Quality of responses will vary.
    - If you get help from AI, you must include a statement in your assignment submission describing (a) which AI agent you used; (b) a list of all prompts you entered; and (c) a summary explaining the aspects of your code you used AI for help with.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import shap
    import sklearn as sk
    return mo, pd, shap, sk


@app.cell
def _():
    import numpy as np
    # shap requires 2.3 or lower
    np.__version__
    return (np,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Background Information
    The data file `vehicles.csv` contains information from https://www.fueleconomy.gov/feg/ws/index.shtml#vehicle. It has 84 columns representing a wide range of vehicle type and fuel economy information.

    The data file `penguins.csv` contains data used in https://github.com/christophM/interpretable-ml-book/blob/master/scripts/shap/shap-notebook.ipynb.

    __Dataset Source__: Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data. R package version 0.1.0. <https://allisonhorst.github.io/palmerpenguins/>. doi:10.5281/zenodo.3960218. From Gorman et al. (2014):

    <blockquote>Individuals interested in using these data are expected to follow the US LTER Network’s Data Access Policy, Requirements and Use Agreement.”<br/> <a href='https://lternet.edu/data-access-policy/'>https://lternet.edu/data-access-policy/</a> </blockquote>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Read CSV files and store as DataFrames

    - Load `vehicles.csv` using the `read_csv` method and store the resulting `DataFrame` as a variable called `mpg_df`
    - Load `penguins.csv` using the `read_csv` method and store the resulting `DataFrame` as a variable called `penguins`
    """)
    return


@app.cell
def _(pd):
    mpg_df = pd.read_csv("vehicles.csv") #loading mpg

    penguins = pd.read_csv("penguins.csv") #loading penguins
    return mpg_df, penguins


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Subset by columns / Complete other pre-processing steps

    - Remove any rows that contain missing values (NaN) from 'mpg_df` and `penguins` and store the results as 'mpg_df_no_na` and `penguins_no_na`
    - Create a subset of `mpg_df` called `mpg_df_subset` that includes every row, but only the following columns: 'model', 'make', 'year', 'city08', 'highway08', 'fuelType', and 'baseModel'
    """)
    return


@app.cell
def _(mpg_df, penguins):
    mpg_df_no_na = mpg_df.dropna() #dropping na's
    penguins_no_na = penguins.dropna() #dropping na's

    mpg_df_subset = mpg_df[["model", "make", "year", "city08", "highway08", "fuelType", "baseModel"]]
    return mpg_df_subset, penguins_no_na


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Rename

    - Working with `mpg_df_subset`, rename the 'city08' column to `avg_city_mpg` and the 'highway08' column to `avg_highway_mpg`
    """)
    return


@app.cell
def _(mpg_df_subset):
    mpg_df_subset.rename(columns={"city08": "avg_city_mpg", "highway08":"avg_highway_mpg"}, inplace=True)
    mpg_df_subset
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Check and convert data types (`dtypes`, `astype`)

    - Look at the datatype for each column. If necessary, convert `avg_city_mpg` and `avg_highway_mpg` to float-point data
    - Use `mo.vstack` to display: (1) each column name with the datatype for each column; (2) the first ten rows of `df_subset` using the `head()` method
    """)
    return


@app.cell
def _(mo, mpg_df_subset):
    mpg_df_subset.dtypes

    mpg_df_subset["avg_city_mpg"] = (mpg_df_subset["avg_city_mpg"].astype(float))
    mpg_df_subset["avg_highway_mpg"] = (mpg_df_subset["avg_highway_mpg"].astype(float))

    mo.vstack([mo.md("### Column Names and Data Types"), mpg_df_subset.dtypes, mo.md("### First 10 rows of mpg_df_subset"), mpg_df_subset.head(10)])

    #AI Disclosure: I used ChatGPT to see how to use mo.vstack and format it. My prompt was "How do I format a mo.vstack call in marimo to display specific columns in a dataframe?" It's response was very generic and explained the order in which to write things in the code.
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Multi-column sort and custom sort order

    - Sort the columns of `mpg_df_subset` by `avg_city_mpg` and subsort by `avg_highway_mpg` (largest values at top) and display the first ten rows after sorting
    """)
    return


@app.cell
def _(mpg_df_subset):
    mpg_df_subset.sort_values(by=["avg_city_mpg", "avg_highway_mpg"], ascending=[False, False]).head(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Filter on a numerical range, one or more text values, and/or multiple conditions using `isin()`

    In the cell below this question, add Python that:

    - Uses `mo.ui.tabs()` to create a tabbed interface displaying three tabs with different labels and content
    - Label the first tab 'Best MPGs' and set it to display a DataFrame containing all rows in the data where the value for `avg_city_mpg` is greater than 120 (~66 rows)
    - Label the second tab 'Nissan' and set it to display a DataFrame containing all rows in the data where the vehicle in question is a Nissan
    - Label the third tab 'NTH' and set it to display a DataFrame containing all rows in the data where the vehicle is any of Nissan, Toyota, or Honda
    """)
    return


@app.cell
def _(mo, mpg_df_subset):
    mo.ui.tabs(
        {
            "Best MPGs": mpg_df_subset[mpg_df_subset["avg_city_mpg"]> 120], 
            "Nissan": mpg_df_subset[mpg_df_subset["make"] == "Nissan"], 
            "NTH": mpg_df_subset[mpg_df_subset["make"].isin(["Nissan", "Toyota", "Honda"])],
        }
    )

    # AI Disclosure: Similar to my previous usage, I used ChatGPT to see how to format the mo.ui.tabs function. My prompt was "How do I use the mo.ui.tabs function in marimo?". It's response was that it is a pattern of labels and content, and I used that to format my table.
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Create a new column; calculate a correlation between two columns

    - Working with `mpg_df_subset`, calculate and display the Pearson correlation of `avg_city_mpg` and `avg_highway_mpg`
    """)
    return


@app.cell
def _(mpg_df_subset):
    mpg_df_subset["avg_city_mpg"].corr(mpg_df_subset["avg_highway_mpg"], method="pearson")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Create a new column with a mathematical operation

    - Create a new column in `mpg_df_subset` called `euclidean_distance` that represents `avg_city_mpg` * `avg_highway_mpg`, then calculates the square root of that product
    - Create a new variable called `df_best_top_200` representing the rows in `mpg_df_subset` with the highest 200 `euclidean_distance` scores
    - Display `df_best_top_200` in your code cell
    """)
    return


@app.cell
def _(mpg_df_subset, np):
    mpg_df_subset["euclidean_distance"]=np.sqrt(mpg_df_subset["avg_city_mpg"] * mpg_df_subset["avg_highway_mpg"])
    df_best_top_200 = mpg_df_subset.nlargest(200, "euclidean_distance")
    df_best_top_200
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Write a custom function and run it using apply()

    - Create another subset of `mpg_df_subset` called `mpg_tesla` that includes only rows that represent Teslas
    - Using your own function and the `apply` method, create a new column in `mpg_tesla` called `model_abbrev` that represents the letter or number of the Tesla model (e.g. "Model A" becomes "A", "Model 1" becomes "1")
    - Display `mpg_tesla` in your code cell
    """)
    return


@app.cell
def _(mpg_df_subset):
    mpg_tesla = mpg_df_subset[mpg_df_subset["make"] == "Tesla"].copy()

    def get_model_abbrev(model_name):
        parts = model_name.split () 
        if len(parts) > 1:
            return parts[1]
        else: 
            return ""

    mpg_tesla["model_abbrev"] = mpg_tesla["model"].apply(get_model_abbrev)

    mpg_tesla
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Group data and calculate aggregated statistics

    - Display the data in `mpg_df_subset` so that it shows each unique vehicle make from the `make` column, as well as the mean of all `avg_city_mpg` values matching that `make`. (Hint: You will likely use `groupby` with `mean` for this question)
    """)
    return


@app.cell
def _(mpg_df_subset):
    make_avg_city = mpg_df_subset.groupby("make")["avg_city_mpg"].mean().reset_index()
    make_avg_city
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Display a plot

    - Display a scatter plot of `avg_city_mpg` by `avg_highway_mpg`
    """)
    return


@app.cell
def _(mpg_df_subset):
    import matplotlib.pyplot as plt

    mpg_df_subset.plot.scatter(
        x="avg_city_mpg",
        y="avg_highway_mpg",
        title="city mpg vs highway mpg"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Display another plot

    - Display a line plot of `euclidean_distance` by year
    """)
    return


@app.cell
def _(mpg_df_subset):
    mpg_df_subset.plot.line(
        x="year",
        y="euclidean_distance"
    )

    #did a line plot, very aware that this does not look good. 
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Use the SHAP library in Python

    - Copy paste the following code below and run the cell
    - Make sure this runs. We'll discuss it in class.

    ```python
    # convert the categorical "species" column into numeric dummy variables (0/1)
    penguins_dummy = pd.get_dummies(penguins_no_na, columns=["species"], drop_first=True, dtype=int)
    y = (penguins_dummy ["sex"].values == "female").astype(int)
    X = penguins_dummy .drop(["island", "sex", "year", "rowid"], axis=1)
    mod = sk.ensemble.RandomForestClassifier(n_estimators = 5, random_state = 42)
    mod.fit(X,y)

    explainer = shap.Explainer(mod, X)
    shap_values = explainer.shap_values(X)
    ```
    """)
    return


@app.cell
def _(pd, penguins_no_na, shap, sk):
    # convert the categorical "species" column into numeric dummy variables (0/1)
    penguins_dummy = pd.get_dummies(penguins_no_na, columns=["species"], drop_first=True, dtype=int)
    y = (penguins_dummy ["sex"].values == "female").astype(int)
    X = penguins_dummy .drop(["island", "sex", "year", "rowid"], axis=1)
    mod = sk.ensemble.RandomForestClassifier(n_estimators = 5, random_state = 42)
    mod.fit(X,y)

    explainer = shap.Explainer(mod, X)
    shap_values = explainer.shap_values(X)

    #could not download shap
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 14. Display SHAP-specific plots

    - Paste the following code snippets below (one per cell) and run code to display the results for each
    - Make sure these run. We'll discuss them in class.

    ```python
    shap.summary_plot(shap_values[:,:,1], X)
    ```

    ```python
    shap.summary_plot(shap_values[:,:,1], X, plot_type = "bar")
    ```

    ```python
    shap.dependence_plot("body_mass_g", shap_values[:, :, 1], X)
    ```
    """)
    return


@app.cell
def _():
    # your code here 
    return


if __name__ == "__main__":
    app.run()
