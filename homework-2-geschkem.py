import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    __Maddie Geschke__

    # Coding Homework 2: Text Classification

    ## Assignment Setup

    This assignment assumes you have completed the following setup tasks in class:

    #### 1. Installed Python's Natural Language Toolkit (NLTK)
    #### 2. Downloaded the NLTK stopwords list
    #### 3. Loaded the Glassdooor jobs data as a Pandas DataFrame, with the variable `glassdoor_df`
    #### 4. Created a new column called `label` with values as described in the setup slides
    #### 5. Created a new Pandas DataFrame called `glassdoor_df_binary` that represents only rows labeled as 'data science' or 'data analytics'

    Enter your setup code in the Python cell directly below.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import shap
    import sklearn as skode
    import nltk
    #nltk.download('stopwords')
    import sklearn
    import seaborn as sns
    nltk.download('punkt_tab')
    return nltk, pd


@app.cell
def _(pd):
    #snippet 1
    glassdoor_df = pd.read_csv('glassdoor_jobs.csv', index_col=0)

    def label_jobs(desc):
        job = desc.lower()
        da = False
        ds = False 
        if 'data science' in job:
            ds = True
        if 'data analytics' in job:
            da = True
        if da and ds: 
            return 'both'
        if da: 
            return 'da'
        if ds:
            return 'ds'
        return 'neither'

    glassdoor_df['label'] = glassdoor_df['extracted_text'].apply(label_jobs)
    glassdoor_df_binary = glassdoor_df.loc[glassdoor_df['label'].isin(['da','ds'])]
    glassdoor_df_binary['label'].value_counts()
    return (glassdoor_df_binary,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Natural Language Processing

    It also assumed that you will follow along in class to complete the following Natural Language Processing (NLP) steps

    1. Converted the job description column text from raw string to a list of tokens (using nltk.word_tokenize)
    2. Converted all alphabetic characters in the text to lowercase
    3. Removed from the token list any tokens that are in the nltk stopwords list (`stopwords.words('english')`)
    4. Removed all non-word tokens from the list (such as punctuation, numerical digits, symbols, links to media files, etc.)
    6. Converted your list of token lists to a list of term-frequency dictionaries or Counters (as we did in the TF-IDF activity)

    Enter your NLP code in the Python cell directly below.
    """)
    return


@app.cell
def _(glassdoor_df_binary, nltk):
    from collections import Counter

    def remove_non_alpha(token):
        new_token = "".join([i for i in token if i.isalpha()]) 
        return new_token
    
    # convert column to list of lowercase strings
    jobs = glassdoor_df_binary['extracted_text'].str.lower().to_list()

    # lowercase
    lowered = [i.lower() for i in jobs]

    # tokenize
    tokens = [nltk.tokenize.word_tokenize(i) for i in lowered]
    
    tokens_alpha = []

    for i in tokens:
        alpha_only = [remove_non_alpha(t) for t in i]
        tokens_alpha.append([z for z in alpha_only if z !=''])

    # make term counters for each separate clue
    job_counters = [Counter(i) for i in tokens_alpha]
    job_counters[0]
    return (job_counters,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Text Classification

    Lastly, you will adapt the Github Sandbox code to complete the following supervised machine learning steps:

    1. Use `scikit-learn` to convert the list of term-frequency dictionaries to document-term matrix under the variable name `X`
    2. Use `scikit-learn` to apply the TF-IDF transformation to `X` and store the result under the variable name `V`
    3. Convert the "label" column in `glassdoor_df_binary` to a Python `list` under the variable name `y`
    4. Use the `scikit-learn` library to perform a train/test split on the `V` and `y` variables and output the variables `V_train`, `V_test`, `y_train`, `y_test`. Train on about 75% of the data.
    5. Train a logistic regression model (from the `scikit-learn` library) using `V_train` and `y_train`
    6. Evaluate model performance, create a `DataFrame` of term coefficients, and explore the results.
    """)
    return


@app.cell
def _(glassdoor_df_binary, job_counters):
    #snippet 3 - code for training model 

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    v = DictVectorizer(sparse=False)
    X = v.fit_transform(job_counters)
    transformer = TfidfTransformer()
    V = transformer.fit_transform(X).toarray()

    y = glassdoor_df_binary['label'].tolist()
    V_train, V_test, y_train, y_test = train_test_split(V, y, test_size=0.2, random_state=7712)

    logit = LogisticRegression()
    trained = logit.fit(V_train, y_train)
    predictions = trained.predict(V_test)
    probs = trained.predict_proba(V_test)
    return predictions, probs, trained, v, y_test


@app.cell
def _(pd, predictions, probs, y_test):
    # snippet 4 - turning above info into data frame 
    df_results = pd.DataFrame()
    df_results['prediction'] = predictions
    df_results['prob_0'] = [i[0] for i in probs]
    df_results['prob_1'] = [i[1] for i in probs]
    df_results['label'] = y_test
    df_results['correct'] = df_results['prediction'] == df_results['label']
    df_results

    return (df_results,)


@app.cell
def _(df_results):
    # finding accuracy 
    float(df_results['correct'].sum()/df_results.shape[0] * 100)
    return


@app.cell
def _(pd, trained, v):
    # coefficients? 
    df_coef = pd.DataFrame()
    df_coef['term'] = v.feature_names_
    df_coef['score'] = trained.coef_[0]
    df_coef.sort_values(by='score', ascending=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    In this section, you will write 3-5 sentences addressing each of the questions below. Use markdown as needed to format your responses.

    #### 1. Describe and interpret the performance of your model.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    My model uses logistic regression to predict whether a job posting is listed as Data Analytics (da) or Data Science (ds). To do this, we first are taking these job postings and either classifying them as DA positions (contains data analytics), DS positions (contains Data Science), both (contains both data analytics or data science), or neither (contains neither data analytics or data science). We are then web scraping job postings that were listed in these categories on Glassdoor, and tokenizes the job descriptions to only contain one work alphebetic tokens. We also used TF-IDF to give weight to certain words and phrases that may have lower frequency, but when they are used they have a high likelihood of predicting a certain outcome. It uses 80% of the data to train the model and 20% of the data to test the data. We test the accuracy of the model with the code chunk that starts with the 'float' code, and we see from that that we have about 83% accuracy, which is pretty good!
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 2. Discuss the term coefficients for each label. What were the most important predictors of each type category, and why do you think they were strong predictors?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The term coefficients for each label involve us looking at the final table we see in the code blocks above. Big picture, terms with stronger positive values are more likely to predict Data Science positions, while terms with stronger negative values are more likely to predict Data Analytics positions. Terms with coefficients that are closer to 0 are likely to be used in either job description. We see the values with the largest positive values (like science 4.24 and ai at 2.675) are the strongest indicator words for the ob listing being a Data Science position. Looking at the very end of the table we can see the terms with the strongest negative values (like analytics -2.57 and sales -1.54) being the strongest indicator terms for Data Analytics. We see that 'science' is the strongest predictor for DS and 'analytics' is the strongest predictor for DA, which maeks sense and is a reality check that our model is working correctly. Other than these obvious predictor words, strong terms for DS are 'ai' and 'care', which is interesting. AI being a high word right now makes sense because that is an area that a lot of companies are looking for, but I would love to look closer as to why 'care' is such a high word for DS. For DA, other than 'analytics' being a high frequency word we have 'sales' and 'improvement', which does make sense to me because I think there are a lot of positions in sales that want to use and incorporate more analytical skills, and improvement is another area that people like to incorporate analytics.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 3. Areas for Expansion and Further Inquiry

    This section can be notes/bullet points. In the markdown cell below, provide at least three areas for how this assignment could be deepened, expanded, or extended. Be prepared to share your ideas in class.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - more context as to why 'care' appeared so high for data science, perhaps incorporating bigram detection
    - incorporating bigram detection in general, for da interested to see 'project' and 'management' together
    - taking out more common words like 'you', 'your', etc, we see a lot of these in the data science position
    - i observe a lot of meditcal-esque terms in the data science part of the coefficient table, would love to look closer at specifically healthcare and look as to why they want more data scientists rather than analysts
    - would love to alter the code to include scientist rather than just science in the search term
    """)
    return


if __name__ == "__main__":
    app.run()
