import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    import nltk
    import pandas as pd
    import statsmodels.formula.api as smf
    from sklearn.feature_extraction import DictVectorizer
    from nltk.corpus import stopwords
    from collections import Counter

    return Counter, DictVectorizer, mo, nltk, pd, smf, stopwords


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    __Maddie Geschke__

    # Coding Homework 3: This [am_is_are_was_were] Jeopardy!

    ## (Complete This Before Monday's Class) Assignment Setup

    This assignment asumes you have installed Python's Natural Language Toolkit (NLTK) and downloaded both the NLTK stopwords list and the Punkt Tokenizer. Before class on Monday, please complete the following setup steps:

    #### 1. Load the Jeopardy! clues dataset as a Pandas DataFrame; name your variable `clues_df`
    #### 2. Convert the `air_date` column to a Pandas datetime dtype if it isn't already recognized as such
    #### 3. Create a new column in the dataset called `difficulty`. Derive a dififculty of `na`, or 1 to 5 as follows:

    - Single Jeopardy smallest possible \$ value: 1
    - Single Jeopardy largest possible \$ value: 5
    - Double Jeopardy smallest possible \$ value: 1
    - Double Jeopardy largest possible \$ value: 5
    - Final Jeopardy clue, daily doubles, and any clues with bonus points outside the normal scale: `na`

    __Hints:__

    1. At some point, the single and double Jeopardy! clue values were changed from 100-500 and 200-1000 respectively, to 200-1000 and 400-2000 respectively. There are various ways to find and control for this shift in smallest and largest possible values as you normalize to a 1-5 scale.

    2. Daily Doubles are marked `na` because the `value` column gives the contestant's wager. For an extra and __not required__ challenge, try deriving the original $ value where the clue was placed. For the questions with bonus points, something similar may be possible too ...
    """)
    return


@app.cell
def _(pd):
    clues_df = pd.read_csv("master_season1-35.tsv", sep='\t') #Since its a tsv 
    single_double = clues_df.loc[clues_df['round'] != 3]
    single_double_no_dd = single_double.loc[single_double['daily_double'] == 'no'].copy()
    single_double_no_dd['air_date_dt'] = pd.to_datetime(single_double_no_dd['air_date'].copy())
    before = single_double_no_dd.loc[single_double_no_dd['air_date_dt'] <= '2001-11-23'].copy()
    after = single_double_no_dd.loc[single_double_no_dd['air_date_dt'] > '2001-11-23'].copy()

    before['scaled'] = before['value'].copy()
    after['scaled'] = after['value']/2
    rejoined = pd.concat([before, after])

    rejoined['difficulty'] = rejoined['scaled']/rejoined['round']/100

    # this last line of code will only drop one edge case that wasn't captured by everything above 
    final = rejoined.loc[rejoined['scaled'] % 100 == 0]
    return (final,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##  (Complete This Before Monday's Class) Natural Language Processing Steps

    In the code cell below, write a function called `process_text` that executes the following operations using any mix of standard Python, Pandas, and NLTK functionalities:

    1. Accepts a string as an input/parameter
    2. Converts text from raw string to list of tokens (using nltk.word_tokenize)
    3. Convert all alphabetic characters in the text to lowercase
    4. Removes from the token list any tokens that are in the nltk stopwords list (`stopwords.words('english')`)
    5. Removes all non-word tokens from the list (such as punctuation, numerical digits, symbols, links to media files, etc.)
    6. Outputs a list of remaining tokens

    __Notes:__

    This function will be used to convert the text in the `answer` column of `clues_df`. The specifics are identical to the pre-processing steps we did for Homework 2, but this time I'm asking you to encapsulate it in one function.
    """)
    return


@app.cell
def _(Counter, final, nltk, stopwords):
    def remove_non_alpha(text):
        '''
        helper function for non-alphabetical characters
        '''
        return ''.join([char for char in text if char.isalpha()]) 

    def process_text(text): 
        ''' input string and return term-count Counter (dictionary-like object) ... 
            text here is assumed to be one row in a DataFrame
        '''
    
        stops = [i.lower() for i in stopwords.words('english')]
        lowered = text.lower()
        tokens = nltk.tokenize.word_tokenize(lowered)
    
        alpha_only = [remove_non_alpha(t) for t in tokens]
        no_stops = [i for i in alpha_only if i not in stops]
        tokens_alpha = [z for z in no_stops if z !='']
        return Counter(tokens_alpha)

    sample = final.sample(frac=0.1, random_state=217) 
    list_of_counters = sample['answer'].apply(process_text)
    return list_of_counters, sample


@app.cell
def _(DictVectorizer, list_of_counters, pd, sample):
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(list_of_counters)

    full_matrix = pd.DataFrame(X)
    full_matrix.columns = v.feature_names_
    sums = full_matrix.sum()
    top_200 = sums.transpose().sort_values(ascending=False).iloc[0:200].index.to_list()
    matrix = full_matrix[top_200]
    matrix['difficulty'] = sample['difficulty'].to_list()
    matrix_no_na = matrix.dropna()
    matrix_no_na['difficulty'] = matrix_no_na['difficulty'].astype('int64')
    matrix_no_na['sum'] = matrix_no_na.sum(axis=1, numeric_only=True)
    matrix_no_zero_sums = matrix_no_na.loc[matrix_no_na['sum'] > 0].copy()
    mysample_w_sum_column = matrix_no_zero_sums.sample(3500, random_state=44)
    mysample = mysample_w_sum_column.drop('sum', axis=1)

    return (mysample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (In Class Code-Along) Vectorization with Pandas

    In class, we will call the function on the `answer` column with an `apply` statement and output the results for vectorization. We'll then calculate the top 50 most frequent non-stopword terms in the entire `answer` column and output the results as a Pandas DataFrame, with the following structure:

    | index | word_1 | word_2 | ... | word_50 | difficulty|
    |---|---|---|---|---|---|
    |\<row # for first clue\>|\<count of most frequent word in clue 1\>|\<count w2\>|...|\<count w50\>|\<value 1-5\>|

    __Note:__ Most word count values for each clue will be 0 or 1 since the clues tend to be short.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## (In Class Code-Along) Fit a Multinomial Logit Choice Model Using the `statsmodels` Library

    In class, we will also use the `statsmodels` library to fit a multinomial logit choice model on some sample of the data (max size may vary based on your computer's strength, so start small and see what your system can handle). You are evaluating the base hypothesis that specific keywords can predict clue difficulty.

    Fitting a multinomial logit choice model in `statsmodels` looks at lot like using an Ordinary Least Squares (OLS) linear regression model, though the a completely different "best fit" method is being used, and interpreting the results differs significantly. See https://www.statsmodels.org/stable/generated/statsmodels.formula.api.mnlogit.html and https://en.wikipedia.org/wiki/Multinomial_logistic_regression
    """)
    return


@app.cell
def _(mysample, smf):
    formula = "difficulty ~ " + " + ".join([i for i in mysample.columns if i != 'difficulty'])
    m = smf.mnlogit(formula, data=mysample).fit(method='bfgs', maxiter=1000)

    # access coefficients directly with m.params 
    # access p-values directly with m.pvalues
    return (m,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Homework Questions

    ### 1. Results and Interpretation

    Working with `m.summary()` (we'll cover this in class), do your best to interpret the results. For some information on interpreting multinomial logistic regression results, see https://stats.oarc.ucla.edu/stata/output/multinomial-logistic-regression-2/ (Note that some of the information on that site might be specific to the Stata implementation.)
    """)
    return


@app.cell
def _(m):
    m.summary()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interpretation

    This model is showing us the affect of that certain word at predicting what category the clue is in. If the coefficient value is closer to 1, then that word being in the clue means the clue is likely to be in the category. Likewise, if the coefficient value is closer to -1, then the clue is less likely to be in the given category. As the coefficient vales get closer to zero, they are less accurate at predicting what category (difficulty level) the clue is in. The p values show us whether or not the findings are statistically significant, so if the p-values are less than 0.05 then they are statistically significant. Looking closely at the data we do not have too many words that are statistically significant predictors of this category.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2. Comparison with Boettcher's Methods

    This section can be notes/bullet points. In the markdown cell below, describe at least three ways in which this homework assignment resembles the approach outlined in the Boettcher reading, as well as at least three ways in which this homework assignment differs from the approach outlined in the Boettcher reading. What could we do to follow Boettcher's approach more closely? Be prepared to share your ideas in class.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Similarities:
    - They both measured jeapordy questions based on difficulty and if they can predict it
    - they both used text features to make those difficulty level predictions
    - both used logistic regression in some sense

    Differences:
    - boettcher used linguistic features on top of the text features
    - boettcher also did more on the predictive side (train test splits etc)
    - this homework did not use bigram detection
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Areas for Expansion and Further Inquiry

    This section can be notes/bullet points. In the markdown cell below, provide at least three areas for how this assignment could be deepened, expanded, or extended (beyond making it more like what Boettcher did). Be prepared to share your ideas in class.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - add bigram detection to our visualization to see if certain sets of words are more likely to predict a certain difficulty level
    - switch models to account for multicollinearity, see if the predictions significantly change
    - do a time-based analysis, make a psosible split be when the scores changed to see the average difficulty
    - going off previous point, make two separate prediction models trained off the pre change data and the post change data then use the models on the opposite data to see if predictions are consistent
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. Exploratory Visualization

    Create an exploratory data visualization that demonstrates something interesting about the Jeopardy! clues data. You visualization can be based on analyzing the cateogries, the difficulty levels, keywords found in clues, or anything else you like. Display the results in the cell below.
    """)
    return


@app.cell
def _(final):
    # im looking at whether harder questions have longer answers

    import matplotlib.pyplot as plt

    def count_words(x):
        return len(str(x).split())
    
    final['answer_length'] = final['answer'].apply(count_words)

    avg_length_by_difficulty = final.groupby('difficulty')['answer_length'].mean()

    plt.figure()
    plt.bar(avg_length_by_difficulty.index.astype(str),
            avg_length_by_difficulty.values)

    plt.xlabel("difficulty")
    plt.ylabel("avg word length")
    plt.title("avg word length by difficulty")
    plt.show()

    avg_length_by_difficulty
    return


if __name__ == "__main__":
    app.run()
