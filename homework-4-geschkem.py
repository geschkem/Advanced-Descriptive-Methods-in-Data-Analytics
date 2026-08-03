import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import nltk
    import pandas as pd
    import spacy
    import statsmodels.formula.api as smf
    from sklearn.feature_extraction import DictVectorizer
    from nltk.corpus import stopwords
    from collections import Counter
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    return Counter, mo, pd, spacy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    __Maddie Geschke__
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework 4 -- Ngrams, Named Entities, and Phrases
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## (Before Monday's Class) Assignment Setup ... mostly same as Homework 2

    This assignment assumes you have completed the following setup tasks before class:

    #### 1. Installed Python's Natural Language Toolkit (NLTK), pandas, and spacy
    #### 2. Downloaded the NLTK stopwords list and the punkt tokenizer
    #### 3. Imported the `pandas` and `nltk` libraries
    #### 4. Loaded the Glassdooor jobs data as a Pandas DataFrame, with the variable `glassdoor_df`
    #### 5. Created a new column called `label` with values as described in the setup slides for homework 2
    #### 6. Created a new Pandas DataFrame called `glassdoor_df_binary` that represents only rows labeled as 'data science' or 'data analytics'
    """)
    return


@app.cell
def _(pd):
    def label_jobs(desc):
            job = desc.lower()
            da = False
            ds = False
            if 'data science' in job:
                da = True
            if 'data analytics' in job:
                ds = True
            if da and ds:
                return 'both'
            if da:
                return 'da'
            if ds: 
                return 'ds'
            return 'neither'
    #loading in data set, making label column and binary column
    glassdoor_df = pd.read_csv('glassdoor_jobs.csv', index_col=0)       
    glassdoor_df['label'] = glassdoor_df['extracted_text'].apply(label_jobs)
    glassdoor_df_binary = glassdoor_df.loc[glassdoor_df['label'].isin(['da', 'ds'])]
    glassdoor_df_binary 
    return (glassdoor_df_binary,)


@app.cell
def _(mo):
    mo.md(r"""
    ## (Before Monday's Class) Assignment Setup ... new stuff

    #### 1. Install the Spacy library in your virtual environment (using the terminal or Powershell) with this command:

    ```uv pip install spacy```

    #### 2. Download Spacy's `en_core_web_sm` model using the terminal or Powershell with this following command:

    ```uv run -- spacy download en_core_web_sm ```

    #### 2. Import `spacy` in your code cell below

    #### 3. Call `spacy.load()` in your code cell below to load the "en_core_web_sm" model to a variable called `nlp`, like this:

    ```nlp = spacy.load("en_core_web_sm")```

    #### 4. Write python code in the cell below to opened the file called `wiki_quality_entities.txt` and read through the text of the file so that you end up with a Python list of phrases, one phrase per element in the list.

    #### 5. Name your list variable `wiki_entities`

    #### 6. Create a list of `spacy.doc` instances by looping through the Glassdoor job descriptions and converting them to `doc` objects like this:

    ```
    # after compelting all setup steps above, paste this code in the cell below and run it
    ```

    __Hint:__ the text file is formatted with one phrase per line.)
    """)
    return


@app.cell
def _(spacy):
    # i commented both of these out because i did all of the loading in the first code chunk
    #import pandas as pd
    #import spacy

    # python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

    #metadf = pd.read_csv('glassdoor_jobs.csv', index_col=0)
    #function for labeling the jobs
    return (nlp,)


@app.cell
def _(glassdoor_df_binary, nlp):
    sample_doc = nlp(glassdoor_df_binary['extracted_text'].iloc[0])
    return


@app.cell
def _(glassdoor_df_binary, nlp):
    # warning: running the code below might take a while depending on your system
    # for me, it took 3 mins, 22 seconds
    job_ads_binary = glassdoor_df_binary['extracted_text'].to_list()
    job_ads_spacy = [nlp(i) for i in job_ads_binary]
    return job_ads_binary, job_ads_spacy


@app.cell
def _(job_ads_binary, job_ads_spacy):
    len(job_ads_binary), len(job_ads_spacy)
    return


@app.cell
def _():
    # code for 4, 5, 6 of set up code 


    with open ('wiki_quality_entities.txt') as f:
        wiki_entities_raw = f.read()
    wiki_entities = wiki_entities_raw.split('\n')
    wiki_entities[0:10]
    return (wiki_entities,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Worksheet Questions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 1

    1. Iterate through all job advertisements (as Spacy `Docs`) and use Spacy's `sents` generator to break each document into sentences.
    2. Keep a running total of the number of sentences in all job advertisements and save that total as a variable called `sentence_count`.
    3. Print the `sentence_count` variable below.
    """)
    return


@app.cell
def _(job_ads_spacy):
    sentance_count = 0
    for doc in job_ads_spacy: 
        for sent in doc.sents:
            sentance_count +=1
    print(sentance_count)
    return (doc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 2

    1. Create a DataFrame called `df_pos_summary` containing columns for
    word and part-of-speech.
    2. Use a `groupby` and `agg` approach to show a count for every unique word/part-of-speech pair.
    3. Sort the DataFrame by `count` (descending), then `word`, then `part_of_speech`, so that it has the following structure:

    | index | word | part_of_speech | count |
    |---|---|---|---|
    | 0 | bear | noun | 512 |
    | 1 | lion | noun | 486 |
    | 2 | bear | verb | 387 |
    | 3 | apple | noun | 311 |

    __Note__: In this example DataFrame, the `count` for the first row would be a running total of how many times the word 'bear' was tagged as a noun in all the documents. Read more about part-of-speech tags in Spacy at https://spacy.io/usage/linguistic-features#pos-tagging

    4. Make sure your code cell displays your final `df_pos_summary` DataFrame
    """)
    return


@app.cell
def _(job_ads_spacy, pd):
    rows = []
    for i in job_ads_spacy: 
        for token in i: 
            if token. is_alpha:
                rows.append({
                    "word": token.text.lower(),
                    "part_of_speech": token.pos_
                })
    df_pos = pd.DataFrame(rows)
    df_pos_summary = (
        df_pos
        .groupby(["word", "part_of_speech"])
        .agg(count=("word", "size"))
        .reset_index()
        .sort_values(["count", "word", "part_of_speech"], ascending=[False, True, True])
    )
    df_pos_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 3

    1. Choose any one job advertisement that you think might be interesting to look at more closely.
    2. Loop through `Doc.ents` for the Spacy instance associated with this advertisement to access the Named Entities that were recognized by Spacy.
    3. Using whatever approach makes sense to you, write some Python code that results in a DataFrame called `df_ent_summary` with columns for the entity text, start, end, and label, like this:

    | index | text | start | end | label |
    |---|---|---|---|---|
    | 0 | this morning | 272 | 284 | TIME |
    | 1 | Steven Spielberg| 2089 | 2096 | PERSON |


    4. Make sure your code cell displays your final `df_ent_summary` DataFrame
    """)
    return


@app.cell
def _(job_ads_spacy, pd):
    document = job_ads_spacy[5]

    ents_data = []
    for ent in document.ents:
        ents_data.append({
            "text": ent.text, 
            "start": ent.start_char, 
            "end": ent.end_char, 
            "label": ent.label_
        })
    df_ent_summary = pd.DataFrame(ents_data)
    df_ent_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 4

    1. Explain which job ad you chose and why
    2. What kinds of entities did Spacy find in this ad?
    3. Do any of them seem like they might be clues about the job ad's category (DS vs. DA)?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. I chose this because it was the first one out of the first six job ads that looked interesting to me. I saw that it was about research in Cincinnati, which I thought would be interesting to look at because I am semi-familiar with the Cincinnati area, and figured that any findings that I found in Cincinnati would be somehwat applicable to the Columbus or Cleveland area.
    2. Spacy found ORG (an organization, in this case The Cincinnati Regional Chamber and the center for research and data),GPE (geopolitical entities, primarily Cincinnati, U.S.), and DATE (dates, annual and week).
    3. There were a couple things that Spacy recognized that may clue us as to whether this would be a DS or a DA ad. The one that stood out to me was SPSS, which Spacy classified as a person which I thought was interesting because to my knowledge it is not a person it is a platform. I figured that since this tool is used more analytically that it would indicate more for being a DA job rather than DS. I also noticed that Spacy detected 'preferred 5+ years as a date, which makes sense classification-wise but I thought it could be useful when classifying it as DA or DS, because I feel like having more experience may lead it to be more of a DS job rather than DA.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 5

    1. Choose a new job advertisement (not the same one that you used for questions 3 and 4)
    2. Use the `noun_chunks` generator method of the corresponding Spacy `Doc` to access the noun chunks that were recognized by Spacy.
    3. Create a DataFrame called `df_noun_chunk_summary` with the following structure:

    | index | chunk_text | root_text| root_dependency | root_token_head |
    |---|---|---|---|---|
    | 0 | any good results | results | dobj | get |
    | 1 | All the Americans | Americans | nsubj | going |

    That is, the DataFrame will have columns for the noun chunk text, the root text, the root dependency, and the root token's head.

    4. Make sure your code cell displays your final `df_noun_chunk_summary` DataFrame
    """)
    return


@app.cell
def _(doc, job_ads_spacy, pd):
    document2 = job_ads_spacy [6]
    chunks = []
    for chunk in doc.noun_chunks:
        chunks.append({
            "chunk_text": chunk.text,
            "root_text": chunk.root.text,
            "root_dependency": chunk.root.dep_,
            "root_token_head": chunk.root.head.text
        })
    df_noun_chunk_summary = pd.DataFrame(chunks)
    df_noun_chunk_summary

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 6

    1. Explain which job ad you chose for the noun chunk DataFrame and why
    2. What kinds of noun chunks did Spacy find in this ad? Do any of them seem like they might be clues about the job ad's category (DS vs. DA)?
    3. How do noun chunks seem to differ from Named Entities?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. I picked this job ad because it was the job ad that directly followed the one I selected in questions 3 and 4, and I was not feeling particularly creative with which number I should pick next. Also when I looked at the job ad the first qualification said a masters degree in statistics, which I interpretted as the job listing being very math-focused, so I was interested (as I am also very math-focused) to see what other qualities a job of this nature would look for.
    2. Spacy found a lot of noun chunks in this ad. The main ones were duties and responsibilities, which makes sense and I would assume that those two nouns would be the same across all job postings. Following that, they found that solutions, environments, effort, and exposure were also noun chunks that were found, so I am guessing that this job would have a lot of exposure to new scenarios and environments, and that adaptability will be important.
    3. From my observation, it seems that noun chunks have a lot more quantity than Named Entities, which makes sense since it is more common to talk about general nouns than specific names. It also seems that noun chunks will give more information about the context of a general situation (in this case, what the job entails) rather than the Named Entities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 7

    1. Write some Python code to iterate through the list of words and phrases from `wiki_quality_entities.txt` that you created during the "assignment setup" stage of assignment
    2. Reduce the term list to all word/phrase entries that are greater than one token in length.
    3. Instantiate a Spacy `PhraseMatcher` object and load it with this set of entities.
    4. Iterate through all job advertisements (as Spacy `Docs`) and use the `PhraseMatcher` object to match the phrases in each job advertisement.
    5. Use the phrases to construct a DataFrame called `df_wiki_phrases`, which each phrase and a count of how many times it occurred in the job advertisements.
    6. Sort the DataFrame by `count` (descending), and subsort by `phrase` so that the DataFrame has the following structure:

    | index | phrase | count |
    |---|---|---|
    | 0 | post office | 285|
    | 1 | income tax | 246 |
    | 2 | high school | 228 |

    7. Make sure your code cell displays your final `df_wiki_phrases` DataFrame
    """)
    return


@app.cell
def _(Counter, doc, job_ads_spacy, nlp, pd, wiki_entities):
    from spacy.matcher import PhraseMatcher

    multi_phrases = [p.strip().lower() for p in wiki_entities if len(p.split()) > 1 and p != ""]

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = list(nlp.pipe(multi_phrases))
    matcher.add("WIKI_PHRASES", patterns)
    phrase_counts = Counter()

    for doc2 in job_ads_spacy:
        matches = matcher(doc2)
        for match_id, start, end in matches:
            phrase = doc[start:end].text.lower()
            phrase_counts[phrase] += 1

    df_wiki_phrases = (
        pd.DataFrame(phrase_counts.items(), columns=["phrase", "count"])
        .sort_values(["count", "phrase"], ascending=[False, True])
        .reset_index(drop=True)
    )

    df_wiki_phrases
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 8

    1. How do the entities from `wiki_quality_entities.txt` compare with the Spacy named entities and noun chunks?
    2. What kinds of generalizations or interpretations can you make from the output of `df_wiki_phrases`?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. The entities from wiki_quality_entities.txt have very similar words high up with the noun chunks, but does not have as much overlap with the named entities. This makes sense, because throughout all of these job descriptions you would expect the general content to be the same, but less of the specific named entities would be consistent throughout all the job postings.
    2. From the output of this dataframe, we can see that some of the most important things throughout this job listings is risk management and risk exposure. Data solutions is another very frequent pair, but this should be expected since these are all data jobs. Another significant pair of works are key stakeholders, which helps with the overall conclusion that it is very important to keep in mind risk throughout the data process.
    """)
    return


if __name__ == "__main__":
    app.run()
