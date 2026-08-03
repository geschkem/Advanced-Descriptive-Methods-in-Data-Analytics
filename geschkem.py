import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Maddie Geschke
    # Coding Homework 5: Image Classification Using Traditional CV Methods

    ## Assignment Setup

    This assignment assumes you have installed the `opencv` and have access to `mpl_toolkits` from having installed a recent version of `matplotlib`. Run the code cells below to confirm. If you see images of Calvin and Hobbes (three sets, each with nine images), everything is working.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd 
    import numpy as np
    import seaborn as sns
    import cv2
    import glob
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import utils

    return ImageGrid, cv2, glob, mo, np, pd, plt, utils


@app.cell
def _(glob, mo):
    # load file path info for images

    calvin = glob.glob("labeled-segments/calvin/*.png")
    hobbes = glob.glob("labeled-segments/hobbes/*.png")
    both = glob.glob("labeled-segments/both/*.png")

    image_paths_all = calvin + hobbes + both 

    mo.vstack([len(calvin), len(hobbes), len(both), len(image_paths_all)])
    return both, calvin, hobbes, image_paths_all


@app.cell
def _(cv2, image_paths_all):
    # load images by file name and convert from BGR to RGB color values

    images_loaded = [cv2.imread(i) for i in image_paths_all]
    rgb_images = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in images_loaded ]
    return images_loaded, rgb_images


@app.cell
def _(ImageGrid, plt, rgb_images):
    # display some Calvin examples 

    _fig = plt.figure(figsize=(12.0, 12.0))
    _grid = ImageGrid(_fig, 111,  nrows_ncols=(3, 3), axes_pad=0.1)

    for _ax, _im in zip(_grid, rgb_images[3:12]):
        # Iterating over the grid returns the Axes.
        _ax.imshow(_im)

    _fig.suptitle(' Examples of Calvin without Hobbes', fontsize=14, y=0.95)

    plt.show()
    return


@app.cell
def _(ImageGrid, plt, rgb_images):
    # display some Hobbes examples 

    _fig = plt.figure(figsize=(12.0, 12.0))
    _grid = ImageGrid(_fig, 111,  nrows_ncols=(3, 3), axes_pad=0.1)

    for _ax, _im in zip(_grid, rgb_images[256:258] + rgb_images[264:265] + rgb_images[267:268] + rgb_images[270:273] + rgb_images[275:277]):
        _ax.imshow(_im)

    _fig.suptitle('Examples of Hobbes without Calvin', fontsize=14, y=0.95)

    plt.show()
    return


@app.cell
def _(ImageGrid, plt, rgb_images):
    # display some Calvin with Hobbes examples 

    _fig = plt.figure(figsize=(12.0, 12.0))
    _grid = ImageGrid(_fig, 111,  nrows_ncols=(3, 3), axes_pad=0.1)

    for _ax, _im in zip(_grid, rgb_images[324:333]):
        _ax.imshow(_im)

    _fig.suptitle('Examples of Hobbes with Calvin', fontsize=14, y=0.8)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calculating Summary Statistics

    Four baseline statistics will be used for this lab. The first three are associated with functions in the file `utils.py`. The fourth statistic can be calculated by performing Pandas operations on the data in `ocr.csv`.  In class, I will share code for importing and running the first three functions in a Marimo Notebook and generating the `word_count` data yourself. The four summary statistics are:

    __white_pixels_ratio__:	After thresholding the image at the grayscale value of 128, a the ratio of white pixels in the image to total pixels is calculated. In general, images with lighter shades or lots of white space should have higher scores for this measure.

    __box_area_to_total_area_ratio__: For this variable, convex hulls are derived and converted to rectangular bounding boxes. Any bounding boxes smaller than 4% of the area of the image are dropped from the data. The areas of the remaining bounding boxes are summed, and then divided by the total area of the image. Note that it is possible for this ratio to be greater than 1.0 if the bounding boxes have enough overlap.

    __colorfulness__: For each image, an index of colorfulness is derived based on the article "Measuring colorfulness in natural images" (Hasler and Süsstrunk, 2003). Their method, based on the idea of opponent colorspace representation, derives a numerical score that correlates with how study participants rated the colorfulness of a set of images. Their algorithm does not factor in the hue of the colors.

    __word_count__: Before the blur stage, each comic image was thresholded, and OCR was conducted using the pytesseract library (https://github.com/h/pytesseract). Recognized text was then tokenized and a recognized word count was derived for each image. Comics with fewer words could have less dialogue, or a preponderance of harder-to-recognize text, or a combination of the two.

    The idea of these functions is to run them on each comic segment, build up a Pandas DataFrame, and use that DataFrame to classify the images as either "Hobbes" or "Not Hobbes."
    """)
    return


@app.cell
def _(image_paths_all, images_loaded, mo, pd, threshed1, utils):
    # do the other extracting image features first so we can extracted the threshed_image that is used in the white black ratio function that is in the utils.py file 
    # theory: if mostly black a high number, if mostly white itll be a low number

    white_pixels_ratio = utils.white_black_ratio(threshed1)

    # box area to total theory: find an object, if a found object is big enough its added to the area, area of the boxes are as big as the total area of the image

    box_area_to_total_area_ratio = utils.box_area_to_total(threshed1)


    # colorfullness theory: look at utils.py function, this should be related to whether or not hobbs is there (because he is orange)
    # what does this number mean? 
    colorfulness = utils.image_colorfulness(images_loaded[324])


    # comment of what this is
    df_terms = pd.read_csv('ocr.csv') #this is loading the OCR data from the csv file

    df_terms_grouped = df_terms.groupby('file_root')[['text']].count().reset_index()
    df_terms_grouped

    word_count = df_terms_grouped.loc[df_terms_grouped['file_root'] == image_paths_all[324].split('/')[-1].replace('.png', '')].iloc[0]['text'] # this was to see one particular OCR image

    # this is printing out all of the outputs
    mo.vstack([white_pixels_ratio, box_area_to_total_area_ratio, colorfulness, word_count])
    return (df_terms_grouped,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extracting Image Features

    We'll do this section of the assignment in class together. I'll share code that covers the following methods:

    1. Grayscale / threshold images
    2. Blur / sharpen images
    3. Dilate / Erode images
    4. Finding edges, contours, and convex hulls
    5. Color masking
    6. Combining masks
    """)
    return


@app.cell
def _(ImageGrid, cv2, plt, rgb_images, utils):
    # creating image features block 
    # run this first, this uses the same image and makes the grayscale, blurred, eroded, dialated, the next five are thresholded at different levels 
    # note that all of these above variables are numpy arrays 
    # because they are black and white every number is a number between 0-255 and that indicates how much white there is in the vision 
    # in rgb color scheme they represent the amt of that individual color in each pixel 
    # pixel has either (number, number, number) (will say multipscale liek rgb) or just number (will say one scale, ex just blue or just black)

    gray = cv2.cvtColor(rgb_images[324], cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    eroded = utils.erode(gray,3,3,1)
    dilated = utils.dilate(gray,3,3,1)
    th1, threshed1 = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    th2, threshed2 = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
    th3, threshed3 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    th4, threshed4 = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    th5, threshed5 = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)

    _fig = plt.figure(figsize=(12.0, 12.0))
    _grid = ImageGrid(_fig, 111,  nrows_ncols=(3, 3), axes_pad=0.1)

    for _ax, _im in zip(_grid, [gray, blurred, eroded, dilated, threshed1, threshed2, threshed3, threshed4, threshed5]):
        _ax.imshow(_im, cmap='gray')

    _fig.suptitle('Examples of Transformations', fontsize=14, y=.85)

    plt.show()
    return (threshed1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train and Test a Classifier

    Using the labels in the metadata as your response variable and the scores about each image as predictors, train and test a KNN model that attempts to predict if an image contains Hobbes. You should be able to write the code for this part of the assignment yourself by adapting code from previous homeworks and following our standard machine learning workflow.

    1. Perform a train/test split
    2. Train a KNN model
    3. Generate predictions on the test data
    """)
    return


@app.cell
def _(
    both,
    calvin,
    cv2,
    df_terms_grouped,
    hobbes,
    image_paths_all,
    images_loaded,
    pd,
    utils,
):
    # we have done this in scikit learn with the data analytics/science code 
    # in order to do this we need to set up a dataframe that can do this 
    # columns of the dataframe: image_file, colorfulness, whiteblackratio, boxratio, wordcount, label (hobbes or no hobbes)

    # making this dataframe, starting with a list of png file names (list of strings)
    # first step: label by src (or grc) folder (this is label column)
    # loop
    # A. load image data from file name
    # B. BGR -> colorfulness funciton, run on every item in the list 
    # C. RGB -> greyscale -> BW -> black white ratio (have to force it down to one chanel and then decide whether it makes the cutoff)
    # D. Box ratio (has to do with BGR image, have to save)
    # output, save as a dictionary of lists, {"col_name": [], "col_name": [], etc}


    # code for above:
    # Convert to DataFrame of stats and labels
    summary_data = {'file_root': [],
                    'colorfulness': [], 
                    'black_pixels_ratio': [],
                    'box_area_to_total_area_ratio': []
                    }
    # loop BGR images
    for e, i in enumerate(images_loaded):
        # convert to rgb
        _rgb = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        # convert to grayscale
        _gray = cv2.cvtColor(_rgb, cv2.COLOR_BGR2GRAY)
    
        # at what level?
        _th, _threshed = cv2.threshold(_gray, 127, 255, cv2.THRESH_BINARY)
        _black_pixels_ratio = utils.white_black_ratio(_threshed)
        _box_area_to_total_area_ratio = utils.box_area_to_total(_threshed)
        _colorfulness = utils.image_colorfulness(i)

        _file_root = image_paths_all[e].split('/')[-1].replace('.png', '')
    
        # add to dictionary 
        summary_data['file_root'].append(_file_root)
        summary_data['colorfulness'].append(_colorfulness)
        summary_data['black_pixels_ratio'].append(_black_pixels_ratio)
        summary_data['box_area_to_total_area_ratio'].append(_box_area_to_total_area_ratio)
    

    df = pd.DataFrame(summary_data)
    df['label'] = ['no_hobbes' for i in range(len(calvin))] + ['hobbes' for i in range(len(hobbes))] + ['hobbes' for i in range(len(both))]

    df_final = df.set_index('file_root').join(df_terms_grouped.set_index('file_root'), how='outer')
    df_final = df_final.rename(mapper={'text':'word_count'}, axis=1)
    df_final
    return (df_final,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate Performance

    In the code chunk below, write code that displays the model's predictive accuracy on your test set, as well as the following indicators of performance.

    1. A confusion matrix (based on this documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)
    2. Per-class precision scores (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
    3. Per-class recall scores (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
    """)
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    return StandardScaler, train_test_split


@app.cell
def _(StandardScaler, df_final, train_test_split):
    # features
    X = df_final[['colorfulness', 'black_pixels_ratio', 'box_area_to_total_area_ratio', 'word_count']]
    y = df_final['label']

    # making my train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # standardising for knn
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, X_train_scaled, y_test, y_train


@app.cell
def _(X_test_scaled, X_train_scaled, y_train):
    # training knn 
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)
    return KNeighborsClassifier, knn, y_pred


@app.cell
def _(y_pred, y_test):
    # accuracy
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    print("Model A Accuracy:", accuracy)
    return (accuracy_score,)


@app.cell
def _(knn, plt, y_pred, y_test):
    # confusion matrix based off documentation 
    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=knn.classes_
    )

    plt.title("Confusion Matrix Hobbes Classification")
    plt.show()
    return (ConfusionMatrixDisplay,)


@app.cell
def _(knn, y_pred, y_test):
    # precision and recall scores based off documentation
    from sklearn.metrics import precision_score, recall_score

    precision = precision_score(y_test, y_pred, average=None, labels=knn.classes_)
    recall = recall_score(y_test, y_pred, average=None, labels=knn.classes_)

    # print results
    for a, label in enumerate(knn.classes_):
        print(f"\nClass: {label}")
        print(f"Precision: {precision[a]:.3f}")
        print(f"Recall: {recall[a]:.3f}")
    return label, precision_score, recall_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try to Improve the Model (Model B)

    1. Design an additional metric/statistic that you can generate or derive using `opencv` methods. Implement your idea by running a function on all images in the dataset and storing the result in your primary `DataFrame`
    2. Use your new metric--along with all previously generated/derived scores (including __white_pixels_ratio__, __box_area_to_total_area_ratio__, __colorfulness_score__, and __word_count__)--to train, test, and evaluate a new KNN model.
    3. As with Model A, write code to display the model's predictive accuracy on your test set; a confusion matrix, and per-class precision scores and recall scores
    """)
    return


@app.cell
def _(cv2, np):
    # my idea: if an image is brighter then less likely to be hobbes, use pixel intensity in the greyscale

    def brightness_score(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    return (brightness_score,)


@app.cell
def _():
    summary_data_new = {
        'file_root': [],
        'colorfulness': [],
        'white_pixels_ratio': [],
        'box_area_to_total_area_ratio': [],
        'brightness': [],
        'label': []
    }
    return (summary_data_new,)


@app.cell
def _(brightness_score, cv2, image_paths_all, summary_data_new, utils):
    for path in image_paths_all:

        img = cv2.imread(path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray1 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        _, threshed = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)

        white_ratio = utils.white_black_ratio(threshed)
        box_ratio = utils.box_area_to_total(threshed)
        color = utils.image_colorfulness(img)
        bright = brightness_score(img)

        file_root = path.split('/')[-1].replace('.png', '')

        summary_data_new['file_root'].append(file_root)
        summary_data_new['colorfulness'].append(color)
        summary_data_new['white_pixels_ratio'].append(white_ratio)
        summary_data_new['box_area_to_total_area_ratio'].append(box_ratio)
        summary_data_new['brightness'].append(bright)

        if "calvin" in path:
            summary_data_new['label'].append("no_hobbes")
        else:
            summary_data_new['label'].append("hobbes")
    return


@app.cell
def _(pd, summary_data_new):
    dfn = pd.DataFrame(summary_data_new)
    return (dfn,)


@app.cell
def _(StandardScaler, dfn, train_test_split):
    #  train test features
    Xn = dfn[
        [
            'colorfulness',
            'white_pixels_ratio',
            'box_area_to_total_area_ratio',
            'brightness'
        ]
    ]
    yn = dfn['label']

    #makign train test
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(
        Xn, yn,
        test_size=0.2,
        random_state=42,
        stratify=yn
    )
    # scaling data
    scalern = StandardScaler()
    Xn_train_scaled = scalern.fit_transform(Xn_train)
    Xn_test_scaled = scalern.transform(Xn_test)
    return Xn_test_scaled, Xn_train_scaled, yn_test, yn_train


@app.cell
def _(KNeighborsClassifier, Xn_test_scaled, Xn_train_scaled, yn_train):
    # train knn
    knn_b = KNeighborsClassifier(n_neighbors=5)
    knn_b.fit(Xn_train_scaled, yn_train)

    yn_pred_b = knn_b.predict(Xn_test_scaled)
    return knn_b, yn_pred_b


@app.cell
def _(accuracy_score, yn_pred_b, yn_test):
    # accuracy 
    accuracyn = accuracy_score(yn_test, yn_pred_b)
    print("Model B Accuracy:", accuracyn)
    return


@app.cell
def _(ConfusionMatrixDisplay, knn_b, plt, yn_pred_b, yn_test):
    # confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        yn_test,
        yn_pred_b,
        display_labels=knn_b.classes_
    )

    plt.title("Model B (brightness) Confusion Matrix ")
    plt.show()
    return


@app.cell
def _(knn_b, label, precision_score, recall_score, yn_pred_b, yn_test):
    # precision and recall stuff
    precision_b = precision_score(yn_test, yn_pred_b, average=None, labels=knn_b.classes_)
    recall_b = recall_score(yn_test, yn_pred_b, average=None, labels=knn_b.classes_)

    for u, label2 in enumerate(knn_b.classes_):
        print(f"\nClass: {label}")
        print(f"Precision: {precision_b[u]:.3f}")
        print(f"Recall: {recall_b[u]:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    In this section, you will write 3-5 sentences addressing each of the questions below. Use markdown as needed to format your responses.

    #### 1. Describe and interpret the performance of both models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first model we did (model a) had a 75% accuracy, which is not bad. This shows us that the colorfullness, black pixel ratio, box area, and word count give us a reasonable clue as to whether or not the image does have Hobbes in it. Along with accuracy we also looked at precision and recall values, and we can see from the values there that model a is better at predicting if an image has hobbes, and slightly worse at predicting if an image does not have hobbes.
    Switching gears, the second model (which looks at the brightness of an image) had an 86% accuracy, which is much much better than the first model. Also looking at the precision and recall values, we see much more similar values when comparing hobbes vs no hobbes, which makes us more confident in the model since it isn't more biased or more accurate towards a specific prediction value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 2. Explain your rationale for the metric you designed and implemented, including what aspect of the comics you were hoping to capture. (Note that the rationale here is more important than the metric itself... I will mostly be grading how you are thinking about this idea.) Did your chosen measure seem to be improve the model's performance?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    My rationale for my design and implementation of the brightness model was to see if brightness, alongside the other predictors that we previously identified, would have an affect on whether a certain comic contained hobbes or not. My idea was that, since our first model did have a pretty good accuracy, to not take away any of those factors, and just add to it a potential new variabel that may have an affect. Regarding specifically brightness, I thought that since Hobbes is orange and black (which i think are darker colors, in comparison to yellow or white or other colors), that having him be in an image would make the overall image darker. And even alongside the colors of hobbes alone, i thought that having his presence in an image would add more dark features, like having an additional figure would add more shadows to the picture, for example. And looking at my models performance, it did have a postiive impact on the accuracy of the models predictions (going from 75% accuracy to 86%).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3. Areas for Expansion and Further Inquiry

    This section can be notes/bullet points. In the markdown cell below, provide at least three areas for how this assignment could be deepened, expanded, or extended. Be prepared to share your ideas in class.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - I think the biggest thing that would be interesting to look closer into would be to specify a color code range, and then use that filter to see if an image had hobbes or not. I think this model would have the highest accuracy of any other, because if a certain comic is deemed highly orange (if you use that as your specified color) I think that by far it would have the highest accuracy out of any other model.
    - I think another area for further expansion would be to look vertically at the pixels and see if having a higher color concentration vertically would have a higher impact on whether hobbes is there or not, intuitively I would think that if there are more colorecd pixels vertically then it is more likely for hobbes to be there since hobbes is usually depicted as being much taller than calvin
    - another potential area could be to look closely and detect the consistency black lines in the image? this may be a stretch but hobbes is a tiger so maybe if there are more black lines in an image that are not words then the image is more likely to be hobbes
    """)
    return


if __name__ == "__main__":
    app.run()
