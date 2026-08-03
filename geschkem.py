import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Maddie Geschke
    # Coding Homework 6: Advanced Image Classification

    ## Assignment Setup

    This assignment assumes you have created a new virtual enviroment with Python 3.12 and installed `marimo`, `pandas`, `seaborn`, `opencv`, `sklearn`, and `tensorflow` in this new environment.

    Run the code cells below to confirm. If the final code cell before section II, "Coding Homework," prints an accuracy score, everything is working properly.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import tensorflow as tf
    import pandas as pd
    from collections import Counter
    import matplotlib.pyplot as plt
    import seaborn as sns

    return Counter, mo, pd, plt, sns, tf


@app.cell
def _(tf):
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 200x200 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 
        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.summary()
    return (model,)


@app.cell
def _(model, tf):
    RMSprop = tf.keras.optimizers.RMSprop
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    return


@app.cell
def _(tf):
    training_set = tf.keras.utils.image_dataset_from_directory(
      'sample_birds',
      seed=123,
      image_size=(200, 200),
      subset='training',
      validation_split=0.3,
      batch_size=5)

    validation_set = tf.keras.utils.image_dataset_from_directory(
      'sample_birds',
      shuffle=True,
      seed=17,
      image_size=(200, 200),
      validation_split=0.3,
      subset='validation',
      batch_size=5)

    holdout_set_all = tf.keras.utils.image_dataset_from_directory(
      'sample_birds',
      shuffle=False,
      seed=17,
      image_size=(200, 200),
      batch_size=1) # batch size has to be one for this set

    train_file_paths = training_set.file_paths
    validation_file_paths = validation_set.file_paths
    holdout_file_paths = holdout_set_all.file_paths
    return (
        holdout_file_paths,
        holdout_set_all,
        training_set,
        validation_file_paths,
        validation_set,
    )


@app.cell
def _(holdout_file_paths, holdout_set_all, validation_file_paths):
    images = []
    labels = []
    for e, image_label in enumerate(holdout_set_all):
        f = holdout_file_paths[e]
        if f in validation_file_paths:
            images.append(image_label[0].numpy())
            labels.append(image_label[1].numpy())
    len(validation_file_paths), len(images), len(labels)
    return images, labels


@app.cell
def _(model, training_set, validation_set):
    history = model.fit(training_set,
          epochs=11,
          verbose=1,
          validation_data = validation_set)

    model.evaluate(validation_set)
    return


@app.cell
def _(images, model, tf):
    img_arrays = []
    for i in images:
        img_arrays.append(i)

    test_dataset = tf.data.Dataset.from_tensor_slices(img_arrays)
    preds = model.predict(test_dataset)
    preds[0]
    return (preds,)


@app.cell
def _(Counter, labels):
    c = Counter([i[0] for i in labels])
    c
    return (c,)


@app.cell
def _(c, labels, pd, preds):
    df = pd.DataFrame()
    df['true_label'] = [i[0] for i in labels]
    df['predict_probability'] = [i[0] for i in preds]
    df = df.sort_values(by='predict_probability')
    inferred_labels = [0 for i in range(c[0])] + [1 for i in range(c[1])]
    df['predicted_label'] = inferred_labels 
    df['correct'] = df['true_label'] == df['predicted_label']
    df
    return (df,)


@app.cell
def _(df):
    len(df.loc[df['correct'] == True])/len(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## II. Coding Homework

    To complete this assignment, you will need to:

    1. Download the full dataset and unzip the compressed files
    2. Select bird species to classify (see below)
    3. Set up files and folders for classification
    4. Adapt the code above to train and evaluate various classifiers



    ### 1. Download the full dataset and unzip the compressed files

    Dataset link is located in `README.md`. The CUB 200 dataset has a unique id number and a species name in each folder name.

    ### 2. Select Bird Species to Classify

    Identify three sets of bird pairings and write a hypothesis about how difficult it will be for a model to differentiate the species from one another. You should select pairings so that you can reasonably predict different levels of success (e.g. an easy, medium, and hard task). For example, the sample pairing is the Black-footed Albatross vs. Artic Tern, and we might predict this classification task to be relatively easy.  (When making your selections, please do not use the sample pairing.)

    ### 3. Set up Files and Folders

    Tensorflow's `image_dataset_from_directory` method is _much_ faster than openCV, but it requires your files and folders to be set up in a specific way. You will need move files around as you go to create the following structure:

    ```
    > parent_folder
    	> class_a_folder
    		class_a_img_1
    		class_a_img_2
    		etc.
    	> class_b_folder
    		class_b_img_1
    		class_b_img_2
    		etc.
    ```

    Once you have this structure, you can use it to define training and validation sets. Labels can be supplied, but it's much easier to have Tensorflow infer them from the folder names.

    __Note:__ I have set up the sample_birds folder this way as a guide for you. You can easily create three new folders and call them `easy`, `medium`, and `hard` respectively.

    ### 4. Train and Evaluate Your Models

    Once your setup is complete, you will train and validate binary classification models for all of your pairs. Every model should be a CNN with the same architecture. (You can modify the architecture I've provided, but you shouldn't use different setups for different classifiers.)
    """)
    return


@app.cell
def _():
    # 1 download dataset and unzip files 
    # did it
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2 select birds to classify and write hypotheses
    - set 1: (easy) I chose the lazuili bunting because its colors are very bright orange and blue and the mockingbird is just grey, so it shoud be fairly easy in detecting a difference between them. They are also different species.
    - set 2: (medium) These are both flycatchers so they are the same species so I would expect the general body shape to be the same, but they are olive and vermillion so the colors would be different.
    - set 3: (hard) Both are the same species and very similar colors (brown), and the only main difference that I saw was a little bit of pattern difference.

      Note that I did use AI here, I prompted it to find pairs of birds from the list that would be easy to distingish, average to distinguish, and hard do distinguish between.
    """)
    return


@app.cell
def _():
    import shutil

    shutil.rmtree("easy", ignore_errors=True)
    shutil.rmtree("medium", ignore_errors=True)
    shutil.rmtree("hard", ignore_errors=True)
    return (shutil,)


@app.cell
def _(shutil):
    # 3 set up files and folders
    import os

    base = "CUB_200_2011/CUB_200_2011/images"

    def setup_pair_safe(folder_name, species1, species2, label1, label2, limit=200):
        shutil.rmtree(folder_name, ignore_errors=True)

        os.makedirs(f"{folder_name}/{label1}")
        os.makedirs(f"{folder_name}/{label2}")

        for f in os.listdir(f"{base}/{species1}")[:limit]:
            shutil.copy(f"{base}/{species1}/{f}", f"{folder_name}/{label1}")

        for f in os.listdir(f"{base}/{species2}")[:limit]:
            shutil.copy(f"{base}/{species2}/{f}", f"{folder_name}/{label2}")

    return (setup_pair_safe,)


@app.cell
def _(setup_pair_safe):
    # 2+3 setting up pairs

    setup_pair_safe("easy", "013.Bobolink", "091.Mockingbird", "bobolink", "mockingbird")

    setup_pair_safe("medium", "040.Olive_sided_Flycatcher", "042.Vermilion_Flycatcher", "olive", "vermilion")

    setup_pair_safe("hard", "119.Field_Sparrow", "120.Fox_Sparrow", "field", "fox")
    return


@app.cell
def _(tf):
    # 4 building a function to build the model
    def build_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200,200,3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=['accuracy']
        )

        return model

    return (build_model,)


@app.cell
def _(build_model, precision_score, recall_score, tf):
    # 4 doing the models 
    def run_model1(data_dir, epochs=10):

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.3,
            subset="training",
            seed=123,
            image_size=(200,200),
            batch_size=16
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.3,
            subset="validation",
            seed=123,
            image_size=(200,200),
            batch_size=16
        )

        assert len(train_ds.class_names) == 2, "Dataset is NOT binary!"
        print("Classes:", train_ds.class_names)

        model = build_model()

        model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)

        loss, accuracy = model.evaluate(val_ds)

        y_true = []
        y_pred = []

        for images, labels in val_ds:
            preds = model.predict(images, verbose=0)

            y_true.extend(labels.numpy())
            y_pred.extend((preds > 0.5).astype(int).flatten())

        return {
            "accuracy": accuracy,
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "y_true": y_true,
            "y_pred": y_pred
        }

    return (run_model1,)


@app.cell
def _(run_model1):
    easy = run_model1("easy")
    medium = run_model1("medium")
    hard = run_model1("hard")
    return easy, hard, medium


@app.cell
def _(plt, sns):
    # 4 continued - setting up function that can be used for all 3
    from sklearn.metrics import confusion_matrix, precision_score, recall_score

    # putting this in so it can print a heat map of it
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)
        plt.show()


    return confusion_matrix, precision_score, recall_score


@app.cell
def _():
    #4 now running all three using functions above
    #easy_results = run_model("easy")
    #medium_results = run_model("medium")
    #hard_results = run_model("hard")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Results Code

    For each model, report the following results:

    1. Overall accuracy of the model
    2. A confusion matrix of your validation set's True Positives, True Negatives, False Negatives, and False Positives
    3. Per class precision and recall (using `scikit-learn` functions)
    """)
    return


@app.cell
def _(confusion_matrix, easy, hard, medium, plt, sns):
    # making function to print results 
    
    def print_results(name, results):
        print(f"\n===== {name.upper()} =====")
        print("Accuracy:", results["accuracy"])
        print("Precision:", results["precision"])
        print("Recall:", results["recall"])

        cm = confusion_matrix(results["y_true"], results["y_pred"])

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.show()
    
    print_results("easy", easy)
    print_results("medium", medium)
    print_results("hard", hard)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interpretation

    1. Interpret your output, highlighting the key results and explaining the main takeaways. Revisit your hypotheses from the introduction. How did your models do compared to your hypotheses? What was surprising and why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looking at the results above, we can see that the medium difficulty model performed the best out of the three, it has a 97% accuracy with a very high precision and recall. The other two models performed very poorly, with the easy model performing with a 50% accuracy and the hard model performing with a 34% accuracy. Those moderately go along with the hypotheses I made earlier, with the two birds of the hard model being the same color and species it makes sense that it struggled to classify it. The results of the easy model does not align however, and suggests that the model failed in some way with classifying them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    2. Discuss the precision and recall scores of each class for your various models. What seems to be your strongest model and why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    My strongest model had a 1.0 precision score and a 0.956 recall score. This means that there are almost no false positives and identified almost all the actual positives, so it is a very strong model.
    My easy model had a 0.73 precision score and a 0.35 recall score. This means that it as a model is often correct but there are a lot of responses that it fails to classify as the correct one. This means that thsi model is missing a lot of things that it should be detecting and is too conservative of a model.
    My hard model had a 0.34 accuracy but a 0 precision and a 0 recall. This means that this model failed and did not detect anything of value, and that it cannot distinguish between similar species of birds,
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    3. As a whole, what does this analysis tell us? What are the strengths/limitations of  this data set? What are the strengths/limitations of this method? What is one future direction you could envision for future data analysts or data collectors?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This analysis tells us that this dataset is a good source of real life images, and that the amount of specificity that this dataset contains with its species makes it a good source of very specific data categories that we can train models from. However the amount of granular detail can make it difficult to differentiate (as we saw in my hard model) and in order to differentiate the species there needs to be a really high amount of specificity.

    Regarding this method, we can see how CNN can automatically learn visual features without us having to specify what to look at. We can see that it is also good at differentiating thing that are very clear, looking at our medium model we can see that it performed well there. However it does not do well with minute differences, like we saw in the hard model, and overall it is sensitive to various different factors.

    One future direction that I could envision for future data analysts or collectors would be to make it a bit more clear in the data itself about if there are certain pictures or species that contain a lot more variation in the picutures that are taken and used in the model. If one kind of bird has all kinds of different backgrounds and it is being trained against a bird that is consistently photographed in a similar environment then that could cause issues in the training of the model. My suggestion to this would perhaps be a more specific and in depth read me file that gave more context to the qualities of the pictures in the different folders.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    4. Take a step back and analyze your own use of code. Provide some rationale for choices you've made. How did you (or how might we) refactor the code to avoid repeating the same blocks three times?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I think separating the different parts of the code into separate functions worked really well especially when debugging, because I could look at what was going wrong in smaller chunks rather than one large function. I also thought that using the same CNN and overall architecture for all of the different models worked semi-well because it allowed us to fully compare between the three models. I will say that I did read this question and the homework before writing the code so this may have inspired me to do the functions rather than writing the same code chunk three times.
    """)
    return


if __name__ == "__main__":
    app.run()
