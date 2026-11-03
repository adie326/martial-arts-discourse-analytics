# Sentiment and Discourse Analysis of Martial Arts Communities

A text-mining project that studies how people express opinions and
attitudes toward six martial arts through publicly available YouTube
comments. The project follows the **Knowledge Discovery in Databases
(KDD)** process and evaluates sentiment-analysis approaches designed to
identify attitudes toward a martial art---not merely the emotional tone
of the words used.

## Martial Arts Covered

-   Brazilian Jiu-Jitsu (BJJ)
-   Karate
-   Krav Maga
-   Taekwondo
-   Boxing
-   Mixed Martial Arts (MMA)

## Project Objectives

1.  Collect and preprocess a corpus of public YouTube comments about
    martial arts.
2.  Build a manually labeled sentiment dataset using documented
    annotation guidelines.
3.  Evaluate off-the-shelf sentiment tools and supervised classifiers
    using standard classification metrics.
4.  Apply association rule mining and clustering to discover patterns in
    discussions.
5.  Organize analytical results in a star-schema data warehouse and
    present them through an interactive dashboard.

## Dataset Overview

The current progress report records the following dataset sizes:

  Stage                                                           Comments
  ------------------------------------------------------------- ----------
  Raw comments collected through the YouTube Data API               28,121
  Unique comments after duplicate-ID removal                        27,189
  Comments remaining after preprocessing and length filtering       23,041
  Comments in the martial-arts focus set                             9,481
  Hand-labeled sentiment comments                                      476

The focus set contains comments that explicitly mention at least one of
the six martial arts. A comment may mention more than one style.

### Sentiment Labels

The manually labeled dataset contains three sentiment classes:

  Label        Count   Share
  ---------- ------- -------
  Positive       208   43.7%
  Neutral        205   43.1%
  Negative        63   13.2%

Labels represent the author's **attitude toward the martial art or
practicing it**, rather than the general tone of the comment. For
example, martial-arts slang that sounds negative in ordinary language
may express praise in this context.

## Methodology

The project follows these KDD stages:

1.  **Data selection and collection:** Public YouTube comments were
    collected using YouTube Data API v3.
2.  **Preprocessing:** Duplicate IDs were removed; URLs, timestamps, and
    excess whitespace were cleaned; comments outside the selected
    word-length range were filtered.
3.  **Transformation:** Comments were tagged by martial art using
    keyword patterns and represented with TF-IDF features for supervised
    models.
4.  **Data mining:** VADER, a pretrained RoBERTa sentiment model, Naive
    Bayes, Linear SVM, Decision Tree, and a combined logistic-regression
    model were evaluated.
5.  **Evaluation and interpretation:** Models were compared using
    accuracy, precision, recall, F1-score, macro F1, and confusion
    matrices.

## Models Evaluated

-   **VADER:** Lexicon-based sentiment analysis.
-   **RoBERTa:** `cardiffnlp/twitter-roberta-base-sentiment-latest`, a
    pretrained social-media sentiment model.
-   **Multinomial Naive Bayes:** Trained on TF-IDF features.
-   **Linear Support Vector Machine:** Trained on TF-IDF features.
-   **Decision Tree:** Trained on TF-IDF features.
-   **Combined model:** Logistic regression using TF-IDF features,
    RoBERTa class probabilities, and VADER's compound score.
-   **Majority-class baseline:** Always predicts the most frequent
    class.

Supervised models were evaluated using 5-fold stratified
cross-validation.

## Current Results

The best-performing model so far is the combined logistic-regression
model.

  -----------------------------------------------------------------------
  Model                                Accuracy                  Macro F1
  ------------------- ------------------------- -------------------------
  Majority-class                          43.7%                     0.203
  baseline                                      

  Decision Tree                           42.2%                     0.391
  (TF-IDF)                                      

  Naive Bayes                             54.4%                     0.387
  (TF-IDF)                                      

  Linear SVM (TF-IDF)                     54.0%                     0.416

  Logistic regression                     53.6%                     0.427
  (TF-IDF only)                                 

  RoBERTa, without                        50.2%                     0.464
  project-specific                              
  training                                      

  Logistic regression                     49.6%                     0.480
  (RoBERTa and VADER                            
  scores only)                                  

  **Combined logistic                 **56.3%**                 **0.513**
  regression (TF-IDF,                           
  RoBERTa and                                   
  VADER)**                                      
  -----------------------------------------------------------------------

The results show that general-purpose sentiment tools can mistake the
emotional tone of combat-sport vocabulary for the author's attitude
toward a martial art. The combined model performs best in the current
evaluation, but its predictions remain imperfect and should be
interpreted with care.

## Repository Scripts

The progress report describes the following scripts:

  -----------------------------------------------------------------------
  Script                              Purpose
  ----------------------------------- -----------------------------------
  `collect_youtube.py`                Collects comments through the
                                      YouTube Data API and saves raw JSON

  `build_dataset.py`                  Merges raw files and removes
                                      duplicate IDs

  `preprocess.py`                     Cleans text, filters by length, and
                                      tags martial arts mentioned

  `label_sentiment.py`                Applies VADER and draws the random
                                      sample for manual labeling

  `evaluate_vader.py`                 Compares VADER predictions with
                                      hand labels

  `evaluate_roberta.py`               Compares pretrained RoBERTa
                                      predictions with hand labels

  `make_label_batch2.py`              Draws the second English-only
                                      sample for manual labeling

  `train_classifiers.py`              Trains and cross-validates Naive
                                      Bayes, SVM, and Decision Tree

  `train_stacked.py`                  Evaluates the combined TF-IDF and
                                      pretrained-score model
  -----------------------------------------------------------------------

The exact repository folder layout and dependency manifest are not
specified in the progress report; consult the files included in the
repository for the current implementation details.

## Planned Work

The following components are planned and are not presented as completed
results:

-   Association rule mining using Apriori or FP-Growth on martial-art
    tags, sentiment, and attribute keywords such as self-defense, cost,
    and injury.
-   Clustering comments by motivation using K-means, hierarchical, and
    density-based methods, with validation such as silhouette scores.
-   A SQLite star-schema data warehouse with a comment fact table and
    dimensions for style, time, source, and cluster.
-   Further model improvement, including applying the best current model
    to the remaining comments and exploring RoBERTa fine-tuning.
-   Adding a second data source (Reddit, once access is approved, or
    Stack Exchange) and applying language filtering to the full dataset.
-   An interactive dashboard showing sentiment trends, association
    rules, and clusters by martial art.

## Getting Started

### Prerequisites

-   Python
-   Access to the project source code
-   A YouTube Data API v3 key for running the collection step

The report identifies the YouTube Data API v3, pandas, scikit-learn, and
Hugging Face Transformers among the tools used. Install the dependencies
required by the scripts in your local environment. A complete dependency
list and exact environment setup are not specified in the progress
report.

### API Key and Data Handling

Keep your YouTube API key in a local `.env` file and **never commit it
to GitHub**. Ensure `.env` is listed in `.gitignore`. The raw data
directory should also remain excluded from the public repository, as
planned in the project report.

No usernames were stored in the collected comment data.

### Suggested Script Order

For the current pipeline, the scripts described in the report can be run
in this general order:

1.  `collect_youtube.py`
2.  `build_dataset.py`
3.  `preprocess.py`
4.  `label_sentiment.py` and `make_label_batch2.py` for annotation
    sampling
5.  `evaluate_vader.py` and `evaluate_roberta.py`
6.  `train_classifiers.py`
7.  `train_stacked.py`

Check each script's arguments and configuration before running it; this
README does not assume command-line options that are not documented in
the report.

## Limitations

-   The hand-labeled dataset contains 476 comments and only 63 negative
    examples.
-   Labels were created by one annotator; inter-annotator agreement has
    not been computed.
-   YouTube search results and selected videos may introduce sampling
    bias.
-   Keyword-based tagging can miss comments that discuss a martial-arts
    category without naming one of the tracked styles.
-   The current work is based on YouTube; a second source is still
    planned.
-   At 56.3% accuracy, individual model predictions are not highly
    reliable. The report proposes using aggregate patterns cautiously
    and presenting model performance alongside results.

## Project Status

**Progress report:** Data collection, preprocessing, sentiment
annotation, and initial model evaluation have been completed.
Association rule mining, clustering, the star-schema warehouse, and the
dashboard remain planned work.

## References and Tools

The report references:

-   Han, J., Kamber, M., and Pei, J. *Data Mining: Concepts and
    Techniques*, 3rd edition.
-   Hutto, C. and Gilbert, E. (2014). "VADER: A Parsimonious Rule-Based
    Model for Sentiment Analysis of Social Media Text." Proceedings of
    ICWSM.
-   Loureiro, D. et al. (2022). "TimeLMs: Diachronic Language Models
    from Twitter." Proceedings of ACL (System Demonstrations).
-   Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in
    Python." *Journal of Machine Learning Research*, 12.

**Tools:** YouTube Data API v3, pandas, scikit-learn, Hugging Face
Transformers.
