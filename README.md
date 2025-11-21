# The task is to train 3 models (text, speech, text & speech i.e multi) to classify the top 10 dialogue acts from the SwDA corpus.
## There are 2 main Python notebooks: features.ipynb and classification .ipynb and the respective accompanying 9 .csv data files.

1. features.ipynb extracts text features (3 csv files - text_features_{train, valid, test}.csv), and speech features (3 csv files - speech_features_{train, valid, test}.csv) based on the original {train, valid, test}.csv.
- Text features have 29 NLTK features which are added to the original LiWC features.

NLTK features extracted (29 total):
['word_count', 'sentence_count', 'avg_sentence_length', 'noun_count', 'proper_noun_count', 'verb_count', 'modal_verb_count', 'adj_count', 'adv_count', 'pronoun_count', 'determiner_count', 'preposition_count', 'conjunction_count', 'interjection_count', 'particle_count', 'pos_tag_diversity', 'named_entity_count', 'person_entity_count', 'organization_entity_count', 'location_entity_count', 'stopword_count', 'stopword_ratio', 'unique_bigrams', 'unique_trigrams', 'bigram_diversity', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 'sentiment_compound']

- Speech features have 13 Librosa features instead of the initial 45 which were too computationally time-consuming and expensive to run.

2. classification.ipynb trains 3 models for text, speech, and speech & text. Speech & text 
- train.csv files are used to train the models, which are also saved as .pkl extensions.
- valid.csv files are used for performance analysis (accuracy and F1, generating confusion matrices).
- test.csv is used to fill in the 'da_tag' column with predictions based on the best performing model. 

3. The classification notebook shows different cells training different models for text, speech, and text & speech. All models leveraged sklearn libraries. The following models yielded the best performance and were chosen for further analysis in dk3424_task2_responses.pdf:
- text model: random forest classifier 
reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html 
- speech model: MLP (multi-layer perceptron), feed forward neural classifier 
reference: https://scikit-learn.org/stable/modules/neural_networks_supervised.html 
- text and speech model: gradient boost classifier 
reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

4. The writeup for task 1 is in dk3424_task1_responses.pdf, while task 2 is in dk3424_task2_responses.pdf. In total, the 9 csv files are:
- text_features_train.csv
- text_features_valid.csv
- text_features_test.csv 
- speech_features_train.csv
- speech_features_valid.csv 
- speech_features_test.csv
- test_dk3424_text.csv (text model predictions) with Random Forest Classifier
- test_dk3424_speech.csv (speech model predictions) with MultiLayerPerceptron (FFNN i.e feed forward neural network)
- test_dk3424_multi.csv (text and speech model predictions) with Gradient Boost Classifier



