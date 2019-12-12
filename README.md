# Overview

The project is the mini project of CS 6140. We propose a probable structure on
analyzing fine-grained sentiment for each text sequence.

# Environment

- Python v3.6.9
- PyTorch v1.2.0
- NLTK v3.4.5
- Pandas v0.25.3
- SciPy v1.3.1

# Usage

Run "summary.py" to predict results and get brief summary. If there is no "relations.txt"
file in specific output folder, it will call "relation_extractor.py" to predict results.

## Project Structure

- **src/**: all codes. Read README.md in the folder.
- **results/**: save all results in the folder, each run will built an independent
folder and all files related to current run will be saved in the folder. Due to the
size of the folder, it is not uploaded.
- **data/**:
	- **bert-embedding/**: put all BERT model file in the folder. It can be changed to
	any pre-trained model.
	- **black_decker/**: put all raw and processed data in the folder. Since the data
	can not be disclosed now, the folder is ignored in the project.

## File Description

**Relation Extraction**:
- **Combine_all_data**: aggregate data; combine data in different file into a single table.
- **UtilityFunction**: pre-process data; contain utility functions.
- **relation_extractor**: extract relationships among words.
- **summary**: call relation extractor, if need, and provide brief summary.
- **result_analytics**: draw word cloud for examples.
- **roc_curve**: draw ROC curve for results provided by "relation_extractor".
- **positiveandnegativewordsdictionary**: build dict for positive and negative words
correspondingly.
- **data_processor_minor**: prepare batch data for the model.

**Fine-tune pre-trained model**:
- **data_processor**: prepare batch data to train model.
- **main**: preprocess data, train model, test model, output results.
- **settings**: set (default) hyper parameters for models.
- **utils**: contains utility functions to help train models.
- **parser**: parse terminal parameters

