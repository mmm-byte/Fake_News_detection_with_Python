# Fake News Detection with Python

This repository contains code and resources for building a fake news detection system using Python and machine learning techniques. The aim of this project is to classify news articles as either real or fake based on their content and other relevant features.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Feature Extraction](#feature-extraction)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Fake news has become a prevalent issue in today's digital age. This project focuses on developing a fake news detection system that can help users distinguish between authentic and fabricated news articles. The system leverages machine learning algorithms to automatically analyze the content of news articles and predict their authenticity.

## Dataset
The dataset used for this project is the [Fake and real news dataset](https://github.com/mmm-byte/Fake_News_detection_with_Python/blob/main/Data/news.zip) from Kaggle. It consists of a collection of news articles labeled as either fake or real. The dataset is split into two CSV files: one for fake news and another for real news.

## Dependencies
The following dependencies are required to run the code:
- Python 3.x
- pandas
- scikit-learn
- numpy

Install the dependencies using pip:

pip install pandas scikit-learn numpy


## Usage
1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Place the downloaded dataset CSV files (`News.csv`) into the `data` directory.
4. Run the `Code.py` script:


## Feature Extraction
To classify news articles, we need to extract relevant features from the text data. This project uses the following features:
- Word frequency: Extracts the frequency of words in each article.
- TF-IDF: Computes the TF-IDF (Term Frequency-Inverse Document Frequency) values for words in each article.
- N-grams: Generates n-grams from the text data.

These features are then used as input to train the machine learning models.

## Machine Learning Models
This project utilizes the following machine learning algorithms for classification:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

The models are trained on the extracted features and used to predict the authenticity of new news articles.

## Evaluation
The performance of the models is evaluated using common evaluation metrics such as accuracy, precision, recall, and F1-score. The evaluation results are displayed after training and testing the models on the dataset.

## Conclusion
Fake news detection is a challenging task, and machine learning techniques can be useful in addressing this issue. This project demonstrates how to build a basic fake news detection system using Python and machine learning. However, it is important to note that this is a simplified implementation and may not capture all nuances of fake news detection.

## Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvement, please open an issue or submit a pull request.

## License
This project is licensed under the [Apache License 2.0](LICENSE). Feel free to use and modify the code as per the license terms.









