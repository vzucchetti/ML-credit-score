# Credit Scoring with Machine Learning in Python
### Building an Automated Credit Score Prediction Model

This repository contains a Python project focused on building a machine learning model to predict credit scores for bank clients. The project was undertaken as part of the "Python Journey" crash course offered by Hashtag and demonstrates a practical application of AI in finance.

## Project Overview

Manually assessing credit scores can be time-consuming and subjective. This project aims to automate the process by developing an AI model that can analyze client information and accurately predict their creditworthiness. The goal is to classify clients into credit score categories: POOR, STANDARD, and GOOD.

## Dataset

The project utilizes two datasets:

- ``clientes.csv``: Contains information about existing clients, including their credit score, which serves as the target variable for training the model. The features encompass various aspects of a client's financial profile:
    - ``id_cliente``: Unique identifier for each client
    - ``mes``: Month of the record
    - ``idade``: Age of the client
    - ``profissao``: Client's profession
    - ``salario_anual``: Annual salary
    - ``num_contas``: Number of bank accounts
    - ``num_cartoes``: Number of credit cards
    - ``juros_emprestimo``: Interest rate on loans
    - ``num_emprestimos``: Number of loans
    - ``dias_atraso``: Number of days overdue on payments
    - ``idade_historico_credito``: Age of credit history (in months)
    - ``investimento_mensal``: Monthly investment amount
    - ``comportamento_pagamento``: Payment behavior (e.g., high spending, low payment)
    - ``saldo_final_mes``: End-of-month balance
    - ``score_credito``: Credit score (Good, Standard, Poor)
    - ``emprestimo_carro``: Indicator for car loan (1 if has loan, 0 otherwise)
    - ``emprestimo_casa``: Indicator for home loan (1 if has loan, 0 otherwise)
    - ``emprestimo_pessoal``: Indicator for personal loan (1 if has loan, 0 otherwise)
    - ``emprestimo_credito``: Indicator for credit card loan (1 if has loan, 0 otherwise)
    - ``emprestimo_estudantil``: Indicator for student loan (1 if has loan, 0 otherwise)

- ``novos_clientes.csv``: Contains the same information as ``clientes.csv``, except for the ``score_credito`` column, which the model aims to predict for these new clients.

**Note:** The data in both CSV files is fictional and for illustrative purposes.

## Methodology

The project follows a structured approach:

1. **Data Loading and Exploration:** The datasets are loaded using Pandas, and their structure and contents are examined.
2. **Data Pre-processing:**
    - The ``LabelEncoder`` from ``scikit-learn`` is used to transform categorical features (e.g., profession, credit mix, payment behavior) into numerical representations, as machine learning models typically require numerical input.
    - The data is split into training and testing sets using ``train_test_split`` to evaluate the model's performance on unseen data.
3. **Model Selection and Training:**
    - Two machine learning models are trained:
        - *Random Forest*: A powerful ensemble method that combines multiple decision trees.
        - *K-Nearest Neighbors (KNN)*: A simpler algorithm that classifies a data point based on its similarity to neighboring data points.
    - Both models are trained using the training data.
4. **Model Evaluation:** The accuracy of each model is evaluated using the testing data and the ``accuracy_score`` metric from ``scikit-learn``. The model with the higher accuracy is chosen for making predictions.
5. **Prediction on New Clients:** The chosen model is used to predict the credit scores of new clients in the ``novos_clientes.csv`` dataset.

## Libraries
- ``pandas``: For data manipulation and analysis.
- ``scikit-learn``: For machine learning tasks including:
    - ``LabelEncoder``: To encode categorical variables.
    - ``train_test_split``: To split data into training and testing sets.
    - ``RandomForestClassifier``: To build the Random Forest model.
    - ``KNeighborsClassifier``: To build the KNN model.
    - ``accuracy_score``: To evaluate model performance.

## ow to Run the Project

1. Clone the repository:
```
git clone https://github.com/vzucchetti/ML-credit-score.git
```
2. Install the required libraries:
```
pip install pandas scikit-learn
```

3. Open and run the Jupyter Notebook (``main.ipynb``) to execute the code and explore the analysis.

## Conclusion

This project demonstrates how machine learning can be applied to automate and improve credit scoring processes. By leveraging the power of algorithms like Random Forest and KNN, financial institutions can enhance efficiency, reduce subjectivity in decision-making, and potentially improve the accuracy of credit risk assessments.
