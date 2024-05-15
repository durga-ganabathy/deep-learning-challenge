# deep-learning-challenge
# DEEP LEARNING
NEURAL NETWORK MODEL 

OVERVIEW
The main purpose of this analysis is to create a tool for a nonprofit organization Alphabet Soup. The binary classifier aims to predict whether the applicant will be successful in their ventures if funded by Alphabet Soup. 
PROCESS
Alphabet Soup’s business team provided a CSV that contains 34,000 organizations that received funding from Alphabet Soup over the years. The columns in the CSV are:
EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding.
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested.
IS_SUCCESSFUL—Was the money used effectively.

Data Preprocessing
•	Used Pandas and scikit-learn’s “StandardScaler()” to preprocess the data.
•	Identified the target and feature variables. 
•	IS_SUCCESSFUL is the target variable y. The remaining columns are features variables X.
•	Dropped the columns EIN and NAME.
•	Determined unique values for each column. Assigned a cutoff point to combine "rare" categorical variables together in a new value, “Other”. 
•	Encoded the categorical variables. Split the preprocessed data into a features array, X, and a target array, y. Split the data into training and testing datasets.
•	Scaled the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

COMPILE, TRAIN, AND EVALUATION OF THE MODEL
The main goal of the model is to achieve 75% accuracy. 
Used 3 to 4 hidden layers in different attempts.
Attempted both ‘relu’ and ‘tanh’ activation function to check on the accuracy of the model. 
Tried different combinations of neuron units to achieve 75% accuracy of the model. 
Added/ reduced number of epochs.
The final analysis folder has Alphabetsoup starter file, AlphabetSoupCharity_Optimization1.ipynb and AlphabetSoupCharity_Optimization_2.ipynb. 
SUMMARY:
On many attempts of running the model with ‘relu’ and ‘tanh’ activation function with minimum of 3 hidden layers and trying different combinations of neurons the model did not achieve 75% of accuracy. 

In AlphabetSoupCharity_Optimization1.ipynb file the model achieved 72% of accuracy and 55% of loss. 
An accuracy of 72% suggests that the model correctly predicts the outcome (success or failure) for approximately 72% of the applicants. 
A loss of 55% is extraordinarily high and indicates significant errors in the model's predictions. One reason could be Overfitting. 

However, in AlphabetSoupCharity_Optimization_2.ipynb file the Name column is included back in the pandas dataframe and the model achieved 78% of accuracy and 45% of loss. 

Overall, this model did not meet the required accuracy. There may be opportunities for improvement in the model's optimization techniques. I would suggest trying a logistic regression, decision tree or random forest model to increase the accuracy of the model. Logistic regression is computationally efficient and can handle large datasets with many features. Here the goal is to predict the probability of a binary outcome (success or failure) based on one or more predictor variables and Logistic regression is a well-established method for binary classification tasks. Random forest is also a powerful method in predicting applicant success for Alphabet Soup and is less prone to overfitting. Its robust performance makes it a suitable alternative to neural networks.
