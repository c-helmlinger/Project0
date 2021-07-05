## Data

This project is trying to find the relationship between those attibutes and the target, which is an inference problem.
Due to the confidentality issues, the dataset uses "V" (i.e. V1, V2, V3, etc) to represent the features,and there 28 features.
In the dataset, the value of "class" is used to repsent fraud. If the class is one, then it’s a credit card fraud; if the class is zero, then it's not a fraud.
There are 284808 cases, among which there are 490 fraud cases.

## Brainstorm

We can divide the dataset into two groups: one is training set, and the other one is testing set. 
Then, we find  the characteristics of fraud through  machine learning from the training data set. 
Third, we test the accuracies of the model in the test set.
Also, we can repeat the steps by updating the training set to make the model more accurate.

## Intention

By building such a model, it can rapidly identify the fraud transaction when a credid card is making payments according to the characteristics.
Hence, the bank can cancel it to protect the clients' propoerties.
Specifically, the model can analyze massive amounts of transaction data 
and identify anomalies in the data, which is helpful for the financial industry and customers.
