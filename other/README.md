# Other

### Training and Test Splits

We can use a model to accurately fit to a set of data, with a 100% accuracy.

The issue with this is that the model would likely do a horrible job at modelling unseen data. There'd be a lack of generalization. You've essentially overfit.

A means to mitigate this is to **test** a model on unseen data, prior to inference.

To do so, you'd split your dataset into training and test sets. 

The Training set would then be used to learn the optimal parameters, while the test set, being held as unseen data, would be used to evaluate the model using a variety of metrics (i.e., F1-Score, ROC-AUC, Accuracy, MSE,$R^2$, etc).

You want to make sure you avoid any type of data leakage, make sure your test data is always independent from the training data.

TLDR

- Training Data is used to fit the model
- Test data is used to measure performance, by predicting the label with a model, comparing the label with the real value, and then measuring the error (MSE, MAE, etc)

You can use `sklearn.model_selection.train_test_split` or `sklearn.model_selection.ShuffleSplit` to split a dataset or `sklearn.model_selection.StratifiedShuffleSplit`

### Cross Validation


#### K-Fold Cross Val

In K-fold cross validation, the dataset is split into different folds of training and testing sets, where each fold has a unique set of training and testing data, of same size across folds.

A unique model is trained across different folds and the loss and accuracy metrics are averaged to get a true error metric.

This can eliminate the risk of overfitting and gives a more broader view of what true model error is on a generalized dataset.

#### LOOCV

An extreme end to K-fold Cross Validation, where one model is created for the number of samples in a dataset.

Here, a model is trained on all samples with one sample left out to be used for testing.

This is done over all samples, with a then averaged loss to get a more representative error.