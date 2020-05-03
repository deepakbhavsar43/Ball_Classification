# Ball_Classification
##### A beginner ML project 

This is one of the simplest machine learning project to get started.

#### What is the project about?
This project predicts the lable of given input data Whether it is a **Tennis Ball** or **Cricket Ball**.

#### Understanding the project:
- First of all we need to read the dataset from the given **Ball_Dataset.csv** file
	- To read csv file use the **read_csv()** function from pandas.


- To train a ML model we need numerical data, but here we have string data.
	- So, convert the string data into numerical data using the function **factorize()**.
	- For more details refer the links [**stackoverflow-factorize()**](https://stackoverflow.com/questions/51311831/how-to-convert-categorical-data-to-numerical-data) or [**pydata-factorize()**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.factorize.html).

- After converting string data into numeric form, you need to seperate it for **Training** and **Testing**
	- For this use the function **train_test_split()** from sklearn.model_selection. This function splits the data into two parts as per the given ratio.
	- For more understanding of this function [**click here**](https://medium.com/contactsunny/how-to-split-your-dataset-to-train-and-test-datasets-using-scikit-learn-e7cf6eb5e0d).


- Now to train the data using the training dataset:
	- Create an object of class **DecisionTreeClassifier()** and load the training data to train the model using the **fit()** function.
	- To learn more about DecisionTreeClassifier() and fit() [**click here**](https://www.datacamp.com/community/tutorials/decision-tree-classification-python).


- After training its time to test the model:
	- use the **predict()** function of the DecisionTreeClassifier() class.
	- This will give the lable for given testing data.
	- To know more about **predict()** function [**click here**](https://www.datacamp.com/community/tutorials/decision-tree-classification-python).
