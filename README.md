# Ball_Classification
##### A beginner ML project 

This is one of the simplest machine learning project to get started.

#### What is the project about?
This project predicts the lable of given input data Whether it is a **Tennis Ball** or **Cricket Ball**. This project also demonstrate the use of pickle file. To know more about pickle file in detail [**click here**](https://www.datacamp.com/community/tutorials/pickle-python-tutorial)

#### How the dataset looks like?
| Weight | Surface | Label |
|---------|----------|-------|
| 35 | Rough | Tennis |
| 47 | Rough | Tennis |
| 90 | Smooth | Cricket |
| 48 | Rough | Tennis |

#### Understanding the project:
- First of all we need to read the dataset from the given **Ball_Dataset.csv** file
	- To read csv file use the **read_csv()** function from pandas.

- To train a ML model we need numerical data, but here we have string data.
	- So, convert the string data into numerical data using the function **factorize()**.
	- For more details refer the links [**stackoverflow-factorize()**](https://stackoverflow.com/questions/51311831/how-to-convert-categorical-data-to-numerical-data) or [**pydata-factorize()**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.factorize.html).
	- After conversion the dataset looks like as shown below:
	
| Weight | Surface | Label |
|---------|----------|--------|
| 35 | 0 | 0 |
| 47 | 0 | 0 |
| 90 | 1 | 1 |
| 48 | 0 | 1 |

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

#### How to execute program?
- Here the program is divided into 2 modules train ans test
- To execute Inbuilt_argparse.py to train model, use the below command:

```
> python InBuilt_argparse.pr -tr
# or youcan use
> python InBuilt_argparse.py --train
```
- To execute Inbuilt_argparse.py to test model, use the below command:

```
> python InBuilt_argparse.pr -te
# or you can use
> python InBuilt_argparse.py --test
```

- To execute Inbuilt_streamlit.py, use the below command in terminal:

```
> streamlit run InBuilt_streamlit.py
```

##### If you have any doub or problem then feel free to raise an issue 
If you find this repository helpful :star: this repository and share with your friends. 
