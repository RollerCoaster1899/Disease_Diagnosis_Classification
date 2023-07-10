# Disease_Diagnosis_Classification
1. Imports the necessary libraries:
   - `numpy` as `np`: A library for numerical computations.
   - `pandas` as `pd`: A library for data manipulation and analysis.
   - `matplotlib.pyplot` as `plt`: A library for creating visualizations.
   - Various modules from the `sklearn` (Scikit-learn) library, including `train_test_split` for splitting the data, `StandardScaler` for feature scaling, `LogisticRegression` for logistic regression modeling, `confusion_matrix` for evaluating model performance, and `classification_report` for generating a classification report.

2. Loads the dataset from the UCI Machine Learning Repository:
   - The data is retrieved from the URL specified in the `url` variable.
   - The column names for the dataset are provided in the `column_names` list.
   - The data is read using `pd.read_csv()` with the options `header=None` (since the dataset doesn't have a header row) and `names=column_names` (to assign the provided column names to the dataframe columns).
   - The resulting dataframe is stored in the `data` variable.

3. Performs data cleaning:
   - Drops the 'id' column from the dataframe using `data.drop()` with `axis=1` and `inplace=True`. This removes the column permanently from the dataframe.

4. Performs data engineering:
   - Maps the values of the 'diagnosis' column from 'M' and 'B' to 1 and 0, respectively. This is done using `data['diagnosis'].map()` with a dictionary mapping of `{'M': 1, 'B': 0}`. The updated values are stored back in the 'diagnosis' column.

5. Conducts exploratory data analysis (EDA):
   - Prints the first few records of the dataframe using `data.head()` to display a preview of the data.
   - Prints the summary statistics of the dataframe using `data.describe()` to provide statistical information about the data.
   - Prints the count of each class in the 'diagnosis' column using `data['diagnosis'].value_counts()` to show the distribution of classes.

6. Performs data preprocessing:
   - Assigns the features (all columns except 'diagnosis') to the variable `X` and the target variable ('diagnosis') to the variable `y`.
   - Splits the data into training and testing sets using `train_test_split()`. It assigns 80% of the data to the training sets (`X_train` and `y_train`) and 20% to the testing sets (`X_test` and `y_test`).
   - Performs feature scaling on the training and testing data using `StandardScaler()`. It transforms the data to have zero mean and unit variance. The scaling is applied separately to the training and testing data.

7. Trains a logistic regression model:
   - Creates a logistic regression classifier using `LogisticRegression()` and assigns it to the variable `classifier`.
   - Trains the classifier on the training data using `classifier.fit()` with the features (`X_train`) and the target variable (`y_train`).

8. Tests the logistic regression model:
   - Predicts the target variable for the testing data using `classifier.predict()` and assigns the predicted values to `y_pred`.

9. Evaluates the model:
   - Computes the confusion matrix using `confusion_matrix()` by comparing the predicted values (`y_pred`) with the actual values (`y_test`) from the testing data. The result is stored in the variable `confusion_mat`.
   - Generates a classification report using `classification_report()` to provide a detailed evaluation of the model's performance. The report includes metrics such as precision, recall, F1-score, and support for each class.
   - Prints the confusion matrix and classification report using `print()`.

10. Plots a visualization of the confusion matrix:
   - Creates a figure using `plt.figure()` with a specified size.
   - Sets the title of the plot to 'Confusion Matrix' using `plt.title()`.
   - Displays the confusion matrix as an image using `plt.imshow()` with the 'Blues' colormap and 'nearest' interpolation.
   - Sets the labels for the x-axis and y-axis using `plt.xticks()` and `plt.yticks()`, respectively.
   - Adds a colorbar to the plot using `plt.colorbar()`.
   - Adds text annotations to the plot using nested loops and `plt.text()` to display the values of the confusion matrix elements.
   - Sets the labels for the x-axis and y-axis using `plt.xlabel()` and `plt.ylabel()`, respectively.
   - Displays the plot using `plt.show()`.
