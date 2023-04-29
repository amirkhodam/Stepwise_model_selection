StepwiseSelection Class README
------------------------------

This repository contains a class named `StepwiseSelection` that is used for performing forward stepwise selection of predictors in a Generalized Linear Model (GLM).

### Requirements

This class requires the following dependencies to be installed:

*   [Python](https://www.python.org/) (version >= 3.6) (used==3.11.2)
*   [NumPy](https://numpy.org/)
*   [pandas](https://pandas.pydata.org/)
*   [statsmodels](https://www.statsmodels.org/)

### Usage

To use the `StepwiseSelection` class, you need to first import the `statsmodels` module:

pythonCopy code

`import statsmodels.api as sm`

Then you can import the `StepwiseSelection` class from the module where you saved it. You can then create an instance of the class and pass the response variable, predictor variables, and the distribution family for the GLM model. Additionally, you can specify the comparator parameter which determines the method used for selecting the best model. The default comparator is `aic`, which stands for Akaike Information Criterion. You can also set the `trace` parameter to `True` to print the summary of the model in each step of backward or forward method.

Example usage:

```pythonCopy code
from StepwiseSelection import StepwiseSelection
import pandas as pd

# Load data
data = pd.read_csv('my_data.csv')

# Define response variable
response_var = data['response']

# Define predictor variables
predictor_vars = data[['predictor_1', 'predictor_2','predictor_3']]

# Instantiate the StepwiseSelection object 
selection = StepwiseSelection(response_var, predictors=predictor_vars)  

# Perform forward stepwise selection of predictors
selection.forward()

# Get the best model
best_model = selection.selected_model 

# Print the summary of the best model 
print(best_model.summary())
```

### Methods

#### `__init__(self, response, comparator, predictors, family=sm.families.Gaussian(), trace=True)`

Initialize the `StepwiseSelection` object with the response variable, predictor variables, and the distribution family for the GLM model.

##### Parameters

*   `response`: pandas.Series
    *   The response variable.
*   `comparator`: str, {'aic', 'bic'}
    *   The method used for selecting the best model.
*   `predictors`: pandas.DataFrame
    *   The predictor variables.
*   `family`: sm.families.Family, optional (default=sm.families.Gaussian())
    *   The distribution family for the GLM model.
*   `trace`: bool, optional (default=True)
    *   If `True`, print the summary of the model in each step of backward or forward method.

#### `forward(self)`

Fit a GLM model using forward stepwise selection of predictors.

#### `backward(self)`

Fit a GLM model using backward stepwise selection of predictors.

### Contributors

*   [Amir Khodam](https://github.com/amirkhodam)

### License

This project is licensed under the MIT License - see the LICENSE file for details.