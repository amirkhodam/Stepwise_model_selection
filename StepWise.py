# Importing the statsmodels module
import statsmodels.api as sm
import warnings

warnings.filterwarnings(
    "ignore", message="The bic value is computed using the deviance formula.")

# Defining a class for performing forward stepwise selection of predictors in a GLM model


class StepwiseSelection:
    """
    Class for performing forward stepwise selection of predictors in a GLM model.
    """

    # Defining a constructor method that initializes the StepwiseSelection object with the response variable,
    # predictor variables, and the distribution family for the GLM model
    def __init__(self, response, comparator, predictors, family=sm.families.Gaussian(), trace=True):
        """
        Initialize the StepwiseSelection object with the response variable, predictor variables,
        and the distribution family for the GLM model.
        """
        # This code disables the warning message "The bic value is computed using the deviance formula." 
        # by using the filterwarnings() function from the warnings module.
        warnings.filterwarnings(
            "ignore", message="The bic value is computed using the deviance formula.")
        # This code disables the warning message "overflow encountered in exp"
        # by using the filterwarnings() function from the warnings module.
        warnings.filterwarnings(
            "ignore", message="overflow encountered in exp")
        
        # Initialize the property of class 
        self.response = response # Response column.
        self.predictors = predictors # Predictor columns.
        self.family = family # Link function of GLM model.
        self.comparator = comparator # Expected compartor parameter for selecting better model between two model.
        self.n = len(response) # Number of samples.
        self.selected_predictors = [] # Empty array that it will fill with selected predictors of best model.
        self.selected_model = None # Empty variable, It will fill with best model.
        self.trace = trace # Trace let print summary of model in each step of backward or forward method.

    # Defining a method for fitting the GLM model using forward stepwise selection of predictors
    def forward(self):
        """
        Fit a GLM model using forward stepwise selection of predictors.
        """
        predictors = self.predictors.copy()
        while len(predictors) > 0:
            best_predictor = None
            best_model = None
            best_score = None
            # Iterating through the predictors and selecting the best predictor for the model
            for predictor in predictors:
                candidate_predictors = self.selected_predictors + [predictor]
                X = sm.add_constant(self.predictors[candidate_predictors])
                model = sm.GLM(self.response, X, family=self.family).fit()
                # Determining the score based on the specified comparator ('aic' or 'bic')
                match self.comparator:
                    case 'bic':
                        score = model.bic
                    case _:
                        score = model.aic

                # Updating the best predictor, model, and score if a better candidate is found
                if best_score is None or score < best_score:
                    best_predictor = predictor
                    best_model = model
                    best_score = score
                if(self.trace):
                    print(best_model.summary())
            # If the score for the selected model is worse than the current best model, stop iterating
            check_best_score = False
            match self.comparator:
                case 'bic':
                    check_best_score = best_score >= best_model.bic
                case _:
                    check_best_score = best_score >= best_model.aic
            if len(self.selected_predictors) > 0 and check_best_score:
                break
            # Add the best predictor to the list of selected predictors
            self.selected_predictors.append(best_predictor)
            # Set the selected model to the best model
            self.selected_model = best_model

    # Defining a method for fitting the GLM model using backward stepwise selection of predictors
    def backward(self):
        """
        Fit a GLM model using backward stepwise selection of predictors.
        """
        predictors = self.selected_predictors.copy()
        while len(predictors) > 1:
            best_predictor = None
            best_model = None
            best_score = None
            # Iterating through the selected predictors and removing the least significant predictor for the model
            for predictor in predictors:
                candidate_predictors = predictors.copy()
                candidate_predictors.remove(predictor)
                X = sm.add_constant(self.predictors[candidate_predictors])
                model = sm.GLM(self.response, X, family=self.family).fit()
                # Determining the score based on the specified comparator ('aic' or 'bic')
                if self.comparator == 'bic':
                    score = model.bic
                else:
                    score = model.aic

                # Updating the worst predictor, model, and score if a worse candidate is found
                if best_score is None or score > best_score:
                    best_predictor = predictor
                    best_model = model
                    best_score = score
            # If the score for the selected model is worse than the current worst model, stop iterating
            match self.comparator:
                case 'bic':
                    check_best_score = best_score >= best_model.bic
                case _:
                    check_best_score = best_score >= best_model.aic
            if check_best_score:
                break
            # Remove the worst predictor from the list of selected predictors
            predictors.remove(best_predictor)
            # Set the selected model to the worst model
            self.selected_model = best_model
            # Set the selected predictors to the remaining predictors
            self.selected_predictors = predictors

    # Defining a method for printing a summary of the selected model
    def summary(self):
        """
        Print a summary of the selected model.
        """
        print(self.selected_model.summary())
