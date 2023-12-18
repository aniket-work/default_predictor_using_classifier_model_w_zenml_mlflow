from zenml.steps.base_parameters import BaseParameters

class ModelConfig(BaseParameters):

    model_name: str = "model"

    model_params = {
        'n_estimators': 100,        # Number of trees in the forest
        'criterion': 'gini',        # Criteria for information gain ('gini' or 'entropy')
        'min_samples_split': 2,     # Minimum samples required to split a node
        'min_samples_leaf': 1,      # Minimum samples required at each leaf node
        'max_features': 'auto',     # Number of features to consider for the best split ('auto', 'sqrt', 'log2', int, float)
        'bootstrap': True,          # Whether bootstrap samples are used when building trees
        'class_weight': 'balanced'  # Handling imbalanced classes ('balanced', 'balanced_subsample', dict)        
    }
