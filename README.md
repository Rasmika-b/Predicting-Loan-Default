# Naming Conventions of the Scripts 

`estimator.py`

Includes the code for loading the fitted model

`predictor.py`

Uses the trained model to make predictions and calibrates it.

`preprocess.py`

Includes the code for preprocessing and cleaning data.

`harness.py` 

Includes the code for reading input, using estimator and predictor to make predictions, and finally writing to output csv

`requirements.txt`

Includes dependencies to install using pip for executing code

`logit.joblib`

Trained Logit model to import in the estimator
