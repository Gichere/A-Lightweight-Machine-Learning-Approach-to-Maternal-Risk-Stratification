# A-Lightweight-Machine-Learning-Approach-to-Maternal-Risk-Stratification

## Overview
This repository contains the code and resources for a proof-of-concept (POC) that leverages lightweight machine learning to enhance maternal risk stratification in healthcare settings. The project, developed for Jacaranda Health, aims to prioritize high-risk mothers (e.g., those at risk of pre-eclampsia or gestational diabetes) during triage, using clinical data such as blood pressure and blood sugar.

By employing interpretable models—logistic regression and decision trees—this tool supports nurses in resource-constrained environments, like Kenyan hospitals, to deliver timely care and reduce maternal complications.

The POC demonstrates how data science can transform maternal healthcare by automating risk assessment and integrating with hospital ticketing systems. The code is implemented in R, using a public maternal health dataset from the UCI Machine Learning Repository.

## Dataset
The dataset used is the Maternal Health Risk Data Set from the UCI Machine Learning Repository. It contains 1,014 patient records with the following features:

- Age: Patient age in years
- SystolicBP / DiastolicBP: Blood pressure measurements (mmHg)
- BS: Blood sugar level (mmol/L)
- BodyTemp: Body temperature (°F)
- HeartRate: Heart rate (bpm)
- RiskLevel: Categorized as low, mid, or high risk

For modeling, RiskLevel was recoded into a binary variable (0 = low risk, 1 = medium/high risk). The dataset has no missing values, and continuous features were normalized for model stability.

## Repository Structure
The repository is organized as follows:

- data/: Contains the UCI dataset (Maternal Health Risk Data Set.csv)
- scripts/: Contains the main R script (maternal_risk_model.R) for data preprocessing, modeling, and evaluation
- outputs/: Contains visualizations (roc_curve.png, feature_importance.png, decision_tree_plot.png)
- README.md: This file
- LICENSE: MIT License


## Installation
To run the code, ensure you have R (version 4.0 or higher) installed. Follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/francisgichere/maternal-risk-stratification.git
   cd maternal-risk-stratification
   ```

2. Install required R packages: The script uses the pacman package to manage dependencies. Run the following in R:
   ```R
   install.packages("pacman")
   pacman::p_load(tidyverse, caret, pROC, rpart, rpart.plot, ggplot2, readr)
   ```

3. Download the dataset: [Maternal Health Risk Data Set](https://archive.ics.uci.edu/datasets?skip=0&take=10&sort=desc&orderBy=NumHits&search=Maternal+Health+Risk)

## Usage

1. Run the main script: Open Lightweight classifier for maternal health stratification.R in RStudio and execute it. The script:

   - Loads and preprocesses the dataset (normalisation, binary recoding)
   - Splits data into 80% training and 20% test sets
   - Trains logistic regression and decision tree models
   - Evaluates models using recall, precision, and AUC
   - Generates visualisations (ROC curves, feature importance, decision tree)

   To run:
   ```R
   source("Lightweight classifier for maternal health stratification.R")
   ```

2. View outputs: Visualisations are saved in your working folder:

   - roc_curve.png: ROC curves comparing model performance
   - feature_importance.png: Logistic regression coefficients
   - decision_tree_plot.png: Decision tree structure

3. Modify parameters:

   - Adjust the decision tree's maxdepth in for different tree complexities
   - Change the prediction threshold (default: 0.5) in log_pred for custom sensitivity/precision trade-offs


## Methodology
The project focuses on building a POC to demonstrate feasibility in resource-constrained settings. Key methodological choices include:

- Skipped EDA: Exploratory Data Analysis was omitted to prioritize rapid POC development, given the dataset's cleanliness (no missing values). Future work should include EDA to explore feature distributions and correlations.
- Logistic Regression Assumptions: Assumptions (e.g., linearity of log-odds, no multicollinearity) were not formally checked to maintain simplicity. Normalization and dataset integrity mitigate some risks, but future iterations should validate these assumptions.
- Models: Logistic regression and decision trees were chosen for interpretability, critical for clinical adoption. Logistic regression prioritizes recall (76%), while the decision tree offers high precision (92%).

For detailed results, see the accompanying article: [Saving Mothers with Data Science: A Lightweight Machine Learning Approach to Maternal Risk Stratification](https://francisgichere.medium.com/saving-mothers-with-data-science-a-lightweight-machine-learning-approach-to-maternal-risk-013bdf47e6c9)


## Results

Logistic Regression:
- Recall: 76% (catches 76% of high-risk mothers)
- Precision: 80%
- AUC: 0.83


Decision Tree:
- Recall: 70%
- Precision: 92%
- AUC: 0.797


Blood sugar and blood pressure were the strongest predictors, aligning with clinical knowledge of gestational diabetes and hypertension as key risk factors.

## Limitations and Future Work

- Data: Lacks features like medical history or symptoms, which could improve predictions
- EDA: Skipped to expedite POC development; future work should include correlation analysis and outlier detection
- Assumptions: Logistic regression assumptions need validation (e.g., VIF for multicollinearity)
- Deployment: Requires integration with hospital systems and user-friendly interfaces

Future enhancements:

- Incorporate additional features (e.g., proteinuria)
- Test ensemble methods like random forests
- Pilot in Kenyan hospitals to validate real-world impact


## License
This project is licensed under the MIT License. See the LICENSE file for details.

## References

- UCI Machine Learning Repository: Maternal Health Risk Data Set
- R packages: tidyverse, caret, pROC, rpart, rpart.plot, ggplot2, readr


## Contact
For questions or collaboration, contact me (Francis) at francisgichere@gmail.com.




