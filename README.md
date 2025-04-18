# A-Lightweight-Machine-Learning-Approach-to-Maternal-Risk-Stratification

## Overview

This repository contains the code for a proof-of-concept (POC) that uses lightweight machine learning classifiers to enhance maternal risk stratification in healthcare settings. The project, presented to Jacaranda Health, aims to prioritise high-risk mothers (e.g., those at risk of pre-eclampsia or gestational diabetes) during triage, using clinical data such as blood pressure and blood sugar.

By employing interpretable models—logistic regression and decision trees—this tool supports nurses in resource-constrained environments, like Kenyan hospitals, to deliver timely care and reduce maternal complications.

The POC demonstrates how data science can transform maternal healthcare by automating risk assessment and integrating with hospital ticketing systems. The code is implemented in R, using a public maternal health dataset from the UCI Machine Learning Repository.

---

## Dataset

The dataset used is the Maternal Health Risk Data Set from the UCI Machine Learning Repository. It contains 1,014 patient records with the following features:
[Maternal Health Risk Data Set](https://archive.ics.uci.edu/datasets?skip=0&take=10&sort=desc&orderBy=NumHits&search=Maternal+Health+Risk)

- **Age**: Patient age in years  
- **SystolicBP / DiastolicBP**: Blood pressure measurements (mmHg)  
- **BS**: Blood sugar level (mmol/L)  
- **BodyTemp**: Body temperature (°F)  
- **HeartRate**: Heart rate (bpm)  
- **RiskLevel**: Categorized as low, mid, or high risk

For modeling, the `RiskLevel` was recoded into a binary variable (`0 = low risk`, `1 = medium/high risk`). The dataset has no missing values, and continuous features were normalized for model stability.

---




