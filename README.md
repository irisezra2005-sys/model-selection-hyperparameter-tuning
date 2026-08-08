# model-selection-hyperparameter-tuning
## Project Overview

This project explores **model selection and hyperparameter optimization** for a binary classification problem. Three machine learning algorithms were evaluated:

- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

Both **GridSearchCV** and **RandomizedSearchCV** were used to systematically search for effective hyperparameter configurations. The tuned models were then evaluated on a held-out test set using accuracy, precision, recall, and F1 score.

The complete workflow is available in:

notebooks/hyperparameter_tuning.ipynb

## Project Objectives

The main objectives of this project are to:

1. Prepare a preprocessed dataset and divide it into training, validation, and test sets.
2. Train multiple machine learning algorithms suitable for the classification task.
3. Define meaningful hyperparameter search spaces.
4. Use GridSearchCV and RandomizedSearchCV to optimize model configurations.
5. Compare tuned models using consistent evaluation metrics.
6. Select the final model based on predictive performance, training time, and model complexity.
7. Document the findings and make the tuning workflow reproducible.

## Models Used

### 1. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve predictive performance and reduce overfitting.

### 2. Gradient Boosting

Gradient Boosting builds an ensemble of weak learners sequentially, where each new learner attempts to correct errors made by previous learners.

### 3. Support Vector Machine (SVM)

SVM attempts to find an effective decision boundary between classes. Different kernels and regularization parameters were explored during hyperparameter tuning.


## Hyperparameter Optimization

Two search strategies were implemented.

### GridSearchCV

GridSearchCV evaluates every combination of hyperparameters specified in the search grid.

The experiments used:

- 5-fold cross-validation
- F1 score as the primary optimization metric

Grid-search candidates:

| Model | Candidates | CV Folds | Total Fits |
|---|---:|---:|---:|
| Random Forest | 24 | 5 | 120 |
| Gradient Boosting | 24 | 5 | 120 |
| SVM | 12 | 5 | 60 |

### RandomizedSearchCV

RandomizedSearchCV samples a fixed number of hyperparameter combinations instead of evaluating every possible combination.

The experiments used:

- 5-fold cross-validation
- 20 sampled configurations per model
- F1 score as the primary optimization metric

Therefore, each randomized search performed:

20 candidates × 5 folds = 100 fits

## Final Test Set Results

The tuned models were evaluated on the held-out test set.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| **Random Forest** | **0.9292** | **0.8947** | 0.5312 | **0.6667** |
| Gradient Boosting | 0.9125 | 0.7037 | **0.5938** | 0.6441 |
| SVM | 0.8833 | 0.6000 | 0.3750 | 0.4615 |

### Best Model

**Random Forest** was selected as the final model because it achieved the highest test-set F1 score:

```text
F1 Score = 0.6667
```

It also achieved the highest test accuracy:

```text
Accuracy = 0.9292
```

Although Gradient Boosting achieved slightly higher recall (0.5938), Random Forest provided the best overall balance according to the primary F1 metric.

## Classification Performance

### Random Forest

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Not Good | 0.93 | 0.99 | 0.96 | 208 |
| Good | 0.89 | 0.53 | 0.67 | 32 |

Overall accuracy:

```text
0.93
```

### Gradient Boosting

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Not Good | 0.94 | 0.96 | 0.95 | 208 |
| Good | 0.70 | 0.59 | 0.64 | 32 |

Overall accuracy:

```text
0.91
```

### SVM

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Not Good | 0.91 | 0.96 | 0.93 | 208 |
| Good | 0.60 | 0.38 | 0.46 | 32 |

Overall accuracy:

```text
0.88
```

---

## Important Observation

The test set contains substantially more **Not Good** samples than **Good** samples (208 versus 32). Therefore, accuracy alone does not fully describe model performance.

For this reason, **F1 score, precision, and recall** were also considered when selecting the final model.

Random Forest produced the strongest F1 score for the minority **Good** class among the three models, while also achieving the highest overall accuracy.


## Observed Hyperparameter Search Results

### Grid Search

The best cross-validation F1 scores observed were approximately:

| Model | Best CV F1 | Validation F1 | Training Time |
|---|---:|---:|---:|
| Random Forest | 0.5001 | 0.6923 | 58.22 s |
| Gradient Boosting | 0.5288 | 0.7037 | 59.19 s |
| SVM | 0.4511 | 0.6538 | 2.78 s |

### Randomized Search

The best cross-validation F1 scores observed were approximately:

| Model | Best CV F1 | Validation F1 | Training Time |
|---|---:|---:|---:|
| Random Forest | 0.4593 | 0.7037 | 46.16 s |
| Gradient Boosting | 0.5282 | 0.6909 | 58.56 s |
| SVM | 0.4892 | 0.5818 | 71.85 s |

The complete best-parameter dictionaries are stored and displayed in the notebook output.

## Repository Structure

The repository follows the structure specified for the internship task:

```text
model-selection-hyperparameter-tuning/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── hyperparameter_tuning.ipynb
│
├── docs/
│   └── comparative_analysis.md
│
└── src/
    └── train_tuned_model.py
```

## Running the Hyperparameter Tuning Notebook

### Option 1: Google Colab

1. Open Google Colab.
2. Upload `notebooks/hyperparameter_tuning.ipynb`.
3. Upload the required dataset if it is not already accessible from the notebook.
4. Make sure the dataset path in the notebook is correct.
5. Run the notebook cells from top to bottom.
6. Allow the GridSearchCV and RandomizedSearchCV cells to finish.
7. Review:
   - Best hyperparameters
   - Cross-validation F1
   - Validation F1
   - Training time
   - Test accuracy
   - Test precision
   - Test recall
   - Test F1
   - Classification reports
   - Final model selection


## Running the Tuned Model Script

The reusable model-training script is located at:

```text
src/train_tuned_model.py
```

Run it from the project root with:

```bash
python src/train_tuned_model.py
```

Before running the script, verify that:

- The dataset path is correct.
- Required preprocessing steps are available.
- The required Python packages are installed.

---

## Evaluation Metrics

The following metrics are used:

### Accuracy

The proportion of all predictions that are correct.

### Precision

Measures how many samples predicted as a particular class are actually from that class.

### Recall

Measures how many actual samples of a class were successfully identified.

### F1 Score

The harmonic mean of precision and recall.

F1 score was used as the primary tuning metric because the dataset has an uneven class distribution and accuracy alone may be misleading.

## Reproducibility

To reproduce the experiments:

1. Use the Python/library versions specified in `requirements.txt`.
2. Use the same dataset and preprocessing procedure.
3. Run the notebook from the first cell to the last cell.
4. Keep the random states specified in the notebook unchanged.
5. Use the same cross-validation configuration and scoring metric.

## Final Conclusion

Three classification algorithms were compared after hyperparameter optimization.

Random Forest achieved the strongest final test performance with:

```text
Accuracy  : 0.9292
Precision : 0.8947
Recall    : 0.5312
F1 Score  : 0.6667
```

Gradient Boosting was the second-best model according to test F1 score, while SVM produced the lowest overall F1 score.

Based on the selected evaluation criteria, **Random Forest was chosen as the final model**.

The notebook provides the complete experimental workflow, while the comparative analysis document provides a detailed interpretation of the results and model-selection decision.
