# ReneWind – Wind Turbine Generator Failure Prediction (Neural Networks)

A neural-network-based predictive maintenance project that flags wind turbine generator failures ahead of time, so components can be repaired before they break instead of being replaced after a costly failure.

## Business Context

Wind energy is one of the most mature renewable energy technologies, and the U.S. Department of Energy has highlighted predictive maintenance as a key lever for operational efficiency. Predictive maintenance uses sensor data to spot degradation patterns before failure occurs — if a failing component can be identified early and replaced proactively, maintenance costs drop significantly.

Turbines are fitted with sensors capturing environmental conditions (temperature, humidity, wind speed, etc.) and the condition of key parts (gearbox, tower, blades, brake, etc.).

## Objective

**ReneWind** collects generator failure data from turbine sensors (shared here in a ciphered/anonymized form to protect confidentiality) and wants a classification model that can predict failures early enough to repair generators before they fail or break — reducing overall maintenance cost.

The cost structure behind the predictions:

| Outcome | Meaning | Cost Implication |
|---|---|---|
| **True Positive (TP)** | Failure correctly predicted | Repair cost |
| **False Negative (FN)** | Real failure missed by the model | Replacement cost (highest) |
| **False Positive (FP)** | False alarm — no actual failure | Inspection cost (lowest) |

Since repair cost < replacement cost, and inspection cost < repair cost, **minimizing false negatives (maximizing recall)** is the priority — missing a real failure is the most expensive mistake.

Target variable: `1` = Failure, `0` = No failure.

## Dataset

- `Train.csv` — 20,000 observations, used for training/validation and model tuning
- `Test.csv` — 5,000 observations, held out for final evaluation only
- 40 anonymized predictor columns (`V1`–`V40`) plus the `Target` column
- The dataset is **highly imbalanced**: ~94.4% "No Failure" vs. ~5.6% "Failure"
- `V1` and `V2` had a small number of missing values (~9%, in alternating rows), handled during preprocessing; no duplicate records were found

## Approach

1. **Exploratory Data Analysis** – distribution checks, class imbalance assessment, and statistical summary of all 40 (anonymized, scaled) predictors.
2. **Data Preprocessing** – missing-value handling, train/validation/test split (≈60/20/20), and class-weight computation to account for the imbalance.
3. **Baseline Model** – a Random Forest classifier trained as a benchmark before moving to neural networks.
4. **Neural Network Model Building** – a series of feed-forward (Dense) neural networks built and compared in Keras/TensorFlow, varying:
   - Number of hidden layers (1–3) and neurons per layer (64–128)
   - Activation functions (ReLU, Tanh, Sigmoid, combinations)
   - Optimizers (SGD, SGD with momentum, Adam)
   - Regularization techniques: Dropout, class weighting for the imbalanced target
5. **Model Performance Improvement** – systematic comparison of all architectures' train/validation loss and accuracy to select the most generalizable configuration.
6. **Final Model Evaluation** – the best-performing network evaluated on the held-out test set using a confusion matrix and accuracy/precision/recall.

## Results

The baseline Random Forest model served as a reference point; the final selected model was a 3-hidden-layer neural network (128 neurons per layer, ReLU → Tanh → Sigmoid activations) trained with the Adam optimizer.

**Final model performance on unseen test data:**

| Metric | Score |
|---|---|
| Accuracy | ~98.6% |
| Precision | ~88.9% |
| Recall | ~85.7% |

Multiple architecture and optimizer combinations were evaluated (varying depth, activation functions, SGD vs. Adam, with and without Dropout/class weighting); the chosen architecture gave the best balance of generalization and recall on validation data without overfitting.

## Business Recommendations

- Deploy the selected model to flag high-risk generators for proactive inspection/repair, reducing costly full replacements.
- Given the cost asymmetry (replacement > repair > inspection), continue to prioritize **recall** over raw accuracy when evaluating future model iterations — missing a real failure is far more expensive than a false alarm.
- Periodically retrain the model as sensor data distributions and turbine fleet characteristics evolve.
- Consider ensembling the neural network with the Random Forest baseline or exploring additional regularization/class-imbalance techniques (e.g., SMOTE) to further improve recall without sacrificing precision.

## Tech Stack

- **Python**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Baseline Modeling**: scikit-learn (Random Forest)
- **Deep Learning**: TensorFlow / Keras (Sequential API, Dense layers, Dropout, BatchNormalization)
- **Preprocessing**: scikit-learn (StandardScaler, LabelEncoder, OneHotEncoder, SimpleImputer, class-weight balancing)

## Repository Structure

```
.
├── Project_Introduction_to_Neural_Networks_ReneWind.ipynb   # Full EDA, modeling, and evaluation notebook
├── Train.csv                                                  # Training data (not included — see Data Access below)
├── Test.csv                                                    # Held-out test data (not included — see Data Access below)
└── README.md
```

## Data Access

The datasets used in this project are not included in this repository. Place `Train.csv` and `Test.csv` in the project root (or update the file paths in the notebook) before running.

## How to Run

1. Clone the repository and install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```
2. Place `Train.csv` and `Test.csv` in the project directory.
3. Open and run `Project_Introduction_to_Neural_Networks_ReneWind.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.

## Author

Mohamed Rifakath Ali A
