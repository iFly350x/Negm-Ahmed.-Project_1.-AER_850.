# Negm-Ahmed.-Project_1.-AER_850.
AER 850 Project_1. 

Course: AER 850 – Intro to Machine Learning

Instructor: Dr. Reza Faieghi

Student: Ahmed Negm (XXXXX1640)
Due: Oct 6, 2025, 11:59 PM

1) Project Overview

This project builds ML classifiers to predict the maintenance step (1–13) for the FlightMax Fill Motion Simulator inverter from its X, Y, Z coordinates.
It implements the full rubric:

Step 1: Data loading & checks (Pandas)

Step 2: Statistical summaries & class-wise visualizations (Matplotlib/Numpy/Pandas)

Step 3: Feature–target relevance (Pearson within features + ANOVA F / Mutual Info vs. categorical Step)

Step 4: Model dev (RF, SVC, KNN) with GridSearchCV + RandomizedSearchCV

Step 5: Metrics (Accuracy, Precision, Recall, F1 incl. macro), confusion matrices

Step 6: StackingClassifier (RF + SVC → Logistic Regression)

Step 7: Packaging final model to Joblib + prediction script

The selected deployment model is SVC, prioritizing precision for maintenance safety.

2) Repository Structure
.
├── data/
│   └── Project_Data.csv
├── main.py                # end-to-end train/visualize/evaluate/stack/save
├── predict_step.py        # loads final model and predicts 5 given coordinates
├── README.md              # this file
├── requirements.txt       # Python dependencies
└── figures/               # plots saved here (auto-created)


If you run from an IDE like Spyder, ensure your working directory is the project root (same level as data/).

3) Setup
Python

Create & activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

Install dependencies
pip install -r requirements.txt


Minimal requirements.txt:

numpy
pandas
matplotlib
seaborn
scikit-learn
joblib

4) Run the Project
A) Train, visualize, evaluate, stack, and save the model
python main.py


What happens:

Loads data/Project_Data.csv

Prints statistical summary

Shows &/or saves plots to figures/

Splits data with stratify=y

Tunes RF, SVC, KNN with GridSearchCV; RF again with RandomizedSearchCV

Reports metrics (Accuracy, Precision, Recall, F1 (weighted & macro))

Builds StackingClassifier (RF + SVC, passthrough=True)

Chooses final model (SVC or stacked) and saves as:

final_model.joblib (deployment)

Optionally also saves best_svc_model.joblib

If figures don’t appear (headless run), they are saved under figures/.

B) Predict maintenance steps for the 5 required coordinates
python predict_step.py


Expected printed output:

[9.375, 3.0625, 1.51] -> Step 5
[6.995, 5.125, 0.3875] -> Step 8
[0, 3.0625, 1.93] -> Step 13
[9.4, 3, 1.8] -> Step 6
[9.4, 3, 1.3] -> Step 4

5) Key Design Choices (brief)

Stratified split & CV: preserves class balance across 13 steps.

Pipelines: StandardScaler + SVC/KNN to ensure fair tuning and reproducible inference.

Metrics: report weighted and macro F1; prioritize precision (safety-critical AR guidance), while monitoring recall.

Stacking: RF captures axis-aligned relations; SVC captures smooth boundaries. In our data, stacking gave marginal gains; SVC remained preferred.

6) Outputs

Saved model: final_model.joblib

Figures (examples):

figures/summary_stats.txt (optional text dump)

figures/hist_X.png, hist_Y.png, hist_Z.png

figures/box_X_by_step.png, box_Y_by_step.png, box_Z_by_step.png

figures/corr_features.png (Pearson among X,Y,Z)

figures/relevance_anova_mutualinfo.png

figures/cm_*_normalized.png (model confusion matrices)

Filenames may differ slightly depending on your implementation; ensure you save or include screenshots in the report.

7) Reproducibility

Seeds: random_state=42 used for split and CV.

Environment: pin versions in requirements.txt if exact reproducibility is required.

Paths: data loaded via Path("data") / "Project_Data.csv".

8) Troubleshooting

FileNotFoundError: data/Project_Data.csv

Confirm you’re running from the repo root or use an absolute path.

In Spyder, set working directory to the project folder.

NameError: file_path is not defined

Call load_and_process_data() with no args (uses default) or define file_path = Path("data")/"Project_Data.csv" before passing it.

Plots not showing

They should display in interactive environments; otherwise, save to figures/ and embed in the report.

ConvergenceWarning for LogisticRegression (stacking)

Already mitigated using max_iter=1000. Increase if needed.

9) Grading Checklist (Rubric → Artifact)

Step 1: main.py → load_and_process_data() (column checks, prints)

Step 2: visualize_data(), visualize_by_class() + saved hist/box/3D plots

Step 3: Pearson among features + feature_target_relevance() (ANOVA F / MI) + bar plot

Step 4: train_classification_models() with GridSearchCV (RF/SVC/KNN) + RandomizedSearchCV (RF)

Step 5: evaluate_model_performance() → Accuracy, Precision, Recall, F1 (macro & weighted) + normalized CMs

Step 6: stacked_model_performance_analysis() (RF+SVC→LR, passthrough=True) + CM + discussion

Step 7: joblib.dump(final_model, 'final_model.joblib') + predict_step.py

10) License & Acknowledgements

For academic submission purposes only.

Thanks to scikit-learn, pandas, numpy, matplotlib, and seaborn.
