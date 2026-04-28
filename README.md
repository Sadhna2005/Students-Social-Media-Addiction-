# Student Social Media Addiction Classification using Random Forest

A machine learning project that classifies students into **addiction levels** (Low, Moderate, High) based on their social media usage patterns and lifestyle indicators, using a **Random Forest Classifier**.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Addiction Level Definition](#addiction-level-definition)
- [Model Pipeline](#model-pipeline)
- [Features Used](#features-used)
- [Results](#results)
- [Project Structure](#project-structure)

---

## Project Overview

This project analyzes student behavior and lifestyle data to predict their level of social media addiction. Using a labeled addiction score from the dataset, students are grouped into three categories — **Low**, **Moderate**, and **High** — and a Random Forest Classifier is trained to predict these categories based on key personal and behavioral features.

The project includes exploratory data analysis (EDA), feature engineering, model training, and evaluation with standard classification metrics.

---

## Dataset

**File:** `Students Social Media Addiction.csv`

The dataset contains student-level data capturing social media behavior and associated lifestyle metrics.

| Column                        | Description                                      |
|-------------------------------|--------------------------------------------------|
| `Addicted_Score`              | Numerical score representing addiction intensity |
| `Age`                         | Age of the student                               |
| `Avg_Daily_Usage_Hours`       | Average hours spent on social media per day      |
| `Sleep_Hours_Per_Night`       | Average sleep hours per night                    |
| `Mental_Health_Score`         | Self-reported mental health score                |
| `Conflicts_Over_Social_Media` | Number of conflicts attributed to social media   |

> Place the CSV file in the same directory as the script before running.

---

## Tech Stack

| Library        | Purpose                                           |
|----------------|---------------------------------------------------|
| `pandas`       | Data loading, preprocessing, feature engineering  |
| `numpy`        | Numerical operations                              |
| `scikit-learn` | Random Forest model, train/test split, metrics    |
| `matplotlib`   | Data visualization                                |
| `seaborn`      | Count plot for addiction level distribution       |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/student-social-media-addiction.git
cd student-social-media-addiction
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Add the Dataset

Place `Students Social Media Addiction.csv` in the project root directory.

---

## Usage

Run the main script:

```bash
python addiction_classifier.py
```

The script will:
1. Load and preview the dataset
2. Engineer the `Addiction_Level` target column from `Addicted_Score`
3. Display the distribution of addiction levels as a bar chart
4. Train a Random Forest Classifier on selected features
5. Print the classification report, confusion matrix, and accuracy score

---

## Addiction Level Definition

Addiction levels are derived from the raw `Addicted_Score` column using the following thresholds:

| Score Range | Addiction Level |
|-------------|-----------------|
| 0 – 3       | Low             |
| 4 – 7       | Moderate        |
| 8 – 10      | High            |

---

## Model Pipeline

```
Raw CSV Data
      │
      ▼
Load & Preview Dataset
      │
      ▼
Engineer Target Column  (Addicted_Score → Addiction_Level: Low / Moderate / High)
      │
      ▼
Select Features  (Age, Usage Hours, Sleep, Mental Health, Conflicts)
      │
      ▼
Train/Test Split  (80% train, 20% test, random_state=42)
      │
      ▼
Random Forest Classifier  (random_state=42)
      │
      ▼
Predictions & Evaluation  (Accuracy, Classification Report, Confusion Matrix)
```

---

## Features Used

The following 5 features are used as predictors for the model:

| Feature                       | Type      | Role in Prediction                    |
|-------------------------------|-----------|---------------------------------------|
| `Age`                         | Numerical | Demographic context                   |
| `Avg_Daily_Usage_Hours`       | Numerical | Core behavioral indicator             |
| `Sleep_Hours_Per_Night`       | Numerical | Lifestyle impact indicator            |
| `Mental_Health_Score`         | Numerical | Psychological impact indicator        |
| `Conflicts_Over_Social_Media` | Numerical | Social consequence indicator          |

---

## Results

The model is evaluated using the following metrics:

| Metric            | Description                                                   |
|-------------------|---------------------------------------------------------------|
| Accuracy Score    | Overall percentage of correctly predicted addiction levels    |
| Precision         | How many predicted positives were actually correct per class  |
| Recall            | How many actual positives were correctly identified per class |
| F1 Score          | Harmonic mean of precision and recall per class              |
| Confusion Matrix  | Breakdown of predicted vs. actual labels across all classes  |

Results are printed to the console after each run.

---

## Project Structure

```
student-social-media-addiction/
│
├── addiction_classifier.py               # Main script
├── Students Social Media Addiction.csv   # Dataset file (add manually)
└── README.md                             # Project documentation
```

---

## License

This project is intended for academic and educational purposes.
