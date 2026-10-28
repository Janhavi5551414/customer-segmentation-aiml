# Customer Segmentation Using Machine Learning

## Run the project (Windows)

1. Install Python 3.10+ if it is not already installed.
2. Extract this ZIP file.
3. Open the extracted folder in File Explorer.
4. Click the folder address bar, type `cmd`, and press Enter.
5. Run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

6. The dashboard should open in your browser.

## Dataset
The app tries to load the public Mall Customers CSV dataset automatically:
https://github.com/MahimaSriK/Mall-Customer-Segmentation-Dataset/blob/main/Mall_Customers.csv

If automatic loading fails, download the CSV from the link above (click **Raw** / download) and upload it in the app.

## What the project demonstrates
- Problem definition: group customers based on similar income and spending patterns.
- Dataset collection: Mall Customers sample dataset.
- Preprocessing: standardizes column names, converts required fields to numeric, removes missing required values.
- Model development: StandardScaler + K-Means clustering.
- Evaluation: silhouette score.
- Visualization: scatter plot and group summary.
- Model use: assign a new example customer to the nearest cluster.
- Result analysis: compare group sizes, average income, and average spending score.

Note: K-Means is unsupervised learning. Cluster numbers are labels, not ranked customer quality scores.
