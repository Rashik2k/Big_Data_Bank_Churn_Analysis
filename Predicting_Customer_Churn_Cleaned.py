
# ------------------------
# Import necessary packages
# ------------------------

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
# Set PySpark to use your current Python environment
os.environ["PYSPARK_PYTHON"] = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
import subprocess
from sklearn.metrics import roc_curve, auc

# ------------------------
# Initialize Spark session
# ------------------------

spark = SparkSession.builder \
    .appName("CustomerChurnPrediction") \
    .master("local[*]") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
    .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse") \
    .config("spark.master", "local[1]") \
    .config("spark.local.dir", "C:/temp_spark") \
    .config("spark.driver.memory", "4g") \
    .config("spark.shuffle.spill", "false") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")\
    .getOrCreate()

# ------------------------
# 1. Load and preview dataset
# ------------------------

data_path = 'Churn_Modelling.csv'
df = spark.read.csv(data_path, header=True, inferSchema=True)
df.printSchema()
df.show(5)

# ------------------------
# 2. Exploratory Data Analysis (EDA)
# ------------------------

# Distribution of the target variable 'Exited'
df.groupBy('Exited').count().show()

# Correlation matrix for numeric features (we'll use Pandas for easier manipulation)
numeric_cols = [col for col, dtype in df.dtypes if dtype in ['int', 'double'] and col != 'Exited']
numeric_df = df.select(numeric_cols)

try:
    numeric_pandas_df = numeric_df.limit(10000).toPandas()
except Exception as e:
    print("⚠️ Error converting to Pandas:", e)
    numeric_pandas_df = pd.DataFrame()

# Compute correlation matrix
if not numeric_pandas_df.empty:
    correlation_matrix = numeric_pandas_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.show()

# Distribution of numeric features (histograms)
    numeric_pandas_df.hist(figsize=(12, 10), bins=30)
    plt.suptitle('Distribution of Numeric Features')
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Skipping correlation and histogram plots due to empty DataFrame.")

# ------------------------
# 3. Data Preprocessing
# ------------------------
#check if winutils is working properly
try:
    output = subprocess.check_output(["winutils.exe", "ls", "C:\\"])
    print("winutils is working.")
except Exception as e:
    print("Error running winutils.exe:", e)

#drop unnecessary data
# Handle missing values by removing rows with missing target variable
df = df.dropna()
df = df.drop('RowNumber', 'CustomerId', 'Surname')

#Writing the dataframe to csv file to perform EDA with tableau
df.coalesce(1).write.mode("overwrite").option("header", "true").csv("output_csv")

# Convert categorical variables into numeric using StringIndexer
categorical_cols = ['Geography', 'Gender']
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
indexers = [StringIndexer(inputCol=col_name, outputCol=col_name + '_index') for col_name in categorical_cols]

# Convert target column "Exited" into "label"
label_indexer = StringIndexer(inputCol="Exited", outputCol="label")
indexers.append(label_indexer)

 # Create a feature vector
assembler = VectorAssembler(inputCols=numeric_cols + [col + '_index' for col in categorical_cols],
                            outputCol='features')

# Create and apply the pipeline
pipeline = Pipeline(stages=indexers + [assembler])
df_transformed = pipeline.fit(df).transform(df)

# Split the data into training and testing sets (70-30 split)
train_df, test_df = df_transformed.randomSplit([0.7, 0.3], seed=42)

# ------------------------
# 4. Define Classification Models
# ------------------------

rf = RandomForestClassifier(featuresCol='features', labelCol='label')
gbt = GBTClassifier(featuresCol='features', labelCol='label')
dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')
svm = LinearSVC(featuresCol='features', labelCol='label')

models = [rf, gbt, dt, svm]
model_names = ['Random Forest', 'GBT', 'Decision Tree', 'SVM']

# ------------------------
# 5. Model Evaluation
# ------------------------

#Evaluate the model using various metrics
def evaluate_model(predictions):
    binary_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedRecall")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

 # Calculate metrics
    accuracy = accuracy_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    roc_auc = binary_evaluator.evaluate(predictions)

    return accuracy, precision, recall, f1, roc_auc

 # Store results
results = {}

for model, name in zip(models, model_names):
    print(f"🚀 Training {name}...")
    model_fit = model.fit(train_df)
    predictions = model_fit.transform(test_df)

    accuracy, precision, recall, f1, roc_auc = evaluate_model(predictions)
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }
    print(f"✅ Finished evaluating {name}")

# ------------------------
# 6. Print Evaluation Metrics
# ------------------------

for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-" * 40)

# ------------------------
# 7. Confusion Matrix
# ------------------------

def compute_confusion_matrix(predictions):
    tp = predictions.filter((col("prediction") == 1) & (col("label") == 1)).count()
    tn = predictions.filter((col("prediction") == 0) & (col("label") == 0)).count()
    fp = predictions.filter((col("prediction") == 1) & (col("label") == 0)).count()
    fn = predictions.filter((col("prediction") == 0) & (col("label") == 1)).count()
    return tp, tn, fp, fn

def plot_confusion_matrix(tp, tn, fp, fn, model_name):
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = ["Not Exited", "Exited"]
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

for model, name in zip(models, model_names):
    model_fit = model.fit(train_df)
    predictions = model_fit.transform(test_df)
    tp, tn, fp, fn = compute_confusion_matrix(predictions)
    plot_confusion_matrix(tp, tn, fp, fn, name)

# ------------------------
# 8. Combined ROC Curve
# ------------------------

plt.figure(figsize=(10, 8))
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.zeros_like(mean_fpr)
all_roc_auc = []

def plot_roc_curve(model_fit, test_df, model_name):
    predictions = model_fit.transform(test_df)

    if 'probability' in predictions.columns:
        probs = predictions.select("probability").rdd.map(lambda row: row[0][1]).collect()
    else:
        raw_preds = predictions.select("rawPrediction").rdd.map(lambda row: row[0]).collect()
        probs = [1 / (1 + np.exp(-x[1])) for x in raw_preds]

    labels = predictions.select("label").rdd.map(lambda row: row[0]).collect()
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    return tpr, roc_auc

for model, name in zip(models, model_names):
    model_fit = model.fit(train_df)
    tpr, roc_auc = plot_roc_curve(model_fit, test_df, name)
    mean_tpr += np.interp(mean_fpr, np.linspace(0, 1, len(tpr)), tpr)
    all_roc_auc.append(roc_auc)

mean_tpr /= len(models)
mean_auc = np.mean(all_roc_auc)

plt.plot(mean_fpr, mean_tpr, color='brown', linestyle='--', label=f'Mean ROC (AUC = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for All Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
