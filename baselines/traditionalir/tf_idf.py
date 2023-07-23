from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Read CSV file
df = pd.read_csv('zero_shot_train_cleaned.csv')
df_test = pd.read_csv('zero_shot_test_cleaned.csv')


# Get CVE texts and labels

# cve_texts = df["cleaned"].apply(lambda x: np.str_(x)).tolist()
cve_texts = df["cleaned"].apply(lambda x: np.str_(x)).tolist()
labels = df.columns[5:].tolist()
cve_test_texts = df_test["cleaned"].tolist()

ground_truth = df_test.iloc[:, 5:].apply(
    lambda row: [labels[idx] for idx, val in enumerate(row) if val == 1], axis=1).tolist()


# Initialize TfidfVectorizer and fit on all_texts
tfidf = TfidfVectorizer(ngram_range=(1,2)).fit(cve_texts)

# Calculate label vectors
label_vectors = tfidf.transform(labels)

# Initialize statistics for evaluation
num_test_data = 0
total_labels = len(labels)
prediction_not_seen = {}
prediction_not_seen_correct = {}
sum_precision_1, sum_recall_1, sum_precision_2, sum_recall_2, sum_precision_3, sum_recall_3 = 0, 0, 0, 0, 0, 0

# For each CVE text, calculate the top 3 most similar labels and update statistics
for idx, cve_text in enumerate(cve_test_texts):
    cve_vector = tfidf.transform([cve_text])
    similarities = cosine_similarity(cve_vector, label_vectors).flatten()
    predictions = similarities.argsort()[-3:][::-1]
    labels_actual = ground_truth[idx]
    num_test_data += 1
    correct_prediction = 0

    # Print predicted labels and actual labels
    # print(f"For CVE Text \"{cve_text}\":")
    # print(f"Predicted labels: {[labels[predictions[i]] for i in range(3)]}")
    # print(f"Actual labels: {labels_actual}")
    # print()

    # Update statistics for top 3 predictions
    for i in range(3):
        if labels[predictions[i]] in labels_actual:
            correct_prediction += 1
            if labels[predictions[i]] not in prediction_not_seen:
                prediction_not_seen_correct[labels[predictions[i]]] = prediction_not_seen_correct.get(
                    labels[predictions[i]], 0) + 1
        if labels[predictions[i]] not in prediction_not_seen:
            prediction_not_seen[labels[predictions[i]]] = prediction_not_seen.get(
                labels[predictions[i]], 0) + 1
        if i == 0:
            sum_precision_1 += correct_prediction / 1
            sum_recall_1 += correct_prediction / len(labels_actual)
        elif i == 1:
            # normalized P@k
            # sum_precision_2 += correct_prediction / min(2, len(labels_actual))
            # Standard P@k
            sum_precision_2 += correct_prediction / 2
            sum_recall_2 += correct_prediction / len(labels_actual)
        elif i == 2:
            # normalized P@k
            # sum_precision_3 += correct_prediction / min(3, len(labels_actual))
            # Standard P@k
            sum_precision_3 += correct_prediction / 3
            sum_recall_3 += correct_prediction / len(labels_actual)

# Calculate final evaluation metrics
precision_1 = sum_precision_1 / num_test_data
recall_1 = sum_recall_1 / num_test_data
f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)

precision_2 = sum_precision_2 / num_test_data
recall_2 = sum_recall_2 / num_test_data
f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2)

precision_3 = sum_precision_3 / num_test_data
recall_3 = sum_recall_3 / num_test_data
f1_3 = 2 * precision_3 * recall_3 / (precision_3 + recall_3)

# Print results
print(f"K = 1\nP@1 = {precision_1}\nR@1 = {recall_1}\nF@1 = {f1_1}\n")
print(f"K = 2\nP@2 = {precision_2}\nR@2 = {recall_2}\nF@2 = {f1_2}\n")
print(f"K = 3\nP@3 = {precision_3}\nR@3 = {recall_3}\nF@3 = {f1_3}\n")
print(f"TOTAL LABELS: {total_labels}")

# Get the feature names
# feature_names = tfidf.get_feature_names_out()

# Print the number of features
# print("Number of features: ", len(feature_names))