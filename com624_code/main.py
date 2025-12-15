import os
import warnings

import pandas as pd  # For data manipulation
# Import machine learning libraries
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import RobustScaler  # For feature scaling
from transformers import logging

from HDBscan_model import HDBscan_after_eda, HDBscan_before_eda
from after_cleaning import after_cleaning_empty_rows
from agglomerative_model import agglomerative_b4_eda, agglomerative_after_eda
from before_cleaning import before_cleaning_plot_missing_values, before_cleaning_empty_rows, \
    before_cleaning_duplicated_columns, before_phishing_label_bias
from cleaning import split_date, clean_text, mask_email_domain, normalise_time
from eda import run_feature_eda
from feature_engineering import count_spelling_errors, subject_length, has_subject_keywords, has_body_keywords, \
    extract_sender_domain_features, is_weekend, is_working_hours, extract_urls_from_dataframe, \
    extract_url_features_from_df
from kmeans_model import kmeans_clustering_after_eda, kmeans_clustering_before_eda
from supervised_learning import run_supervised_models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info/warning logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for consistent results
warnings.filterwarnings("ignore")  # Suppress all Python warnings

pd.set_option('display.max_columns', 10, )

file_path = 'data/email_phishing.csv'

#load file
if os.path.exists(file_path):
    df_raw = pd.read_csv(file_path,
                         encoding='ISO-8859-1',
                         dtype={"label": "int8", "urls": "int8"},
                         na_values=['na', 'NA', 'Unknown', ''])
else:
    raise FileNotFoundError(f"File not found: {file_path}")

expected = {"label", "urls", "subject", "body", "sender", "receiver", "date"}
present = expected & set(df_raw.columns)
missing = expected - present
if len(missing) > 0:
    raise ValueError(f"Missing columns: {missing}")

df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]',
                                                                                          '', regex=True)
# plot columns with no values
fig = before_cleaning_plot_missing_values(df_raw, 'missing values')
# start cleaning
missing_before = before_cleaning_empty_rows(df_raw)
duplicates = before_cleaning_duplicated_columns(df_raw)
before_phishing_label_bias(df_raw)

print(f"\nnumber of rows in DF")
print(len(df_raw))

df = df_raw.copy()

"""
clean data:
1) drop empty rows in receiver and subject
2) split date into multiple. add time column for UTC time for future use. drop rows with malformed timezone



"""
print("\n\n\n\n\n-------\nCleaning Started\n----------")
# drop rows where receiver and subject have missing values
df.dropna(subset=["receiver", "subject"], inplace=True)
after_cleaning_empty_rows(df)

# Regex pattern for valid timezone: ±HHMM (-0700) at end of string
valid_tz_pattern = r"[+-]\d{4}$"

# Identify rows with invalid timezone
invalid_rows = ~df["date"].str.contains(valid_tz_pattern, na=False)
# Drop them in place
print("Dropping these rows:\n", df.loc[invalid_rows, "date"].head(15))
df.drop(df[invalid_rows].index, inplace=True)
# split date into date, time, day and zone columns and normalise time
split_date(df)
normalise_time(df, clean_date_col='clean_date', time_col='time', zone_col='zone', verbose=True)
print(f"-------\n dataframe info\n-----")
print(f"{df.info()}\n\n-------")
rows_with_none = df[df["clean_date"].isnull()]
print(f"-------\n rows with Nan \n{rows_with_none}\n-------")

df['clean_body'] = df['body'].apply(clean_text)
#  drop about 4000 cnn emails and clean subject
df = df[~df["subject"].str.contains("cnn", case=False, na=False)]
df['clean_subject'] = df['subject'].apply(clean_text)
count = df['subject'].str.contains("cnn", case=False, na=False).sum()
print("CNN counts", count)
df = df[~df["clean_subject"].str.contains("cnn", case=False, na=False)]

# drop clean_subject rows with NaN
df.dropna(subset=["clean_subject"], inplace=True)
# remove names from  email addresses for receiver and sender
df['receiver'] = df['receiver'].apply(mask_email_domain)
df['sender'] = df['sender'].apply(mask_email_domain)
# drop columns for subject, date and zone
df.drop("subject", axis=1, inplace=True)
df.drop("date", axis=1, inplace=True)
df.drop("zone", axis=1, inplace=True)
df.to_csv('data/Cleaned_PhishingEmailData.csv', index=False, encoding='ISO-8859-1')
print(f"-------\nCleaned Phishing Email Data columns\n")
print(df.columns.tolist())
print("\n\nCleaning Completed - Clean csv file created")

print(f"\n\n---------\n{df.head(25)}\n\n\n---------")

"""
feature engineering:
2)  spelling error count
1)  domain name features
3)  subject length
4)  subject phishing keywords detected
5)  body phishing keywords detected
6)  detect if email sent on a weekend
7)  detect if email sent during working hours
"""
print("\n\n\n\n\n Feature Engineering Started")
extract_sender_domain_features(df)
df['has_spelling_errors'] = df['clean_body'].apply(count_spelling_errors)
subject_length(df)
has_subject_keywords(df)
has_body_keywords(df)
is_weekend(df)
is_working_hours(df)
extract_urls_from_dataframe(df)
extract_url_features_from_df(df)

"""
EDA 
create plots for features to gain insights
"""
print("\n\n\n\n\n EDA Started")
run_feature_eda(df)
print("\nSuspicious but marked not phishing:\n",
      df[(df['suspicious_tld'] == 1) & (df['label'] == 0)])

print("\nLegit domains but phishing:\n",
      len(df[(df['sender_is_free_email'] == 0) & (df['label'] == 1)]))

"""
Kmeans clustering 

"""
print("\n\n\n\n\n Kmeans Started")
# Select features for clustering
kmeans_clustering_before_eda(df)
kmeans_clustering_after_eda(df)

"""
DBSCan
"""
#print("\n\n\n\n\n DBSCAN Started")
HDBscan_before_eda(df)
HDBscan_after_eda(df)
#
"""
Agg clustering
"""
#print("\n\n\n\n\n Agg Started")
agglomerative_b4_eda(df)
agglomerative_after_eda(df)

df.to_csv('data/feature_added_clean_data.csv', index=False, encoding='ISO-8859-1')
print(df.describe())

"""
supervised learning
"""
logging.set_verbosity_error()  # Suppress transformer warnings

#  Feature engineering from email content
features_super_learning = [
    'has_spelling_errors',
    'keyword_count_body',
    'keyword_count_subject',
    'subject_length',
    'sender_length',
    'sender_domain_freq',
    'is_weekend',
    'is_working_hour',
    'hour',
    'sender_is_free_email',
    "url_length",
    "has_at_symbol",
    "subdomain_count",
    "path_length",
    "digit_count_domain",
    "uses_https",
    "suspicious_tld",
    "subdomain_density",
    "path_to_domain_ratio",
    "is_long_url",
    "https_with_long_url",
    "rare_domain_with_subdomains"

]
X = df[features_super_learning]  # Feature matrix for supervised learning
y = df['label']  # Target labels
# Step 8: Split data into training and testing sets and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
run_supervised_models(X_train_scaled, X_test_scaled, y_train, y_test)
