import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cleaning import split_date, normalise_time, clean_text, mask_email_domain
from email_message import fetch_messages
from feature_engineering import subject_length, has_subject_keywords, has_body_keywords, is_weekend, is_working_hours, \
    extract_urls_from_dataframe, extract_url_features_from_df, extract_sender_domain_features, count_spelling_errors
from supervised_learning import evaluate_model
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

subjects = []
dates = []
senders = []
bodies = []
df_email = None

import streamlit as st
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://mail.google.com/']

def gmail_authenticate():
    creds_info = dict(st.secrets["gmail_oauth"])
    flow = InstalledAppFlow.from_client_config({"installed": creds_info}, SCOPES)

    # If already authenticated, reuse credentials
    if "gmail_creds" in st.session_state:
        return build('gmail', 'v1', credentials=st.session_state["gmail_creds"])

    # Otherwise, prompt user
    if st.button("Authenticate with Google"):
        creds = flow.run_console()   # prints a link + asks for code
        st.session_state["gmail_creds"] = creds
        return build('gmail', 'v1', credentials=creds)

    st.info("Click the button above to authenticate with Google.")
    return None

if "emails" not in st.session_state:
    st.session_state["emails"] = []

st.set_page_config(page_title="Phishing Detection (Robust)", layout="wide")
st.title("Phishing Detection — Full Pipeline (Robust & Defensive)")

if st.button("Fetch emails from Gmail"):
    st.write("Fetching emails from Gmail")
    st.session_state["emails"] = fetch_messages()
    print(len(st.session_state["emails"]))

for email in st.session_state["emails"]:
    subject = email["headers"].get("Subject")
    date = email["headers"].get("Date")
    sender = email["headers"].get("From")
    body = email["body"]
    subjects.append(subject)
    bodies.append(body)
    senders.append(sender)
    dates.append(date)

df_email = pd.DataFrame({
    "subject": subjects,
    "body": bodies,
    "sender": senders,
    "date": dates
})
if len(st.session_state["emails"]) > 0:
    st.dataframe(df_email)
    split_date(df_email)
    normalise_time(df_email, clean_date_col='clean_date', time_col='time', zone_col='zone', verbose=True)
    df_email['clean_body'] = df_email['body'].apply(clean_text)
    df_email['clean_subject'] = df_email['subject'].apply(clean_text)
    df_email['sender'] = df_email['sender'].apply(mask_email_domain)
    df_email['has_spelling_errors'] = df_email['clean_body'].apply(count_spelling_errors)
    subject_length(df_email)
    has_subject_keywords(df_email)
    has_body_keywords(df_email)
    is_weekend(df_email)
    is_working_hours(df_email)
    extract_urls_from_dataframe(df_email)
    extract_url_features_from_df(df_email)
    extract_sender_domain_features(df_email, 'sender')
    #print(df_email.info())

if "training_data" not in st.session_state:
    st.session_state["training_data"] = pd.DataFrame()
print(len(st.session_state["training_data"]))
if st.button("Train Model"):
    if len(st.session_state["training_data"]) == 0:
        st.session_state["training_data"] = pd.read_csv('com624_code/data/feature_added_clean_data.csv',
                                                        encoding='ISO-8859-1',
                                                        dtype={"label": "int8", "urls": "int8"},
                                                        na_values=['na', 'NA', 'Unknown', ''])
        st.session_state["training_data"].columns = st.session_state[
            "training_data"].columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]',
                                                                                               '', regex=True)

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
features_super_learning_defs = {
    0: "Binary flag indicating if the email body contains spelling mistakes - phishing attempts can have bad grammar and spelling mistakes",
    1: "Number of suspicious keywords in the email body - list of suspicious keywords found in the body",
    2: "Number of suspicious keywords in the subject line - list of suspicious keywords found in the subject line",
    3: "Length of the subject line in characters - phishing can contain very short phishing lines or excessively long subject lines",
    4: "Length of the sender’s email address - Very long addresses may indicate auto‑generated or obfuscated accounts. Legitimate senders usually have concise, branded addresses.",
    5: "Frequency of the sender’s domain in the dataset -Rare domains (appearing only once or twice in the dataset) are more likely to be malicious compared to common, trusted domains.",
    6: "Flag for whether the email was sent on a weekend",
    7: "Flag for whether the email was sent during working hours",
    8: "Hour of the day the email was sent",
    9: "Flag for whether the sender uses a free email provider  - Attackers often use Gmail, Yahoo, Outlook, etc. instead of corporate domains.",
    10: "Length of the URL(s) found in the email",
    11: "Flag for whether URLs contain an '@' symbol",
    12: "Number of subdomains in the sender’s domain",
    13: "Length of the URL path component - Long, convoluted paths can disguise malicious intent.",
    14: "Number of digits in the domain name",
    15: "Flag for whether the URL uses HTTPS",
    16: "Flag for whether the domain uses a suspicious TLD",
    17: "Ratio of subdomain length to total domain length - Overly long subdomains relative to domain suggest obfuscation.",
    18: "Ratio of URL path length to domain length -  Suspiciously long paths compared to domain size can indicate phishing..",
    19: "Flag for whether the URL length exceeds a threshold",
    20: "Flag for HTTPS URLs that are unusually long",
    21: "Flag for rare domains that also have multiple subdomains"
}



if len(st.session_state["training_data"] == 0):
    st.metric("data loaded from file for training contains ", len(st.session_state["training_data"]))
    X = st.session_state["training_data"][features_super_learning]  # Feature matrix for supervised learning
    y = st.session_state['training_data']['label']  # Target labels
    # Step 8: Split data into training and testing sets and scale features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    results = {}  # Store evaluation results
    predictions = {}  # Store predictions
    trained_models = {}
    # Loop through each classical model
    if trained_models == {}:
        for name, model in models.items():
            m = model.fit(X_train_scaled, y_train)  # Train model

            # After training your model

            y_pred = model.predict(X_test_scaled)

            predictions[name] = y_pred
            print(f"\n{name} Classification Report:\n")
            print(classification_report(y_test, y_pred, zero_division=0))

            metrics = evaluate_model(y_test, y_pred)
            trained_models[name] = m
            if trained_models != {}:
                st.info(f"{name} Model has been trained")
            results[name] = metrics
            st.write(metrics)


            # --- Feature Importance Section ---
            try:
                # Case 1: Tree-based models (RandomForest, XGBoost, etc.)
                if hasattr(m, "feature_importances_"):
                    importances = m.feature_importances_
                # Case 2: Linear models (LogisticRegression, LinearSVC, etc.)
                elif hasattr(m, "coef_"):
                    importances = abs(m.coef_[0])
                # Case 3: Fallback to permutation importance
                else:
                    perm = permutation_importance(m, X_test_scaled, y_test, n_repeats=10, random_state=42)
                    importances = perm.importances_mean

                # Plot feature importance
                fig, ax = plt.subplots()
                ax.barh(range(len(importances)), importances)
                ax.set_yticks(range(len(importances)))
                ax.set_yticklabels(features_super_learning)  # replace with your feature list
                ax.set_xlabel("Importance")
                ax.set_title(f"{name} Feature Importance")
                st.pyplot(fig)



            except Exception as e:
                st.warning(f"Feature importance not available for {name}: {e}")

    if len(st.session_state["emails"]) > 0:
        chosen_recipe = st.selectbox("choose an algorithm:",
                                     ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'XGBoost'])

        X_email = df_email[features_super_learning]
        X_email_scaled = scaler.transform(X_email)
        rf_model = trained_models[chosen_recipe]
        y_new = rf_model.predict(X_email_scaled)
        df_email['pred_phishing'] = y_new
        y_prob = rf_model.predict_proba(X_email_scaled)[:, 1]
        df_email['pred_prob'] = y_prob
        try:
            # Case 1: Tree-based models (RandomForest, XGBoost, etc.)
            if hasattr(rf_model, "feature_importances_"):
                importances = rf_model.feature_importances_
            # Case 2: Linear models (LogisticRegression, LinearSVC, etc.)
            elif hasattr(rf_model, "coef_"):
                importances = abs(rf_model.coef_[0])
            # Case 3: Fallback to permutation importance
            else:
                perm = permutation_importance(rf_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
                importances = perm.importances_mean

            top3 = sorted(enumerate(importances), key=lambda x: x[1], reverse=True)[:3]
            print(top3)

            st.write(f" YOUR EMAILS THAT ARE CLASSIFIED AS PHISHING ARE DUE TO PROMINENCE OF THESE FEATURES ")

            st.error(features_super_learning_defs[top3[1][0]])
            st.error(features_super_learning_defs[top3[2][0]])
            st.error(features_super_learning_defs[top3[0][0]])
        except Exception as e:
            st.warning(f"Feature importance not available for {chosen_recipe}: {e}")

        st.dataframe(df_email[['subject', 'pred_phishing', 'pred_prob']])
