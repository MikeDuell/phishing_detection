import re
from urllib.parse import urlparse
import pandas as pd
from spellchecker import SpellChecker

phishing_keywords = [
    "urgent", "click", "verify", "account", "security",
    "required", "link", "invoice", "immediately", "expires", 'unusual',
    "payment", "confirmation", "free", "download", "delivery", "message", "package", "re"
]
spell = SpellChecker()


def count_spelling_errors(data):
    if pd.isna(data):
        return 0
    words = str(data).lower().split()
    misspelled = spell.unknown(words)
    return len(misspelled)


def has_body_keywords(data):
    # Compile a regex pattern that matches any keyword
    pattern = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in phishing_keywords) + r')\b', flags=re.IGNORECASE)

    # Apply to your text column
    data['has_body_keyword'] = data['clean_body'].fillna('').apply(lambda text: bool(pattern.search(text))).astype(int)
    data['keyword_count_body'] = data['clean_body'].fillna('').apply(lambda text: len(pattern.findall(text))).astype(
        int)
    return data


def has_subject_keywords(data):
    # Compile a regex pattern that matches any keyword

    pattern = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in phishing_keywords) + r')\b', flags=re.IGNORECASE)

    # Apply to your text column
    data['has_subject_keyword'] = data['clean_subject'].fillna('').apply(
        lambda text: bool(pattern.search(text))).astype(int)
    data['keyword_count_subject'] = data['clean_subject'].fillna('').apply(
        lambda text: len(pattern.findall(text))).astype(int)
    return data


def subject_length(data):
    # Ensure the column exists and is string-typed
    if 'clean_subject' in data.columns:
        data['clean_subject'] = data['clean_subject'].fillna('').astype(str)
    else:
        data['clean_subject'] = ''

    # Compute subject length
    data['subject_length'] = data['clean_subject'].apply(len)
    return data


def extract_sender_domain_features(df, sender_col="sender"):
    """
    Extracts domain-based features from sender email addresses.
    1)  extract domain
    2)  extract domain length
    3)  extract top level domain
    4)  extract if email domain is a free email
    5)  determine the frequency a domain has been received

    """

    df['sender'] = df[sender_col].apply(
        lambda x: str(x).split('@')[-1].lower() if pd.notnull(x) and '@' in str(x) else 'unknown')
    df['sender'] = df['sender'].str.strip().str.replace(r'[^\w\.]', '', regex=True)
    df['sender_length'] = df['sender'].apply(len)
    df['sender_tld'] = df['sender'].apply(lambda d: d.split('.')[-1] if '.' in d else 'unknown')
    free_email_domains = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com'}
    df['sender_is_free_email'] = df['sender'].isin(free_email_domains)
    sender_counts = df['sender'].value_counts()
    df['sender_domain_freq'] = df['sender'].map(sender_counts)
    return df


def is_weekend(data):
    if "day" in data.columns:
        data['is_weekend'] = data["day"].isin(['Sat', 'Sun']).astype(int)
    else:
        data['is_weekend'] = data["day"].dt.dayofweek >= 5  # 5 = Saturday, 6 = Sunday


def is_working_hours(data):
    # Parse into datetime temporarily
    parsed = pd.to_datetime(data["time"], format='%H:%M:%S', errors='coerce')
    data['sending_time'] = parsed.dt.time
    data['hour'] = parsed.dt.hour
    data['is_working_hour'] = data['hour'].between(9, 16).astype(int)
    return data


def extract_urls_from_dataframe(df: pd.DataFrame, column: str = "body"):
    """Extract full URLs from a DataFrame column and store them in a clean 'url' column."""
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'

    def clean_urls(text):
        urls = re.findall(url_pattern, str(text))
        if urls:
            return ", ".join(urls)  # Join multiple URLs into a single string
        else:
            return "no_link"  # Fallback if no URLs found

    df["url"] = df[column].apply(clean_urls)
    return df




def extract_url_features_from_df(df: pd.DataFrame, url_column="url"):
    """Extract phishing-relevant features from the URL column and add them as new columns to df."""

    def extract_features(url):
        try:
            parsed = urlparse(url if url.startswith("http") else "http://" + url)
            domain = parsed.netloc
            path = parsed.path
            full_url = url.lower()

            url_length = len(full_url)
            subdomain_count = domain.count('.') - 1
            path_length = len(path)
            digit_count_domain = sum(c.isdigit() for c in domain)
            uses_https = int(parsed.scheme == "https")
            sender_freq_placeholder = 1  # Replace with actual sender frequency if available

            # Derived features
            subdomain_density = subdomain_count / url_length if url_length else 0
            path_to_domain_ratio = path_length / (subdomain_count + 1)  # avoid div by zero
            is_long_url = int(url_length > 100)
            is_foreign_sender = None  # Placeholder if time_zone_diff is available elsewhere
            https_with_long_url = int(uses_https and url_length > 100)
            rare_domain_with_subdomains = int(sender_freq_placeholder < 5 and subdomain_count > 2)

            return {
                "url_length": url_length,
                "has_at_symbol": int('@' in full_url),
                "subdomain_count": subdomain_count,
                "path_length": path_length,
                "digit_count_domain": digit_count_domain,
                "uses_https": uses_https,
                "suspicious_tld": int(domain.endswith((".xyz", ".top", ".info", ".club"))),
                "subdomain_density": subdomain_density,
                "path_to_domain_ratio": path_to_domain_ratio,
                "is_long_url": is_long_url,
                "https_with_long_url": https_with_long_url,
                "rare_domain_with_subdomains": rare_domain_with_subdomains
            }

        except Exception:
            # Return default values if URL is malformed
            return {
                "url_length": 0,
                "has_at_symbol": 0,
                "subdomain_count": 0,
                "path_length": 0,
                "digit_count_domain": 0,
                "uses_https": 0,
                "suspicious_tld": 0,
                "subdomain_density": 0,
                "path_to_domain_ratio": 0,
                "is_long_url": 0,
                "https_with_long_url": 0,
                "rare_domain_with_subdomains": 0
            }

    # Apply feature extraction and expand into new columns
    feature_df = df[url_column].fillna("").apply(extract_features).apply(pd.Series)

    # Merge new columns into original DataFrame
    for col in feature_df.columns:
        df[col] = feature_df[col]

    return df


