import seaborn as sns
import matplotlib.pyplot as plt


def run_feature_eda(df):
    """
    Visualizes and interprets phishing-relevant features:
    - spelling errors
    - keyword flags
    - subject length
    - domain features
    - temporal signals
    Assumes 'label' column exists (1 = phishing, 0 = legitimate).
    """



    if 'label' in df.columns:
        sns.countplot(x="label", data=df, palette="Set2")
        plt.title("Count of Phishing/not Phishing values after cleaning")
        plt.xlabel("phishing")
        plt.ylabel("Count")
        plt.show()

    if 'spelling_errors' in df.columns:
        print("\n🔍 Spelling Error Distribution")
        sns.histplot(df['spelling_errors'], bins=30)
        plt.title("Spelling Errors in Email Body")
        plt.show()

        sns.boxplot(data=df, x='label', y='spelling_errors')
        plt.title("Spelling Errors by Label")
        plt.show()

    for col in ['has_body_keyword', 'has_subject_keyword']:
        if col in df.columns:
            print(f"\n Keyword Flag: {col}")
            sns.countplot(data=df, x=col, hue='label')
            plt.title(f"{col} vs Label")
            plt.show()

    for col in ['keyword_count_body', 'keyword_count_subject']:
        if col in df.columns:
            print(f"\n Keyword Count: {col}")
            sns.boxplot(data=df, x='label', y=col)
            plt.title(f"{col} by Label")
            plt.show()

    if 'subject_length' in df.columns:
        print("\n Subject Length Distribution")
        sns.histplot(df['subject_length'], bins=30)
        plt.title("Subject Length")
        plt.show()

        sns.boxplot(data=df, x='label', y='subject_length')
        plt.title("Subject Length by Label")
        plt.show()

    if 'sender_is_free_email' in df.columns:
        print("\n Free Email Domain Flag")
        sns.countplot(data=df, x='sender_is_free_email', hue='label')
        plt.title("Free Email Domain vs Label")
        plt.show()

    if 'sender_domain_freq' in df.columns:
        print("\n Sender Domain Frequency")
        sns.boxplot(data=df, x='label', y='sender_domain_freq')
        plt.title("Sender Domain Frequency by Label")
        plt.yscale('log')  # log scale for skewed freq
        plt.show()

    if 'hour' in df.columns:
        print("\n Sending Hour")
        sns.boxplot(data=df, x='label', y='hour')
        plt.title("Sending Hour by Label")
        plt.show()

    if 'is_working_hour' in df.columns:
        print("\n  Working Hour Flag")
        sns.countplot(data=df, x='is_working_hour', hue='label')
        plt.title("Working Hour vs Label")
        plt.show()

    if 'is_weekend' in df.columns:
        print("\n Weekend Flag")
        sns.countplot(data=df, x='is_weekend', hue='label')
        plt.title("Weekend vs Label")
        plt.show()
    heatmap(df)
    print("\n Feature EDA complete.")


def heatmap(df):
    features = [
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
        'label'
    ]

    corr_matrix = df[features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.show()
