import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def before_cleaning_empty_rows(data):
    print("\n---------\nEmpty rows - before cleaning  \n--------\n")
    print(f"{data.isna().sum()}\n\n")
    return data.isna().sum()

def before_cleaning_duplicated_columns(data):
    columns = [["subject"], ["body"]]
    results = {}
    for column in columns:
        col_name = column[0]
        count = data.duplicated(subset=column, keep=False).sum()
        results[col_name] = int(count)

        # Get duplicated rows for this column
        duplicated_rows = data[data[col_name].duplicated(keep=False)]

        # Count how many times each value appears
        value_counts = duplicated_rows[col_name].value_counts()

        # Count phishing occurrences per value
        phishing_counts = (
            duplicated_rows.groupby(col_name)['label']
            .sum()
            .sort_values(ascending=False)
        )

        # Combine into one DataFrame for clarity
        summary = (
            pd.DataFrame({
                'total_count': value_counts,
                'phishing_count': phishing_counts
            })
            .sort_values('total_count', ascending=False)
        )

        print(f"\n---------\nDuplicated {col_name} values with phishing counts:\n---------")
        print(summary.head(10))

    return results

def before_cleaning_plot_missing_values(df, title):
    missing = df.isnull().sum()
    missing = missing[missing >= 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=missing.index, y=missing.values, palette=None, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Missing Count")
    ax.set_xlabel("Columns")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    return fig

def before_phishing_label_bias(df):
    sns.countplot(x="label", data=df, palette="Set2")
    plt.title("Count of Phishing/not Phishing values before cleaning")
    plt.xlabel("phishing")
    plt.ylabel("Count")
    plt.show()