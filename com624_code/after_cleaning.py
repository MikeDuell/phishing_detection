def after_cleaning_empty_rows(data):
    print("\n---------\nEmpty rows - after cleaning \n--------\n")
    print(f"{data.isna().sum()}\n\n")
    return data.isna().sum()