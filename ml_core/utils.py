def detect_target_column(df):
    columns = df.columns

    # Priority keywords
    keywords = ["target", "label", "class", "output"]

    # Step 1: Check keyword match
    for col in columns:
        if any(keyword in col.lower() for keyword in keywords):
            return col

    # Step 2: Find categorical columns with low unique values
    for col in columns:
        unique_vals = df[col].nunique()
        if df[col].dtype == "object" and unique_vals < 20:
            return col

    # Step 3: Find numeric column with low unique values (classification)
    for col in columns:
        unique_vals = df[col].nunique()
        if unique_vals < 20:
            return col

    # Step 4: fallback → last column
    return columns[-1]