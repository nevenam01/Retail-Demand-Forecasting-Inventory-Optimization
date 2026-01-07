import pandas as pd


def split_data(df, target_col, date_col='WeekStartDate', item_col='item_nbr', test_ratio=0.2):
    """
    Splits data by item_nbr so that the last 20% of weeks become the test set.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        date_col: Name of date column (must be datetime)
        item_col: Name of grouping column (e.g., item_nbr)
        test_ratio: Proportion of data for testing (e.g., 0.2 for 20%)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    df_sorted = df.sort_values(by=[item_col, date_col]).copy()

    train_parts = []
    test_parts = []

    for item_id, group in df_sorted.groupby(item_col):
        n = len(group)
        test_size = int(n * test_ratio)
        
        train_part = group.iloc[:-test_size]
        test_part = group.iloc[-test_size:]

        train_parts.append(train_part)
        test_parts.append(test_part)

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test

