import pandas as pd
from pipeline import final_evaluation_pipeline

def predict_leakage(df, target_col, time_col):
    """
    Runs the leakage detection evaluation pipeline
    using the uploaded dataset.
    """

    try:
        # Ensure time column is datetime
        df[time_col] = pd.to_datetime(df[time_col])

        # Run your evaluation pipeline
        results = final_evaluation_pipeline(
            df=df,
            target_col=target_col,
            time_col=time_col
        )

        return results

    except Exception as e:

        return {
            "error": str(e)
        }