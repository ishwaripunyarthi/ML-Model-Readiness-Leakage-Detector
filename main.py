from fastapi import FastAPI, UploadFile, File
import pandas as pd
from app.predict import predict_leakage

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Leakage Detection API is running"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    target_col: str = "",
    time_col: str = ""
):
    try:

        # Read uploaded CSV
        df = pd.read_csv(file.file)

        print("Columns:", df.columns)

        # Run leakage detection
        results = predict_leakage(
            df,
            target_col,
            time_col
        )

        return results

    except Exception as e:

        print("ERROR:", e)

        return {
            "error": str(e)
        }