from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import List
import pandas as pd
import os, requests, json, random, logging
from schemas import Question, QuestionCreate
from auth import authenticate_user
from database import QuestionDatabase

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Market Predictions", description="API for requesting and getting Stock Market Price Predictions")

security = HTTPBasic()

users_db = {
    "admin": "admin",
    "fabian": "fabianpass",
    "mehdi": "mehdipass",
    "leo": "leopass",
    "florian": "florianpass"
}

current_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(current_dir, 'questions_en.xlsx')

# Pull the data :
'''if not os.path.exists(excel_path):
    # url = "https://dst-de.s3.eu-west-3.amazonaws.com/fastapi_en/questions_en.xlsx"
    try:
        response = requests.get(url)
        with open(excel_path, 'wb') as file:
            file.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

df = pd.read_excel(excel_path)
'''

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "OK", "message": "API is functional."}

def get_questions_from_database(use: str, subjects: List[str], num_questions: int) -> str:
    question_db = QuestionDatabase(excel_path)
    questions = question_db.get_questions(use, subjects, num_questions)
    return json.dumps(questions)

@app.get("/AAPL/", response_model=List[Question], tags=["Apple"])
def get_questions(
        use: str,
        subjects: List[str] = Query(..., min_length=1),
        num_questions: int = Query(..., gt=0, le=20),
        credentials: HTTPBasicCredentials = Depends(security)
):
    try:
        if not authenticate_user(credentials.username, credentials.password, users_db):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )

        #
        logger.debug(f"Fetching questions: use={use}, subjects={subjects}, num_questions={num_questions}")
        json_questions = get_questions_from_database(use, subjects, num_questions)
        logger.debug(f"Received data: {json_questions}")
        #

        questions = json.loads(json_questions)
        if isinstance(questions, dict) and "error" in questions:
            raise HTTPException(status_code=404, detail=questions["error"])

        return data

    except Exception as e:
        logger.exception("An error occurred while processing the request")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/AAPL/", response_model=List[Question], tags=["Apple"])
def get_questions(
        use: str,
        subjects: List[str] = Query(..., min_length=1),
        num_questions: int = Query(..., gt=0, le=20),
        credentials: HTTPBasicCredentials = Depends(security)
):
    try:
        if not authenticate_user(credentials.username, credentials.password, users_db):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
'''
        logger.debug(f"Fetching questions: use={use}, subjects={subjects}, num_questions={num_questions}")
        # json_questions = get_questions_from_database(use, subjects, num_questions)
        logger.debug(f"Received data: {json_questions}")

        questions = json.loads(json_questions)
'''
        if isinstance(questions, dict) and "error" in questions:
            raise HTTPException(status_code=404, detail=questions["error"])

        return data
    except Exception as e:
        logger.exception("An error occurred while processing the request")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/AAPL/Filter/", response_model=Question, tags=["Filter"])
def create_question(
    question: QuestionCreate,
    credentials: HTTPBasicCredentials = Depends(security)
):
    if not authenticate_user(credentials.username, credentials.password, users_db) or credentials.username != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials or not an admin",
            headers={"WWW-Authenticate": "Basic"},
        )
'''
    new_question = question.dict()
    df.loc[len(df)] = new_question
    df.to_excel(excel_path, index=False)
    return new_question
'''

@app.post("/GOOGL/Filter/", response_model=Question, tags=["Filter"])
def create_question(
    question: QuestionCreate,
    credentials: HTTPBasicCredentials = Depends(security)
):
    if not authenticate_user(credentials.username, credentials.password, users_db) or credentials.username != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials or not an admin",
            headers={"WWW-Authenticate": "Basic"},
        )
'''
    new_question = question.dict()
    df.loc[len(df)] = new_question
    df.to_excel(excel_path, index=False)'''
    return new_question

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # http://127.0.0.1:8000/docs for the Swagger UI