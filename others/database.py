import pandas as pd
import random, logging, json
from typing import List, Dict
from dataclasses import dataclass
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)

class TypeSubject(Enum):
    Picklist_value1 = "Picklist value1"
    Picklist_value2 = "Picklist value2"
    Picklist_value3 = "Picklist value3"


class TypeUse(Enum):
    Picklist_value1 = "Picklist value1"
    Picklist_value2 = "Picklist value2"

@dataclass
class Question:
    id: int
    use: TypeUse
    subject: TypeSubject
    question: str
    responseA: str
    responseB: str
    responseC: str
    responseD: str
    remark: str

class QuestionDatabase:
    def __init__(self, file_path: str):
        self.df = pd.read_excel(file_path)

    def get_questions(self, use: str, subjects: List[str], num_questions: int) -> List[Dict]:
        logger.debug(f"Filtering questions: use={use}, subjects={subjects}, num_questions={num_questions}")

        try:
            filtered_df = self.df[(self.df['use'] == use) & (self.df['subject'].isin(subjects))]
            logger.debug(f"Filtered dataframe shape: {filtered_df.shape}")

            if not filtered_df.empty:
                questions = []
                sample_size = min(num_questions, len(filtered_df))
                logger.debug(f"Sampling {sample_size} questions")

                for index, row in filtered_df.sample(n=sample_size).iterrows():
                    question_dict = {
                        "question": row['question'],
                        "responseA": row['responseA'],
                        "responseB": row['responseB'],
                        "responseC": row['responseC'],
                        "responseD": row['responseD'] if pd.notna(row['responseD']) else None,
                        "correct": row['correct'],
                        "remark": row['remark'] if pd.notna(row['remark']) else None
                    }
                    questions.append(question_dict)

                random.shuffle(questions)
                logger.debug(f"Returning {len(questions)} questions")
                return questions
            else:
                logger.warning("No questions found with desired criteria.")
                return []
        except Exception as e:
            logger.exception("An error occurred while processing questions")
            raise e