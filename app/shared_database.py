# This file contains the logic for interacting with the shared database.
# Version 0.1
# Fabian
# 2408021049

# app/database.py
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

database_app = FastAPI()

SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/dbname"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    status = Column(String)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@database_app.post("/create-task/")
def create_task(name: str, db: Session = Depends(get_db)):
    new_task = Task(name=name, status="pending")
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    return {"task_id": new_task.id, "name": new_task.name, "status": new_task.status}

@database_app.get("/task/{task_id}")
def get_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        return {"error": "Task not found"}
    return {"task_id": task.id, "name": task.name, "status": task.status}