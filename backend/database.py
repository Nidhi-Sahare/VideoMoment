import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine,text
from sqlalchemy.orm import DeclarativeBase,sessionmaker
from sqlalchemy.engine import make_url

# Load environment variables from project .env
PROJECT_DIR=Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR/".env")
DATABASE_URL=os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing. Add it to your .env file.")

# Database engine
engine=create_engine(DATABASE_URL,pool_pre_ping=True,pool_size=5,max_overflow=10,)

# Session factory
SessionLocal=sessionmaker(bind=engine,autoflush=False,expire_on_commit=False,)

# Base class for future database tables
class Base(DeclarativeBase):
    pass

# FastAPI dependency
def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Test PostgreSQL connection
def test_connection():
    with engine.connect() as connection:
        database_name=connection.execute(text("SELECT current_database()")).scalar_one()
        server_version=connection.execute(text("SHOW server_version")).scalar_one()
    return {
        "connected":True,
        "database":database_name,
        "postgresql_version":server_version,}

if __name__=="__main__":
    try:
        result=test_connection()
        print("\nDATABASE CONNECTION SUCCESSFUL")
        print("Database:",result["database"])
        print("PostgreSQL version:",result["postgresql_version"])
    except Exception as error:
        print("\nDATABASE CONNECTION FAILED")
        print(type(error).__name__+":",error)
        raise SystemExit(1)