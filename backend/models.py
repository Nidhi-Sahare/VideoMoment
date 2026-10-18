from datetime import datetime,timezone
from sqlalchemy import Column,String,Integer,Float,Boolean,DateTime,ForeignKey,Text,JSON,Index
from sqlalchemy.orm import relationship
from database import Base,engine

def utc_now():
    return datetime.now(timezone.utc)

class Video(Base):
    __tablename__="videos"
    id=Column(String(64),primary_key=True)
    filename=Column(String(255),nullable=False)
    file_path=Column(Text,nullable=False)
    file_size=Column(Integer,nullable=True)
    duration=Column(Float,nullable=True)
    status=Column(String(30),nullable=False,default="uploaded")
    indexed=Column(Boolean,nullable=False,default=False)
    progress=Column(Integer,nullable=False,default=0)
    stage=Column(String(100),nullable=True)
    message=Column(Text,nullable=True)
    error=Column(Text,nullable=True)
    indexing_summary=Column(JSON,nullable=True)
    uploaded_at=Column(DateTime(timezone=True),default=utc_now,nullable=False)
    updated_at=Column(DateTime(timezone=True),default=utc_now,onupdate=utc_now,nullable=False)
    indexing_jobs=relationship("IndexingJob",back_populates="video",cascade="all, delete-orphan")
    searches=relationship("SearchHistory",back_populates="video",cascade="all, delete-orphan")

class IndexingJob(Base):
    __tablename__="indexing_jobs"
    id=Column(Integer,primary_key=True,autoincrement=True)
    video_id=Column(String(64),ForeignKey("videos.id"),nullable=False)
    status=Column(String(30),nullable=False,default="pending")
    progress=Column(Integer,nullable=False,default=0)
    stage=Column(String(100),nullable=True)
    message=Column(Text,nullable=True)
    error=Column(Text,nullable=True)
    started_at=Column(DateTime(timezone=True),nullable=True)
    completed_at=Column(DateTime(timezone=True),nullable=True)
    video=relationship("Video",back_populates="indexing_jobs")

class SearchHistory(Base):
    __tablename__="search_history"
    id=Column(Integer,primary_key=True,autoincrement=True)
    video_id=Column(String(64),ForeignKey("videos.id"),nullable=False)
    query=Column(Text,nullable=False)
    fusion_method=Column(String(50),nullable=False)
    top_k=Column(Integer,nullable=False)
    processing_time_seconds=Column(Float,nullable=True)
    total_results=Column(Integer,nullable=False,default=0)
    results=Column(JSON,nullable=True)
    created_at=Column(DateTime(timezone=True),default=utc_now,nullable=False)
    video=relationship("Video",back_populates="searches")

Index("ix_videos_status",Video.status)
Index("ix_indexing_jobs_video_id",IndexingJob.video_id)
Index("ix_search_history_video_id",SearchHistory.video_id)

def create_tables():
    print("\nConnecting to PostgreSQL...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")
    print("\nTables:")
    print(" - videos")
    print(" - indexing_jobs")
    print(" - search_history")

if __name__=="__main__":
    create_tables()