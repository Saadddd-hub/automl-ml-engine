from fastapi import APIRouter, BackgroundTasks
from app.services.job_service import create_job
from app.workers.training_worker import train_job

router = APIRouter()

@router.post("/{job_id}")
async def start_training(job_id: str, background_tasks: BackgroundTasks):

    file_path = f"data/uploads/{job_id}.csv"

    create_job(job_id)

    background_tasks.add_task(train_job, job_id, file_path)

    return {"message": "Training started", "job_id": job_id}