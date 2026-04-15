from fastapi import APIRouter
from app.services.job_service import get_job

router = APIRouter()

@router.get("/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)
    return job if job else {"error": "Not found"}