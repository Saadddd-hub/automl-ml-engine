from fastapi import APIRouter
from app.services.job_service import get_job

router = APIRouter()

@router.get("/{job_id}")
def get_results(job_id: str):
    job = get_job(job_id)

    if not job or job["status"] != "completed":
        return {"error": "Results not ready"}

    return {
        "target_column": job["target_column"],
        "task_type": job["task_type"],
        "best_model": job["best_model"],
        "accuracy": job["accuracy"],
        "model_path": job["model_path"],

        # 🔥 IMPORTANT FOR FRONTEND
        "leaderboard": job["leaderboard"],
        "cv_results": job["cv_results"],
        "metrics": job["metrics"]
    }