from app.services.job_service import update_job
from ml_core.pipeline import run_training_pipeline

def train_job(job_id, file_path):
    update_job(job_id, {"status": "running"})

    result = run_training_pipeline(
        file_path,
        job_id,
        update_fn=update_job
    )

    update_job(job_id, result)