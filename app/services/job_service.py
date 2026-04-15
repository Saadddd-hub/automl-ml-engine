jobs = {}

def create_job(job_id):
    jobs[job_id] = {
        "status": "pending",
        "progress": 0
    }

def update_job(job_id, data):
    jobs[job_id].update(data)

def get_job(job_id):
    return jobs.get(job_id)