from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import upload, train, status, results

app = FastAPI(title="AutoML Backend")

# ✅ CORS CONFIG
origins = [
    "http://localhost:3000",   # React dev server
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # 👈 allow frontend
    allow_credentials=True,
    allow_methods=["*"],            # GET, POST, etc.
    allow_headers=["*"],            # all headers
)

# Routes
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(train.router, prefix="/train", tags=["Train"])
app.include_router(status.router, prefix="/status", tags=["Status"])
app.include_router(results.router, prefix="/results", tags=["Results"])

@app.get("/")
def home():
    return {"message": "AutoML Backend Running 🚀"}