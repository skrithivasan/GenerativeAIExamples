import subprocess
from fastapi import FastAPI
from routers import ui_backend,chat,evaluation
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Include routers from different modules
app.include_router(ui_backend.router, prefix="/ui")
app.include_router(chat.router, prefix="/chat")
app.include_router(evaluation.router, prefix="/evaluation")
app.include_router(visualization.router, prefix="/visualization", tags=["visualization"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)