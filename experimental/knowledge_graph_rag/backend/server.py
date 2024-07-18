import subprocess
from fastapi import FastAPI
from routers import ui_backend,chat,evaluation

app = FastAPI()

# Include routers from different modules
app.include_router(ui_backend.router, prefix="/ui")
app.include_router(chat.router, prefix="/chat")
app.include_router(evaluation.router, prefix="/evaluation")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)