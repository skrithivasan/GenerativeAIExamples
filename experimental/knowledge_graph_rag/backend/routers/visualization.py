from fastapi import APIRouter

router = APIRouter()

@router.get("/visualization")
async def get_visualization():
    return {"message": "Visualization endpoint"}