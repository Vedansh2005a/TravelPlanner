from pathlib import Path
import os
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from backend import run_agent

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="TRAVEL PLANNER",
    description="LANGGRAPH MULTI AGENT TRAVEL PLANNER WITH FASTAPI FRONTEND",
    version="1.0.0"
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

template = Jinja2Templates(directory=str(BASE_DIR / "templates"))

class TravelRequest(BaseModel):
    query: str
    thread_id: str | None = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return template.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.query.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty."
                }
            )

        result = run_agent(
            user_input=user_message,
            thread_id=request_data.thread_id
        )

        # Using .get() prevents KeyError crashes if the backend omits a field
        return JSONResponse(
            content={
                "success": True,
                "thread_id": result.get("thread_id"),
                "answer": result.get("answer"),
                "flight_results": result.get("flight_results"),
                "hotel_results": result.get("hotel_results"),
                "itinerary": result.get("itinerary"),
                "llm_calls": result.get("llm_calls"),
            }
        )

    except Exception as e:
        print("ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "AI Travel Planner API is running"
    }

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )