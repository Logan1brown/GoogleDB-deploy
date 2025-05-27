"""FastAPI proxy server for RT score collection."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import json
import os
from pathlib import Path

app = FastAPI()

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CORS to accept requests from RT domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.rottentomatoes.com",
        "http://localhost:8501",  # Streamlit default port
        "http://127.0.0.1:8501",
        "*"  # Allow all origins for testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting RT Score Proxy server...")
    logger.info("CORS enabled for: localhost:8501, rottentomatoes.com")

class RTScores(BaseModel):
    """RT scores with capture time"""
    title: str
    tomatometer: Optional[int]
    audience: Optional[int]

# Ensure scores directory exists
SCORES_DIR = Path(__file__).parent / "scores"
SCORES_DIR.mkdir(exist_ok=True)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check request received")
    response = {"status": "healthy", "timestamp": datetime.now().isoformat()}
    logger.info(f"Health check response: {response}")
    return response

@app.post("/submit-scores")
async def submit_scores(data: RTScores):
    """Handle score submission from RT pages."""
    try:
        # Add timestamp
        score_data = {
            **data.dict(),
            "captured_at": datetime.now().isoformat()
        }
        
        # Save to file named by timestamp for uniqueness
        filename = f"scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(SCORES_DIR / filename, "w") as f:
            json.dump(score_data, f, indent=2)
            
        return {"status": "success", "message": "Scores saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")
