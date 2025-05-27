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

# Configure CORS to accept requests from RT domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.rottentomatoes.com"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

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
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

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
    uvicorn.run(app, host="127.0.0.1", port=3000)
