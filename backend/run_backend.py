import subprocess, sys, time, os
# Wrapper to run the backend and log errors
from main import app
import uvicorn
uvicorn.run(
    app,
    host=os.environ.get("AVALON_HOST", "127.0.0.1"),
    port=int(os.environ.get("AVALON_PORT", "8771")),
    log_level="info",
)
