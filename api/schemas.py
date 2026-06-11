from pydantic import BaseModel
from typing import List, Optional

class EvalJobRequest(BaseModel):
    prompts: List[str]
    model: str = "gemini-2.5-flash"
    max_tokens: int = 512

class EvalJobResponse(BaseModel):
    job_id: str
    status: str
    prompt_count: int

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    prompt_count: Optional[int] = None
    created_at: Optional[str] = None
