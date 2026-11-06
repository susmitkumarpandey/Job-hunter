from typing import List, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class CandidateProfile(BaseModel):
    detected_role: str = Field(description="Inferred professional title or role (e.g. 'Senior Python Backend Engineer')")
    years_of_experience: str = Field(description="Estimated years or seniority level (e.g., '3-5 years', 'Entry level')")
    top_skills: List[str] = Field(description="Top 5-7 technical skills or technologies")
    domains: List[str] = Field(description="Industry domains or focus areas, e.g. ['Fintech', 'Cloud Infrastructure']")


class SearchQueries(BaseModel):
    queries: List[str] = Field(
        description="2-3 targeted web search queries to find active hiring openings on LinkedIn, Greenhouse, Lever, etc."
    )


class JobOpening(BaseModel):
    title: str = Field(description="Job title or designation")
    company: str = Field(description="Hiring company name")
    experience_required: str = Field(description="Experience required, or 'NA' if unspecified")
    package: str = Field(default="NA", description="Salary or compensation package if explicitly stated, else 'NA'")
    apply_link: str = Field(description="Direct URL or job application link")


class JobBatch(BaseModel):
    jobs: List[JobOpening] = Field(default_factory=list, description="List of validated job openings found")


class AgentState(TypedDict):
    resume_raw_text: str
    target_role: Optional[str]
    candidate_profile: Optional[CandidateProfile]
    search_queries: List[str]
    raw_search_results: List[dict]
    final_jobs: List[JobOpening]


class JobSearchResponse(BaseModel):
    candidate_profile: Optional[CandidateProfile]
    search_queries_used: List[str]
    total_jobs_found: int
    jobs: List[JobOpening]