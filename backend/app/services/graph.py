import os
from typing import List, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END


# =====================================================================
# DATA MODELS & STATE SCHEMA
# =====================================================================

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


# =====================================================================
# LLM INITIALIZATION (GEMINI 2.5 FLASH)
# =====================================================================

# Requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable
# Replace the global llm = ChatGoogleGenerativeAI(...) line with:
def get_llm():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY is not set. "
            "Please provide it in backend/.env or the sidebar."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key,
    )

def parse_resume_node(state: AgentState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert technical recruiter and resume analyzer. "
            "Analyze the candidate's resume and extract structured profile attributes."
        ),
        ("human", "Resume Content:\n\n{resume_text}")
    ])
    llm = get_llm()
    extractor = prompt | llm.with_structured_output(CandidateProfile)
    profile: CandidateProfile = extractor.invoke({"resume_text": state["resume_raw_text"]})
    return {"candidate_profile": profile}


def build_queries_node(state: AgentState) -> dict:
    profile = state["candidate_profile"]
    target_role = state.get("target_role")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a job search specialist. Formulate 2 to 3 targeted web search queries "
            "designed to retrieve active job postings on platforms like LinkedIn, Lever, Greenhouse, or Indeed.\n"
            "Rules:\n"
            "- If a target role is provided, prioritize it alongside candidate skills.\n"
            "- If omitted, use the candidate's detected role and seniority.\n"
            "- Use search terms like 'hiring', 'openings', or 'site:linkedin.com/jobs'."
        ),
        (
            "human",
            "Candidate Profile:\n{profile}\n\nDesired Target Role: {target_role}"
        )
    ])
    llm = get_llm()
    query_generator = prompt | llm.with_structured_output(SearchQueries)
    result: SearchQueries = query_generator.invoke({
        "profile": profile.model_dump_json() if profile else "No profile available",
        "target_role": target_role if target_role else "Not specified (use detected profile)"
    })
    return {"search_queries": result.queries}


def search_web_node(state: AgentState) -> dict:
    """Executes the search queries against the web using Tavily Search."""
    # Requires TAVILY_API_KEY in the environment (.env)
    search_tool = TavilySearchResults(max_results=3)
    raw_results = []
    
    for query in state.get("search_queries", []):
        try:
            hits = search_tool.invoke({"query": query})
            if isinstance(hits, list):
                raw_results.extend(hits)
        except Exception as e:
            print(f"Search warning for query '{query}': {e}")
            
    return {"raw_search_results": raw_results}


def normalize_jobs_node(state: AgentState) -> dict:
    raw_results = state.get("raw_search_results", [])
    if not raw_results:
        return {"final_jobs": []}
    
    formatted_listings = "\n\n".join([
        f"URL: {res.get('url', 'NA')}\nSnippet: {res.get('content', '')}"
        for res in raw_results
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict job listing data parser. Extract individual active job openings "
            "from the search results into the required schema.\n"
            "Guidelines:\n"
            "- Extract: title, company, experience_required, package, and direct apply_link.\n"
            "- If 'package' (salary) is NOT mentioned, set it strictly to 'NA'.\n"
            "- If 'experience_required' is NOT mentioned, set it strictly to 'NA'.\n"
            "- Discard generic blog posts or aggregations that are not direct postings."
        ),
        ("human", "Search Results:\n\n{listings}")
    ])
    llm = get_llm()
    extractor = prompt | llm.with_structured_output(JobBatch)
    batch: JobBatch = extractor.invoke({"listings": formatted_listings})
    return {"final_jobs": batch.jobs}


# =====================================================================
# GRAPH COMPILATION
# =====================================================================

def build_job_discovery_graph():
    builder = StateGraph(AgentState)
    builder.add_node("parse_resume", parse_resume_node)
    builder.add_node("build_queries", build_queries_node)
    builder.add_node("search_web", search_web_node)
    builder.add_node("normalize_jobs", normalize_jobs_node)
    
    builder.add_edge(START, "parse_resume")
    builder.add_edge("parse_resume", "build_queries")
    builder.add_edge("build_queries", "search_web")
    builder.add_edge("search_web", "normalize_jobs")
    builder.add_edge("normalize_jobs", END)
    
    return builder.compile()