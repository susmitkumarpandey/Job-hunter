import os
import sys
from pathlib import Path
from dotenv import load_dotenv


# Add backend directory to sys.path so 'app' imports resolve correctly
backend_path = Path(__file__).resolve().parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))


env_path = backend_path / ".env"
load_dotenv(dotenv_path=env_path)

# Keep existing imports below
import io
import streamlit as st
import pandas as pd
import pdfplumber
import docx



# Now this import will resolve cleanly without module errors
from app.services.graph import build_job_discovery_graph

# Page configuration
st.set_page_config(
    page_title="AI Job Discovery Agent",
    page_icon="💼",
    layout="wide"
)

st.title("💼 AI Resume-Based Job Matcher")
st.write("Upload your resume and optionally specify a target role. Gemini 2.5 Flash and LangGraph will search the web for relevant openings.")

# ---------------------------------------------------------
# Sidebar: API Keys & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    google_key = st.text_input("Gemini / Google API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    tavily_key = st.text_input("Tavily Search API Key", type="password", value=os.getenv("TAVILY_API_KEY", ""))
    
    if google_key:
        os.environ["GEMINI_API_KEY"] = google_key
        os.environ["GOOGLE_API_KEY"] = google_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

# ---------------------------------------------------------
# Document Extraction Utility
# ---------------------------------------------------------
def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    
    if name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
        return "\n".join(pages).strip()
    
    elif name.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    
    elif name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    
    return ""

# ---------------------------------------------------------
# Main UI Form
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload your Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

with col2:
    target_role = st.text_input("Target Role (Optional)", placeholder="e.g. Lead Machine Learning Engineer")

search_clicked = st.button("🚀 Find Matching Jobs", type="primary", use_container_width=True)

if search_clicked:
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        st.error("Please provide a Gemini / Google API Key in the sidebar or `.env` file.")
        st.stop()
        
    if not os.getenv("TAVILY_API_KEY"):
        st.error("Please provide a Tavily API Key in the sidebar or `.env` file.")
        st.stop()

    if not uploaded_file:
        st.warning("Please upload a resume before starting the search.")
        st.stop()

    with st.spinner("Extracting text from resume..."):
        resume_text = extract_text_from_file(uploaded_file)
        
    if not resume_text:
        st.error("Unable to extract text from the uploaded document.")
        st.stop()

    with st.spinner("Analyzing resume with Gemini 2.5 Flash and searching web..."):
        try:
            # Build and invoke the graph
            graph_app = build_job_discovery_graph()
            initial_state = {
                "resume_raw_text": resume_text,
                "target_role": target_role.strip() if target_role.strip() else None,
                "candidate_profile": None,
                "search_queries": [],
                "raw_search_results": [],
                "final_jobs": []
            }
            
            final_state = graph_app.invoke(initial_state)
            
            # Show Extracted Profile
            profile = final_state.get("candidate_profile")
            if profile:
                with st.expander("🔍 Inferred Candidate Profile", expanded=False):
                    st.write(f"**Detected Role:** {profile.detected_role}")
                    st.write(f"**Experience:** {profile.years_of_experience}")
                    st.write(f"**Key Skills:** {', '.join(profile.top_skills)}")
                    st.write(f"**Domains:** {', '.join(profile.domains)}")

            # Display Search Queries Used
            queries = final_state.get("search_queries", [])
            if queries:
                with st.expander("🔎 Search Queries Generated", expanded=False):
                    for q in queries:
                        st.markdown(f"- `{q}`")

            # Display Results in a Table
            jobs = final_state.get("final_jobs", [])
            if jobs:
                st.success(f"Found {len(jobs)} active openings!")
                
                # Format into pandas dataframe for Streamlit interactive table
                table_data = []
                for job in jobs:
                    table_data.append({
                        "Job Title": job.title,
                        "Company": job.company,
                        "Experience Required": job.experience_required,
                        "Package": job.package,
                        "Apply Link": job.apply_link
                    })
                
                df = pd.DataFrame(table_data)
                
                # Render interactive table with clickable link column
                st.dataframe(
                    df,
                    column_config={
                        "Apply Link": st.column_config.LinkColumn(
                            "Apply URL",
                            display_text="Apply Here"
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No matching job postings found with direct apply links. Try adjusting your target role.")

        except Exception as e:
            st.error(f"Workflow execution failed: {str(e)}")