import os
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware, ModelRetryMiddleware
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import tempfile
import streamlit as st
from langchain_groq import ChatGroq

load_dotenv()

# os.environ['GOOGLE_API_KEY'] = os.getenv('gemini_key')
# google_api_key = os.getenv("gemini_key")

os.environ["GROQ_API_KEY"] = os.getenv("groq_api_key")



# llm = "google_genai:gemini-2.5-flash-lite"
# llm = "google_genai:gemini-2.0-flash"
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)


search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

sked_agent = create_agent(
    model= llm,
    tools=[wikipedia],
    middleware=[
        ModelRetryMiddleware(max_retries=1),
        ToolRetryMiddleware(
            max_retries=1,
            retry_on=(ConnectionError, TimeoutError),
        ),
    ],
    system_prompt="""
    You are a resume evaluation assistant focused on skills and education matching.

    Use Wikipedia only when it helps verify broad skill/discipline context, not for opinion or speculative judgments.

    Task:
    - Compare resume skills and education with the job description.
    - Extract explicit matches, missing skills, and education relevance.
    - Be concise, factual, and structured.

    Return format:
    MATCH: XX%
    MISSING: [list]
    RATING: X/10
    NOTES: short explanation
    """
    )

exp_agent = create_agent(
    model=llm,
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=1),
        ToolRetryMiddleware(
            max_retries=1,
            retry_on=(ConnectionError, TimeoutError),
        ),
    ],
    system_prompt="""
    You are a resume evaluation assistant focused on work experience matching.

    Use web search only for public market or role-context references when needed.
    Do not invent experience. Only infer from the resume and job description.

    Task:
    - Compare years of experience, role alignment, seniority, and domain fit.
    - Highlight relevant companies, titles, and scope.
    - Return a concise experience fit summary.

    Return format:
    FIT: XX%
    COMPANIES: [list]
    ROLES: [list]
    RATING: X/10
    NOTES: short explanation
    """
    )

sal_agent = create_agent(
    model=llm,
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=1),
        ToolRetryMiddleware(
            max_retries=1,
            retry_on=(ConnectionError, TimeoutError),
        ),
    ],
    system_prompt="""
    You are a compensation research assistant.

    Use web search only for public salary-market context.
    Do not guess salary. If exact salary data is unavailable, provide an estimated range and clearly label it as an estimate.

    Task:
    - Estimate market-aligned salary range for the role, location, and years of experience.
    - Mention confidence level and whether the profile seems above, below, or aligned with the market.

    Return format:
    RANGE: X - Y LPA
    PERCENTAGE HIKE: XX%
    CONFIDENCE: LOW/MEDIUM/HIGH
    NOTES: short explanation
    """
    )

@tool
def call_skill_edu_matcher(resume: str, job_desc: str) -> str:
    """Analyze skill and education fit between resume and job description."""
    response = sked_agent.invoke({
        "messages": [
            HumanMessage(content=f"""
    Resume:
    {resume}

    Job Description:
    {job_desc}

    Evaluate skill and education fit only.
    """)
            ]
        })
    return response["messages"][-1].content

@tool
def call_exp_matcher(resume_exp: str, job_role: str) -> str:
    """Analyze work experience fit for the role."""
    response = exp_agent.invoke({
        "messages": [
            HumanMessage(content=f"""
    Resume Experience:
    {resume_exp}

    Target Role:
    {job_role}

    Evaluate experience fit only.
    """)
            ]
        })
    return response["messages"][-1].content

@tool
def call_sal_matcher(role: str, location: str, years: float) -> str:
    """Research salary alignment for role, location, and years of experience."""
    response = sal_agent.invoke({
        "messages": [
            HumanMessage(content=f"""
    Role: {role}
    Location: {location}
    Years of Experience: {years}

    Estimate market salary alignment.
    """)
            ]
        })
    return response["messages"][-1].content

# Supervisor Output Schema
class ResumeDecision(BaseModel):
    decision: str = Field(description="APPROVE or REJECT")
    score: int = Field(description="Overall score out of 100")
    summary: str = Field(description="Short recruiter-style summary")
    skill_fit: str = Field(description="Skill and education match summary")
    experience_fit: str = Field(description="Experience match summary")
    salary_fit: str = Field(description="Salary alignment summary")

# Supervisor Agent
sup_agent = create_agent(
    model=llm,
    tools=[call_skill_edu_matcher, call_exp_matcher, call_sal_matcher],
    response_format=ResumeDecision,
    middleware=[
        ModelRetryMiddleware(max_retries=1),
        ToolRetryMiddleware(
            max_retries=1,
            retry_on=(ConnectionError, TimeoutError),
        ),
    ],
    system_prompt="""
    You are a senior recruiting manager screening resumes for shortlisting.

    Workflow:
    1. Evaluate skill and education fit first.
    2. Evaluate experience fit second.
    3. Evaluate salary-market alignment third.
    4. Make a final decision only after reviewing all three signals.
    5. Never call any tool other than the three tools listed above.

    Decision rules:
    - APPROVE if skills are strong, experience is relevant, and compensation seems aligned.
    - REJECT if core requirements are missing or experience is too far from the role.
    - Use a score from 0 to 100.
    - Be strict, realistic, and concise.

    Output a structured final decision only.
    """
    )

# Document loader
def extract_text_from_uploaded_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        text = "\n\n".join(doc.page_content for doc in docs)
        return text
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


st.title("🤖 AI Resume Analysis Tool")

resume_file = st.file_uploader("Upload Resume PDF", type="pdf")
jd_file = st.file_uploader("Upload Job Description PDF", type="pdf")

if resume_file and jd_file:
    if st.button("Evaluate"):
        resume = extract_text_from_uploaded_pdf(resume_file)
        job_desc = extract_text_from_uploaded_pdf(jd_file)

        response = sup_agent.invoke({
            "messages": [
                HumanMessage(
                    content=f"Screen this resume:\n\n{resume}\n\nAgainst this job description:\n\n{job_desc}"
                )
            ]
        })

        st.write(response)
        st.stop()

        st.subheader("📄 Resume Screening Report")

        st.success("Analysis Completed Successfully ✅")

        st.markdown("## 🎯 Final Decision")
        st.write(result.decision)

        st.markdown("## 📊 Overall Score")
        st.write(f"{result.score}/100")

        result = response["messages"][-1].content
        parts = result.split("NOTES:")

        details = parts[0]
        notes = parts[1] if len(parts) > 1 else ""

        st.markdown("## 💰 Salary Analysis")
        st.info(details)

        st.markdown("## 📝 Detailed Notes")
        st.write(notes)
#         try:
#     response = sup_agent.invoke({
#         "messages": [
#             HumanMessage(
#                 content=f"Screen this resume:\n\n{resume}\n\nAgainst this job description:\n\n{job_desc}"
#             )
#         ]
#     })

#     result = response["messages"][-1].content

#     st.write("## Resume Screening Result")
#     st.write(result)

# except Exception as e:
#     st.error(f"Error: {e}")
