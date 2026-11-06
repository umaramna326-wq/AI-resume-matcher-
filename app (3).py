import json
import os
import re
from io import BytesIO

import plotly.graph_objects as go
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader


# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="AI Resume & Job Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: linear-gradient(135deg, #f7f9fc 0%, #eef3fa 100%); }
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
      h1, h2, h3 { font-family: 'Manrope', sans-serif; color: #14243a; }
      .hero {
        padding: 1.7rem 2rem; border-radius: 22px; color: white;
        background: linear-gradient(115deg, #14243a 0%, #174b68 58%, #167c80 100%);
        margin-bottom: 1.4rem; box-shadow: 0 12px 35px rgba(20,36,58,.14);
      }
      .hero h1 { color: white; margin: 0 0 .35rem 0; font-size: 2.15rem; }
      .hero p { color: #dceaf1; margin: 0; font-size: 1rem; }
      .section-card {
        background: white; padding: 1.15rem 1.25rem; border-radius: 16px;
        border: 1px solid #e5ebf2; margin-bottom: .8rem;
      }
      .muted { color: #65758b; font-size: .92rem; }
      div[data-testid="stMetric"] {
        background: white; border: 1px solid #e5ebf2; padding: 1rem;
        border-radius: 15px; box-shadow: 0 4px 14px rgba(20,36,58,.04);
      }
      .pill {
        display:inline-block; padding:.28rem .65rem; border-radius:999px;
        background:#e8f5f2; color:#146b5d; margin:.15rem .25rem .15rem 0;
        font-size:.86rem; font-weight:600;
      }
      .gap-pill {
        display:inline-block; padding:.28rem .65rem; border-radius:999px;
        background:#fff0ed; color:#a64232; margin:.15rem .25rem .15rem 0;
        font-size:.86rem; font-weight:600;
      }
      div.stButton > button[kind="primary"] {
        border-radius: 11px; font-weight: 700; min-height: 3rem;
        background: linear-gradient(90deg,#176b78,#16877e); border: 0;
      }
      [data-testid="stFileUploader"] { background: white; border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Helpers -----------------------------
MAX_FILE_SIZE = 10 * 1024 * 1024
MODEL_NAME = "llama-3.3-70b-versatile"


def get_api_key():
    """Read the key from Streamlit Secrets first, then environment variables."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


def extract_resume(uploaded_file):
    """Extract text from PDF, DOCX, or TXT files."""
    if uploaded_file.size > MAX_FILE_SIZE:
        raise ValueError("The uploaded file is larger than 10 MB. Please upload a smaller file.")

    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(raw))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages)
    elif name.endswith(".docx"):
        document = Document(BytesIO(raw))
        chunks = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                chunks.append(" | ".join(cell.text.strip() for cell in row.cells))
        text = "\n".join(chunks)
    elif name.endswith(".txt"):
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

    if not text.strip():
        raise ValueError(
            "No readable text was found. If this is a scanned PDF, run OCR on it first "
            "or paste the resume text manually."
        )
    return text.strip()


def clean_list(value):
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def parse_ai_json(content):
    """Parse JSON even if the model wraps it in Markdown fences."""
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise ValueError("The AI returned an unreadable response. Please try again.")


def analyze_resume(resume_text, job_description, api_key):
    client = Groq(api_key=api_key, timeout=90.0, max_retries=2)

    system_prompt = """
You are an expert, fair, evidence-based resume reviewer and career coach.
Compare the candidate resume with the supplied job description. Evaluate only
evidence present in the resume. Do not infer protected characteristics and do
not invent experience, qualifications, metrics, employers, or achievements.
A match score is an estimated guidance score, not a hiring decision or a real
ATS result. Missing information should be described as "not demonstrated"
rather than assumed absent. Return ONLY valid JSON with the exact schema below.

{
  "match_score": 0,
  "score_explanation": "brief explanation of the score and its limitations",
  "suitability_summary": "2-4 sentence balanced summary",
  "matching_skills": [{"skill":"...", "evidence":"..."}],
  "skill_gaps": [{"requirement":"...", "importance":"High|Medium|Low", "recommendation":"..."}],
  "strengths": ["..."],
  "experience_assessment": ["..."],
  "resume_improvements": [
    {"current_statement":"exact or concise description of weak wording, or 'No clear example found'",
     "suggested_statement":"truthful, ATS-friendly alternative",
     "why_it_is_better":"..."}
  ],
  "priority_action_plan": [
    {"priority":"1", "action":"...", "reason":"..."}
  ],
  "keywords_to_consider": ["..."],
  "interview_preparation": ["..."]
}

Rules:
- match_score must be an integer from 0 to 100.
- Include 3-8 items per list where the source material supports it; fewer is fine.
- For suggested statements, use placeholders such as [X%], [number], or [time saved]
  if a measurable result would help but is not supplied. Explicitly tell the user
  to replace placeholders only with verified facts.
- Do not suggest adding a skill unless the candidate genuinely has it or can
  acquire it; distinguish "learn/develop" from "add to resume".
- Keep feedback specific to this job description, constructive, and concise.
"""

    user_prompt = f"""
JOB DESCRIPTION:
\"\"\"
{job_description[:18000]}
\"\"\"

RESUME:
\"\"\"
{resume_text[:18000]}
\"\"\"

Analyze the match and return the required JSON object only.
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_completion_tokens=5000,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content or ""
    data = parse_ai_json(content)

    score = data.get("match_score", 0)
    try:
        data["match_score"] = max(0, min(100, int(float(score))))
    except (TypeError, ValueError):
        data["match_score"] = 0

    for key in (
        "matching_skills", "skill_gaps", "strengths", "experience_assessment",
        "resume_improvements", "priority_action_plan", "keywords_to_consider",
        "interview_preparation",
    ):
        if not isinstance(data.get(key), list):
            data[key] = []

    return data


def render_bullets(items, empty_message="No items returned for this section."):
    if not items:
        st.info(empty_message)
        return
    for item in items:
        st.markdown(f"- {item}")


def render_skill_tags(items, css_class="pill"):
    if not items:
        st.info("No items identified.")
        return
    html = "".join(
        f'<span class="{css_class}">{str(item).replace("<", "&lt;").replace(">", "&gt;")}</span>'
        for item in items
    )
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## 🎯 Match Studio")
    st.caption("Your AI-powered career assistant")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload or paste your resume.\n"
        "2. Add the target job description.\n"
        "3. Run the AI analysis.\n"
        "4. Review gaps and improve your resume."
    )
    st.markdown("---")
    st.markdown("**Supported formats**")
    st.write("PDF · DOCX · TXT")
    st.caption("Maximum upload size: 10 MB")
    st.markdown("---")
    st.caption("AI feedback is advisory. Always verify every suggested detail before using it.")

# ----------------------------- Header -----------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Resume & Job Matcher</h1>
      <p>Turn your resume into a targeted career advantage with evidence-based AI feedback.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="muted">Compare your experience with a specific role, uncover skill gaps, '
    'and get practical, ATS-oriented suggestions — without inventing credentials.</p>',
    unsafe_allow_html=True,
)

# ----------------------------- Inputs -----------------------------
left, right = st.columns([0.95, 1.05], gap="large")

with left:
    st.markdown("### 01 · Your resume")
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx", "txt"],
        help="PDF, Word DOCX, or plain text. Maximum 10 MB.",
    )
    if uploaded_file:
        st.success(f"File ready: {uploaded_file.name}")
    resume_manual = st.text_area(
        "Or paste your resume text",
        height=230,
        placeholder="Paste the full text of your resume here...",
        help="If both a file and pasted text are provided, the pasted text takes priority.",
    )

with right:
    st.markdown("### 02 · Target job")
    job_description = st.text_area(
        "Paste the complete job description",
        height=320,
        placeholder=(
            "Paste the job title, responsibilities, required qualifications, "
            "technical skills, and preferred experience..."
        ),
    )

st.markdown("")
consent = st.checkbox(
    "I understand that my resume text will be sent to Groq to generate this analysis.",
    value=False,
)

analyze_clicked = st.button(
    "✨ Analyze resume match",
    type="primary",
    use_container_width=True,
)

if analyze_clicked:
    resume_text = resume_manual.strip()
    if not resume_text and uploaded_file is not None:
        try:
            resume_text = extract_resume(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded resume: {exc}")

    if not consent:
        st.warning("Please confirm that you understand how your resume text is processed.")
    elif not resume_text:
        st.warning("Please upload a resume or paste your resume text.")
    elif not job_description.strip():
        st.warning("Please paste the job description before starting the analysis.")
    elif len(resume_text) < 80:
        st.warning("The resume text looks very short. Please provide more complete resume content.")
    elif len(job_description.strip()) < 80:
        st.warning("The job description looks very short. Please paste the full job requirements.")
    else:
        api_key = get_api_key()
        if not api_key:
            st.error(
                "Groq API key not found. Add GROQ_API_KEY to Streamlit Secrets "
                "or your environment variables, then restart the app."
            )
        else:
            try:
                with st.spinner("Reviewing your resume against the job requirements..."):
                    result = analyze_resume(resume_text, job_description.strip(), api_key)
                st.session_state["analysis_result"] = result
                st.session_state["analysis_job_title"] = "Target role"
                st.success("Analysis complete. Explore your results below.")
            except Exception as exc:
                message = str(exc)
                if "401" in message or "authentication" in message.lower():
                    st.error("Groq authentication failed. Check that your GROQ_API_KEY is correct.")
                elif "429" in message or "rate_limit" in message.lower():
                    st.error("Groq rate limit reached. Wait a moment and try again.")
                elif "model" in message.lower() and ("decommission" in message.lower() or "not found" in message.lower()):
                    st.error(
                        f"The configured model ({MODEL_NAME}) may be unavailable for your account. "
                        "Check Groq's current model availability."
                    )
                else:
                    st.error(f"Analysis failed: {message}")

# ----------------------------- Results -----------------------------
result = st.session_state.get("analysis_result")
if result:
    st.markdown("---")
    st.markdown("## Your analysis dashboard")
    st.caption("The match score is an AI-generated estimate based on the text provided, not a hiring prediction.")

    score = result.get("match_score", 0)
    score_col, summary_col = st.columns([0.85, 2.15], gap="large")

    with score_col:
        st.metric("Estimated match", f"{score}%")
        st.progress(score / 100)
        if score >= 80:
            st.success("Strong alignment")
        elif score >= 60:
            st.info("Moderate alignment")
        else:
            st.warning("More alignment needed")

    with summary_col:
        st.markdown("### Candidate suitability")
        st.write(result.get("suitability_summary", "No summary was provided."))
        if result.get("score_explanation"):
            st.caption(result["score_explanation"])

    tab_overview, tab_skills, tab_resume, tab_plan = st.tabs(
        ["📊 Overview", "🧩 Skills gap", "✍️ Resume improvements", "🚀 Action plan"]
    )

    with tab_overview:
        col1, col2 = st.columns(2, gap="large")
        matching = result.get("matching_skills", [])
        gaps = result.get("skill_gaps", [])

        with col1:
            st.markdown("### Your strengths")
            render_bullets(result.get("strengths", []))
            st.markdown("### Relevant experience")
            render_bullets(result.get("experience_assessment", []))

        with col2:
            st.markdown("### Matching skills")
            if matching:
                for item in matching:
                    if isinstance(item, dict):
                        st.markdown(f"**{item.get('skill', 'Skill')}**")
                        st.caption(item.get("evidence", "Evidence not specified."))
                    else:
                        st.markdown(f"- {item}")
            else:
                st.info("No matching skills were identified.")
            st.markdown("### Important gaps")
            if gaps:
                for item in gaps:
                    if isinstance(item, dict):
                        importance = item.get("importance", "Medium")
                        st.markdown(f"**{item.get('requirement', 'Requirement')}** · {importance} priority")
                        st.caption(item.get("recommendation", ""))
                    else:
                        st.markdown(f"- {item}")
            else:
                st.info("No major gaps were identified.")

        st.markdown("### Keywords to consider")
        render_skill_tags(result.get("keywords_to_consider", []))

        st.markdown("### Interview preparation")
        render_bullets(result.get("interview_preparation", []))

    with tab_skills:
        st.markdown("### Skills you have demonstrated")
        matched_names = [
            item.get("skill", "") if isinstance(item, dict) else str(item)
            for item in matching
        ]
        render_skill_tags(matched_names)

        st.markdown("### Skills or requirements to strengthen")
        gap_names = [
            item.get("requirement", "") if isinstance(item, dict) else str(item)
            for item in gaps
        ]
        render_skill_tags(gap_names, "gap-pill")

        if matching or gaps:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=["Demonstrated matches", "Requirements to strengthen"],
                        y=[len(matching), len(gaps)],
                        marker_color=["#16877e", "#e58b72"],
                        text=[len(matching), len(gaps)],
                        textposition="outside",
                    )
                ]
            )
            fig.update_layout(
                height=330,
                margin=dict(l=20, r=20, t=25, b=20),
                yaxis_title="Number of items",
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        for item in gaps:
            if isinstance(item, dict):
                priority = item.get("importance", "Medium")
                st.markdown(f"**{item.get('requirement', 'Requirement')}** · {priority}")
                st.write(item.get("recommendation", ""))

    with tab_resume:
        improvements = result.get("resume_improvements", [])
        if not improvements:
            st.info("No specific wording changes were returned.")
        for index, item in enumerate(improvements, start=1):
            if isinstance(item, dict):
                with st.expander(f"Suggestion {index}: {item.get('current_statement', 'Resume wording')}"):
                    st.markdown("**Suggested wording**")
                    st.write(item.get("suggested_statement", ""))
                    st.markdown("**Why this is stronger**")
                    st.write(item.get("why_it_is_better", ""))
            else:
                st.markdown(f"- {item}")

        st.warning(
            "Truth check: only use suggested wording that accurately reflects your real work. "
            "Replace placeholders such as [X%] only with metrics you can verify."
        )

    with tab_plan:
        actions = result.get("priority_action_plan", [])
        if not actions:
            st.info("No action plan was returned.")
        for index, item in enumerate(actions, start=1):
            if isinstance(item, dict):
                priority = item.get("priority", str(index))
                st.markdown(f"**Priority {priority} — {item.get('action', 'Next step')}**")
                st.caption(item.get("reason", ""))
                if index != len(actions):
                    st.markdown("---")
            else:
                st.markdown(f"{index}. {item}")

    with st.expander("View raw analysis data"):
        st.json(result)

    st.caption(
        "Privacy note: this app does not save resumes to a database or write them to disk. "
        "Resume and job-description text is sent to Groq for the requested analysis. "
        "Avoid uploading information you do not want processed by the AI provider."
    )

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#718096;font-size:.85rem;">'
    'Built with Streamlit + Groq · Career guidance, powered by AI'
    '</div>',
    unsafe_allow_html=True,
)
