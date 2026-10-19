"""
Study Assistant Agent - Main Streamlit Web Application
An L2 AI-Powered Privacy-Focused Learning Assistant with LLM Tool Loop, Real MCP Client/Server, Reflection, and Prompt Injection Defense.
"""

import os
import shutil
import tempfile
import logging
from datetime import datetime, date
import streamlit as st
import pandas as pd

# Internal Module Imports
from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.embeddings import LocalEmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import ContextRetriever
from src.llm import LocalLLMHandler
from src.rag_agent import RAGAgent
from src.study_planner import StudyPlannerAgent
from src.summarizer import SummarizerAgent
from src.quiz_generator import QuizGeneratorAgent
from src.progress_tracker import ProgressTrackerAgent
from src.agent_router import AgentRouter
from src.evaluation import RAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Setup
st.set_page_config(
    page_title="Study Assistant Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling (Glassmorphism + Sunset Rose & Amber Warm Palette)
st.markdown("""
<style>
    /* Global App Background & Typography */
    .stApp {
        background: linear-gradient(135deg, #18080e 0%, #2a081b 45%, #1f0e06 85%, #120508 100%);
        color: #fff1f2;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1c0910 0%, #15060b 100%) !important;
        border-right: 1px solid rgba(251, 146, 60, 0.15);
    }
    
    /* Card Container with Sunset Glow */
    .glass-card {
        background: rgba(45, 16, 26, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(251, 146, 60, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(243, 85, 52, 0.25);
    }
    
    /* Header Sunset Gradient */
    .main-header {
        background: linear-gradient(90deg, #f43f5e 0%, #fb923c 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    
    /* Badge styling */
    .badge-source {
        background: linear-gradient(90deg, #881337 0%, #9a3412 100%);
        color: #ffedd5;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        border: 1px solid rgba(251, 146, 60, 0.3);
    }
    
    /* Stat Box with Rose-Amber Accent */
    .stat-box {
        background: rgba(251, 146, 60, 0.07);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        border: 1px solid rgba(244, 63, 94, 0.25);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stat-box:hover {
        background: rgba(251, 146, 60, 0.12);
        border-color: rgba(251, 146, 60, 0.45);
    }
    .stat-val {
        font-size: 1.9rem;
        font-weight: 800;
        color: #fb923c;
        text-shadow: 0 0 20px rgba(251, 146, 60, 0.3);
    }
    .stat-lbl {
        font-size: 0.82rem;
        color: #fca5a5;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 18px;
        background-color: rgba(45, 16, 26, 0.5);
        color: #fecdd3;
        border: 1px solid rgba(251, 146, 60, 0.15);
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(244, 63, 94, 0.2);
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #e11d48 0%, #f97316 100%) !important;
        color: #ffffff !important;
        font-weight: 700;
        border: none !important;
        box-shadow: 0 4px 18px rgba(225, 29, 72, 0.4);
    }

    /* Streamlit Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #e11d48 0%, #f97316 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(225, 29, 72, 0.3) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(249, 115, 22, 0.5) !important;
        background: linear-gradient(90deg, #f43f5e 0%, #fb923c 100%) !important;
    }

    /* Inputs & Selectboxes Focus Glow */
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
        border-radius: 8px;
        border: 1px solid rgba(251, 146, 60, 0.25) !important;
        background-color: rgba(24, 8, 14, 0.7) !important;
        color: #fff1f2 !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #fb923c !important;
        box-shadow: 0 0 12px rgba(251, 146, 60, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State Managers (Singletons)
@st.cache_resource
def get_system_components():
    embedding_mgr = LocalEmbeddingManager(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStoreManager(persist_directory="vectorstore")
    retriever = ContextRetriever(vector_store=vector_store, embedding_manager=embedding_mgr)
    llm_handler = LocalLLMHandler(model_name="llama3.2:3b")
    
    rag_agent = RAGAgent(retriever=retriever, llm_handler=llm_handler)
    study_planner = StudyPlannerAgent(llm_handler=llm_handler)
    summarizer = SummarizerAgent(retriever=retriever, llm_handler=llm_handler)
    quiz_generator = QuizGeneratorAgent(retriever=retriever, llm_handler=llm_handler)
    progress_tracker = ProgressTrackerAgent()
    
    agent_router = AgentRouter(
        rag_agent=rag_agent,
        study_planner=study_planner,
        summarizer=summarizer,
        quiz_generator=quiz_generator,
        progress_tracker=progress_tracker
    )
    
    return {
        "embedding_mgr": embedding_mgr,
        "vector_store": vector_store,
        "retriever": retriever,
        "llm_handler": llm_handler,
        "rag_agent": rag_agent,
        "study_planner": study_planner,
        "summarizer": summarizer,
        "quiz_generator": quiz_generator,
        "progress_tracker": progress_tracker,
        "agent_router": agent_router
    }


components = get_system_components()
v_store = components["vector_store"]
emb_mgr = components["embedding_mgr"]
llm_h = components["llm_handler"]
p_tracker = components["progress_tracker"]


# Initialize User Profile in Session State
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "goal": "Score 90%+ in upcoming exams",
        "exam_date": date.today().isoformat(),
        "subjects": ["Computer Science", "Mathematics"],
        "topics": ["Algorithms", "Data Structures", "Probability"],
        "daily_hours": 3.0,
        "preferred_duration": 60
    }

if "generated_plan" not in st.session_state:
    st.session_state.generated_plan = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Helper function to get valid selection options
def get_safe_options(opt_list, default_name="General"):
    cleaned = [str(x).strip() for x in opt_list if str(x).strip()]
    return cleaned if cleaned else [default_name]


# ------------------- SIDEBAR: PROFILE & SYSTEM STATUS -------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/brain.png", width=70)
    st.markdown("## 🎓 Student Profile")
    
    with st.expander("⚙️ Edit Profile Setup", expanded=True):
        profile_goal = st.text_input("Study Goal", value=st.session_state.user_profile["goal"])
        profile_exam_date = st.date_input("Exam Date", value=datetime.strptime(st.session_state.user_profile["exam_date"], "%Y-%m-%d").date())
        profile_subjects = st.text_area("Subjects (comma separated)", value=", ".join(st.session_state.user_profile["subjects"]))
        profile_topics = st.text_area("Topics (comma separated)", value=", ".join(st.session_state.user_profile["topics"]))
        profile_hours = st.number_input("Daily Study Hours", min_value=0.5, max_value=16.0, value=float(st.session_state.user_profile["daily_hours"]), step=0.5)

        if st.button("💾 Save Profile", use_container_width=True):
            subj_list = [s.strip() for s in profile_subjects.split(",") if s.strip()]
            top_list = [t.strip() for t in profile_topics.split(",") if t.strip()]
            
            st.session_state.user_profile.update({
                "goal": profile_goal,
                "exam_date": profile_exam_date.isoformat(),
                "subjects": subj_list if subj_list else ["Computer Science"],
                "topics": top_list if top_list else ["General Study"],
                "daily_hours": profile_hours
            })
            p_tracker.set_topics(top_list)
            st.success("Profile saved successfully!")

    st.markdown("---")
    st.markdown("### 🤖 Agent Architecture Status")
    st.info("🟢 **LLM Tool Loop**: Bounded (Max 5 Steps)\n\n🟢 **MCP Client/Server**: Active\n\n🟢 **Reflection Stage**: Enabled")

    ollama_info = llm_h.check_ollama_availability()
    if ollama_info["available"]:
        st.success(f"🟢 Ollama Active ({ollama_info['active_model']})")
    else:
        st.warning("🟡 Offline Fallback Mode (Ollama not connected)")

    st.markdown("### 📚 Vector Store Stats")
    total_docs = len(v_store.list_indexed_documents())
    total_chunks = v_store.get_total_chunks_count()
    st.info(f"📄 **{total_docs} Documents** ({total_chunks} Chunks indexed)")


# ------------------- MAIN INTERFACE -------------------
st.markdown("<h1 class='main-header'>🎓 Study Assistant Agent</h1>", unsafe_allow_html=True)
st.caption("Agentic AI Study Assistant • LLM Bounded Tool Loop • Real MCP Integration • Reflection & Citation Verification")

# Create Navigation Tabs
tab_dash, tab_planner, tab_docs, tab_qa, tab_summary, tab_quiz, tab_progress, tab_eval = st.tabs([
    "📊 Dashboard",
    "📅 Study Planner",
    "📚 Study Materials",
    "💬 Ask Assistant",
    "📝 Summarizer",
    "🧩 Practice Quiz",
    "📈 Progress",
    "⚙️ Evaluation & MCP"
])


# =========================================================================
# TAB 1: DASHBOARD
# =========================================================================
with tab_dash:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📌 Study Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    days_left = StudyPlannerAgent.calculate_days_remaining(st.session_state.user_profile["exam_date"])
    prog_summary = p_tracker.get_progress_summary()
    
    with col1:
        st.markdown(f"<div class='stat-box'><div class='stat-val'>{days_left} Days</div><div class='stat-lbl'>Until Exam</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-box'><div class='stat-val'>{prog_summary['completion_percentage']}%</div><div class='stat-lbl'>Topics Mastered</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='stat-box'><div class='stat-val'>{prog_summary['logged_hours']} hrs</div><div class='stat-lbl'>Study Time Logged</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='stat-box'><div class='stat-val'>{total_docs}</div><div class='stat-lbl'>Documents Loaded</div></div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### 🎯 Current Study Target")
        st.info(f"**Goal:** {st.session_state.user_profile['goal']}\n\n**Exam Date:** {st.session_state.user_profile['exam_date']} • **Daily Capacity:** {st.session_state.user_profile['daily_hours']} hours/day")
        
        st.markdown("### 🗓️ Today's Study Tasks")
        sched = st.session_state.generated_plan.get("schedule", []) if st.session_state.generated_plan else []
        if sched:
            today_schedule = sched[0]
            for idx, task in enumerate(today_schedule.get("tasks", [])):
                col_chk, col_txt = st.columns([1, 10])
                completed = col_chk.checkbox("", value=task.get("completed", False), key=f"dash_task_{idx}")
                if completed:
                    task["completed"] = True
                    p_tracker.mark_topic_completed(task["topic"])
                    col_txt.markdown(f"~~**[{task['subject']}]** {task['topic']} ({task['duration']})~~")
                else:
                    task["completed"] = False
                    p_tracker.mark_topic_pending(task["topic"])
                    col_txt.markdown(f"**[{task['subject']}]** {task['topic']} — `{task['duration']}`")
        else:
            st.write("No study plan generated yet. Go to the **Study Planner** tab to generate your schedule!")

    with col_right:
        st.markdown("### 📚 Quick Document Status")
        docs = v_store.list_indexed_documents()
        if docs:
            df_docs = pd.DataFrame(docs)
            st.dataframe(df_docs[["filename", "file_type", "chunk_count"]], use_container_width=True, hide_index=True)
        else:
            st.warning("No study materials uploaded yet. Upload documents in the 'Study Materials' tab.")


# =========================================================================
# TAB 2: STUDY PLANNER
# =========================================================================
with tab_planner:
    st.header("📅 Personalized Study Schedule Generator")
    st.write("Generate a custom study plan based on your goal, exam countdown, and daily availability.")

    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        st.markdown("#### Plan Configuration")
        plan_goal = st.text_input("Goal / Target", value=st.session_state.user_profile["goal"])
        plan_exam_date = st.date_input("Target Exam Date", value=datetime.strptime(st.session_state.user_profile["exam_date"], "%Y-%m-%d").date(), key="planner_date")
        plan_subjects_raw = st.text_area("Subjects", value=", ".join(st.session_state.user_profile["subjects"]), key="planner_subjs")
        plan_topics_raw = st.text_area("Topics to Cover", value=", ".join(st.session_state.user_profile["topics"]), key="planner_topics")
        plan_hours = st.slider("Daily Hours", 0.5, 12.0, float(st.session_state.user_profile["daily_hours"]), 0.5)

        if st.button("🚀 Generate Personalized Study Plan", type="primary", use_container_width=True):
            subjs = [s.strip() for s in plan_subjects_raw.split(",") if s.strip()]
            tops = [t.strip() for t in plan_topics_raw.split(",") if t.strip()]
            
            with st.spinner("Calculating optimal study schedule via MCP Planner Tool..."):
                res = components["agent_router"].mcp_client.execute_tool("create_study_plan", {
                    "goal": plan_goal,
                    "exam_date": str(plan_exam_date),
                    "subjects": subjs if subjs else ["General Subject"],
                    "topics": tops if tops else ["General Topic"],
                    "daily_hours": plan_hours
                })
                plan_res = res.result_data if isinstance(res.result_data, dict) else {}
                if plan_res.get("success"):
                    st.session_state.generated_plan = plan_res
                    p_tracker.set_topics(tops)
                    st.success(f"Plan generated! Allocated across {plan_res['total_days_planned']} days.")
                else:
                    st.error(plan_res.get("error", "Failed to generate plan."))

    with col_p2:
        if st.session_state.generated_plan:
            plan_data = st.session_state.generated_plan
            st.markdown(f"### 📋 Study Plan Schedule ({plan_data.get('days_remaining', 0)} Days to Exam)")
            
            with st.expander("🤖 AI Counselor Strategy Overview", expanded=True):
                st.markdown(plan_data.get("ai_summary", ""))

            for day_item in plan_data.get("schedule", []):
                st.markdown(f"#### 📅 {day_item['day']}")
                for task_idx, task in enumerate(day_item.get("tasks", [])):
                    st.markdown(f"- **{task['subject']}**: {task['topic']} (`{task['duration']}`)")
                st.markdown("---")
        else:
            st.info("Configure your study parameters on the left and click **Generate Personalized Study Plan**.")


# =========================================================================
# TAB 3: STUDY MATERIALS
# =========================================================================
with tab_docs:
    st.header("📚 Upload & Process Study Materials")
    st.write("Upload PDF, TXT, or DOCX study notes to build your local RAG knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose study files (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("⚡ Process & Index Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_added_chunks = 0
            
            for idx, uploaded_file in enumerate(uploaded_files, start=1):
                status_text.text(f"Processing '{uploaded_file.name}'...")
                
                save_dir = os.path.abspath("data/uploads")
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    doc_data = DocumentLoader.load_document(file_path)
                    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_text(doc_data["text"], source_metadata=doc_data["metadata"])
                    
                    texts = [c["text"] for c in chunks]
                    embeddings = emb_mgr.embed_documents(texts)
                    
                    added = v_store.add_chunks(chunks, embeddings)
                    total_added_chunks += added
                except Exception as e:
                    st.error(f"Error processing '{uploaded_file.name}': {str(e)}")

                progress_bar.progress(idx / len(uploaded_files))

            status_text.text("Indexing completed!")
            st.success(f"Successfully processed {len(uploaded_files)} document(s) into {total_added_chunks} vector chunks!")

    st.markdown("---")
    st.subheader("📄 Indexed Documents Registry")
    
    docs = v_store.list_indexed_documents()
    if docs:
        col_list, col_actions = st.columns([3, 1])
        with col_list:
            st.dataframe(pd.DataFrame(docs), use_container_width=True, hide_index=True)
        with col_actions:
            doc_to_delete = st.selectbox("Select file to remove", [d["filename"] for d in docs])
            if st.button("🗑️ Delete File"):
                deleted = v_store.delete_document_by_filename(doc_to_delete)
                st.success(f"Removed {deleted} chunks for '{doc_to_delete}'.")
                st.rerun()

            if st.button("🚨 Clear All Index Data"):
                v_store.clear_all()
                st.warning("Vector store index cleared!")
                st.rerun()
    else:
        st.info("No documents currently stored in vector store.")


# =========================================================================
# TAB 4: ASK ASSISTANT (LLM TOOL LOOP + MCP + REFLECTION)
# =========================================================================
with tab_qa:
    st.header("💬 Ask Questions & Interact with Study Agent")
    st.caption("LLM-Driven Bounded Tool Loop (Max 5 steps) • MCP Client/Server Execution • Reflection Verification")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg.get("security_warning"):
                st.warning(f"⚠️ {msg['security_warning']}")

            if msg.get("sources"):
                with st.expander("📌 View Source Citations"):
                    for src in msg["sources"]:
                        st.markdown(f"**[{src.get('filename', 'Doc')} - Chunk #{src.get('chunk_id', '?')} (Relevance: {int(src.get('similarity_score', 0)*100)}%)]**")
                        st.caption(f"\"{src.get('text_snippet', '')}\"")

            if msg.get("loop_detail"):
                with st.expander("🛠️ View Agent Tool Loop & Reflection Trace"):
                    ld = msg["loop_detail"]
                    st.markdown(f"**Steps Taken:** {ld.get('steps_taken', 1)} / 5")
                    
                    st.markdown("#### 🔍 Discovered Observations:")
                    for obs in ld.get("observations", []):
                        st.code(obs.get("formatted_text", str(obs)))

                    st.markdown("#### 🔍 Reflection Result:")
                    ref = ld.get("reflection", {})
                    st.json({
                        "supported": ref.get("supported"),
                        "issues": ref.get("issues"),
                        "required_action": ref.get("required_action")
                    })

    user_query = st.chat_input("Ask a question, request a plan, summary, or quiz...")
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Agent Loop: Deciding tool calls via MCP Client, executing tools & reflecting..."):
                router_res = components["agent_router"].route_and_execute(
                    user_input=user_query,
                    context_params=st.session_state.user_profile
                )
                
                result_info = router_res["result"]
                answer_text = result_info["answer"]
                sources = result_info.get("sources", [])
                sec_warn = result_info.get("security_warning")
                loop_detail = router_res.get("loop_detail", {})

                if sec_warn:
                    st.warning(f"⚠️ {sec_warn}")

                st.markdown(answer_text)
                
                if sources:
                    with st.expander("📌 View Source Citations"):
                        for src in sources:
                            st.markdown(f"**[{src.get('filename', 'Doc')} - Chunk #{src.get('chunk_id', '?')} (Relevance: {int(src.get('similarity_score', 0)*100)}%)]**")
                            st.caption(f"\"{src.get('text_snippet', '')}\"")

                with st.expander("🛠️ View Agent Tool Loop & Reflection Trace"):
                    st.markdown(f"**Steps Taken:** {loop_detail.get('steps_taken', 1)} / 5")
                    
                    st.markdown("#### 🔍 Discovered Observations:")
                    for obs in loop_detail.get("observations", []):
                        st.code(obs.get("formatted_text", str(obs)))

                    st.markdown("#### 🔍 Reflection Verification:")
                    ref = loop_detail.get("reflection", {})
                    st.json({
                        "supported": ref.get("supported"),
                        "issues": ref.get("issues"),
                        "required_action": ref.get("required_action")
                    })

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                    "security_warning": sec_warn,
                    "loop_detail": loop_detail
                })


# =========================================================================
# TAB 5: SUMMARIZER
# =========================================================================
with tab_summary:
    st.header("📝 Generate Summaries & Revision Notes")
    
    topics = get_safe_options(st.session_state.user_profile["topics"], "General Topics")

    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        summary_mode = st.radio("Summarize By:", ["Topic Query", "Raw Text Input"])
        
        target_topic = ""
        raw_text_input = ""
        
        if summary_mode == "Topic Query":
            target_topic = st.selectbox("Select Topic", topics)
        else:
            raw_text_input = st.text_area("Paste text to summarize", height=200)

        if st.button("✨ Generate Summary", type="primary", use_container_width=True):
            with st.spinner("Synthesizing study summary via MCP..."):
                obs = components["agent_router"].mcp_client.execute_tool("summarize_material", {
                    "text": raw_text_input if summary_mode == "Raw Text Input" else None,
                    "topic": target_topic if summary_mode == "Topic Query" else None
                })
                res = obs.result_data if isinstance(obs.result_data, dict) else {}
                if res.get("success"):
                    st.session_state.current_summary = res["summary"]
                else:
                    st.error(res.get("error", "Failed to generate summary."))

    with col_s2:
        if "current_summary" in st.session_state:
            st.markdown("### 📋 Executive Summary & Key Notes")
            st.markdown(st.session_state.current_summary)
        else:
            st.info("Select a topic or paste text to generate a summary.")


# =========================================================================
# TAB 6: PRACTICE QUIZ
# =========================================================================
with tab_quiz:
    st.header("🧩 Practice Quiz Generator")
    st.write("Generate grounded MCQs, Short-Answer, and Conceptual questions from study materials.")

    topics_for_quiz = get_safe_options(st.session_state.user_profile["topics"], "General Topics")

    col_q1, col_q2 = st.columns([1, 2])
    
    with col_q1:
        quiz_topic = st.selectbox("Select Topic for Quiz", topics_for_quiz, key="quiz_top")
        num_q = st.slider("Number of Questions", 2, 10, 5)

        if st.button("🎲 Generate Practice Quiz", type="primary", use_container_width=True):
            with st.spinner("Generating grounded practice questions via MCP..."):
                obs = components["agent_router"].mcp_client.execute_tool("generate_quiz", {
                    "topic": quiz_topic,
                    "num_questions": num_q
                })
                quiz_res = obs.result_data if isinstance(obs.result_data, dict) else {}
                if quiz_res.get("success"):
                    st.session_state.active_quiz = quiz_res
                else:
                    st.error(quiz_res.get("error", "Failed to generate quiz."))

    with col_q2:
        if "active_quiz" in st.session_state:
            quiz = st.session_state.active_quiz
            st.markdown(f"### 📝 Practice Quiz: {quiz['topic']}")
            
            for q in quiz.get("questions", []):
                st.markdown(f"**Q{q['id']}. [{q['type']}] {q['question']}**")
                
                if q['options']:
                    user_ans = st.radio("Select Answer:", q['options'], key=f"q_opt_{q['id']}")
                
                with st.expander("👁️ Reveal Answer & Explanation"):
                    st.markdown(f"**Correct Answer:** {q['answer']}")
                    if q['explanation']:
                        st.caption(f"Explanation: {q['explanation']}")
                st.markdown("---")
        else:
            st.info("Choose a topic and click **Generate Practice Quiz**.")


# =========================================================================
# TAB 7: PROGRESS TRACKER
# =========================================================================
with tab_progress:
    st.header("📈 Learning Progress & Activity Dashboard")
    
    subjs_for_log = get_safe_options(st.session_state.user_profile["subjects"], "General Subject")
    topics_for_log = get_safe_options(st.session_state.user_profile["topics"], "General Topic")

    col_log, col_overview = st.columns([1, 2])
    
    with col_log:
        st.markdown("### ⏱️ Log Study Session")
        log_subj = st.selectbox("Subject", subjs_for_log, key="log_s")
        log_top = st.selectbox("Topic", topics_for_log, key="log_t")
        log_hrs = st.number_input("Time Spent (Hours)", min_value=0.25, max_value=8.0, value=1.0, step=0.25)
        log_notes = st.text_input("Session Notes")

        if st.button("💾 Log Session", type="primary", use_container_width=True):
            components["agent_router"].mcp_client.execute_tool("update_progress", {
                "action": "log_session",
                "subject": log_subj,
                "topic": log_top,
                "hours": log_hrs,
                "notes": log_notes
            })
            st.success(f"Logged {log_hrs} hours for {log_top}!")

    with col_overview:
        prog = p_tracker.get_progress_summary()
        st.markdown("### 📊 Overall Progress Metrics")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Completion Rate", f"{prog['completion_percentage']}%")
        col_m2.metric("Hours Logged", f"{prog['logged_hours']} hrs")
        col_m3.metric("Completed Topics", f"{prog['completed_count']}/{prog['total_topics']}")

        st.progress(prog['completion_percentage'] / 100.0)

        st.markdown("#### Topic Status Breakdown")
        for top in st.session_state.user_profile["topics"]:
            is_done = top in prog["completed_topics"]
            chk = st.checkbox(top, value=is_done, key=f"prog_chk_{top}")
            if chk:
                p_tracker.mark_topic_completed(top)
            else:
                p_tracker.mark_topic_pending(top)


# =========================================================================
# TAB 8: EVALUATION & MCP DISCOVERY
# =========================================================================
with tab_eval:
    st.header("⚙️ RAG System Evaluation & Real MCP Client/Server Discovery")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.markdown("### 🧪 RAG Metric Evaluator")
        test_query = st.text_input("Test Query", value="What are the key concepts?")
        
        if st.button("📊 Run RAG Metric Evaluation"):
            try:
                context_text, sources, found = components["retriever"].retrieve_context(test_query)
                rag_answer_dict = components["rag_agent"].answer_question(test_query)
                rag_answer = rag_answer_dict.get("answer", "")
                
                eval_results = RAGEvaluator.run_full_evaluation(
                    query=test_query,
                    answer=rag_answer,
                    context_text=context_text,
                    sources=sources
                )
                
                st.success("RAG Metric Evaluation Completed!")
                st.json(eval_results)
            except Exception as e:
                st.error(f"Evaluation Error: {str(e)}")

    with col_e2:
        st.markdown("### 🛠️ Model Context Protocol (MCP) Server Tools")
        st.write("Discovered tools retrieved dynamically from MCP Server via MCP Client:")
        
        mcp_client = components["agent_router"].mcp_client
        discovered_mcp_tools = mcp_client.get_discovered_tools()
        for tool in discovered_mcp_tools:
            st.markdown(f"**`{tool['name']}`**")
            st.caption(tool['description'])
            with st.expander("Show MCP Input Schema"):
                st.json(tool["inputSchema"])
            st.markdown("---")
