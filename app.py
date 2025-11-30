import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from tools import FinancialsTool, PDFReadTool, WebSearchTool

# 1. Page Configuration (The Tab Title and Icon)
st.set_page_config(
    page_title="AI Equity Research Analyst",
    page_icon="📈",
    layout="wide"
)

# 2. Load API Keys
load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    st.error("Please set your GROQ_API_KEY in the .env file")
    st.stop()

# 3. Sidebar (User Inputs)
with st.sidebar:
    st.title("🚀 Research Parameters")
    st.markdown("---")
    
    company = st.text_input("Company Name", value="Reliance Industries")
    ticker = st.text_input("Ticker Symbol", value="RELIANCE.NS")
    pdf_name = st.text_input("Annual Report Filename", value="reliance.pdf")
    
    st.markdown("---")
    st.info("Make sure the PDF file is located in the `annual_reports` folder.")
    
    run_btn = st.button("Generate Report", type="primary")

# 4. Main UI Layout
st.title("📈 Institutional Equity Research Agent")
st.markdown("### Powered by Multi-Agent AI (Groq + CrewAI)")

# 5. The Logic (Hidden Function)
def run_analysis(company, ticker, pdf_name):
    # --- SETUP AGENTS (Copy of your profile_agent.py logic) ---
    my_llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY")
    )

    profiler = Agent(
        role='Corporate Historian & Strategist',
        goal='Construct a deep-dive "Company Profile" (Section 1).',
        backstory="Forensic analyst looking for revenue splits and history.",
        verbose=True,
        memory=False,
        llm=my_llm,
        tools=[FinancialsTool(), PDFReadTool(), WebSearchTool()]
    )

    profile_task = Task(
        description=f"""
        Analyze {company} ({ticker}) using PDF: {pdf_name}.
        1. Basics: Industry, Business Desc.
        2. PDF Deep Dive: Segment Revenue, Exports.
        3. History (Web): Acquisitions, Failures.
        4. News (Web): Top news last 1 year.
        """,
        expected_output="Markdown report with Section 1, Top News, Highs & Lows.",
        agent=profiler
    )

    crew = Crew(
        agents=[profiler],
        tasks=[profile_task],
        process=Process.sequential
    )
    
    return crew.kickoff(inputs={'company': company, 'ticker': ticker, 'pdf_name': pdf_name})

# 6. Execution Loop
if run_btn:
    if not company or not ticker or not pdf_name:
        st.warning("Please fill in all fields.")
    else:
        # Create a container for the output
        result_container = st.container()
        
        with st.status("🤖 AI Agents are working...", expanded=True) as status:
            st.write("🔍 Searching Web for History...")
            st.write("📄 Reading Annual Report...")
            st.write("📊 Analyzing Financial Segments...")
            
            # RUN THE AGENT
            try:
                final_report = run_analysis(company, ticker, pdf_name)
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                
                # Display Result
                with result_container:
                    st.markdown("---")
                    st.markdown(str(final_report))
                    
                    # Download Button
                    st.download_button(
                        label="📥 Download Report as Markdown",
                        data=str(final_report),
                        file_name=f"{company}_Analysis.md",
                        mime="text/markdown"
                    )
            except Exception as e:
                st.error(f"An error occurred: {e}")