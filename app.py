import os
import streamlit as st
import requests
import numpy as np
import faiss
from jinja2 import Template
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# Groq API Key
groq_api_key = "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"
llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0.5)

# Serper API Key for Web Search
SERPER_API_KEY = "a96c84d75621f76c07539b58309ca8026840e293"

# FAISS Vector DB Setup
embedding_dim = 768  # Adjust based on your model
index = faiss.IndexFlatL2(embedding_dim)
faiss_metadata = {}  # Stores metadata for each entry


def google_search(query):
    """Perform Google search using Serper API"""
    url = "https://google.serper.dev/search"
    payload = {"q": query, "gl": "us", "hl": "en"}
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    return response.json()


def web_scraper(url):
    """Simple web content scraper"""
    try:
        response = requests.get(url, timeout=10)
        return response.text[:2000]  # Reduce token size
    except:
        return ""


def generate_embedding(text):
    """Generates a numerical embedding using Groq API"""
    embedding_response = llm.invoke([HumanMessage(content=f"Generate embedding for: {text}")]).content

    # Ensure proper embedding conversion
    embedding = [float(x) for x in embedding_response.split() if x.replace('.', '', 1).isdigit()]
    
    if len(embedding) != embedding_dim:
        raise ValueError(f"Invalid embedding size: Expected {embedding_dim}, got {len(embedding)}")
    
    return np.array(embedding, dtype=np.float32)


def store_in_vector_db(content, metadata):
    """Convert text content into embeddings and store in FAISS."""
    try:
        embedding = generate_embedding(content)
        index.add(np.array([embedding]))
        faiss_metadata[len(index) - 1] = metadata
    except ValueError as e:
        print(f"Embedding error: {e}")


def research_agent(topic):
    """Multi-agent research system using Groq AI and FAISS for knowledge retrieval"""
    planner_prompt = f"""Break down this research topic into key sub-questions: {topic}
    Return as a numbered list of questions."""
    questions = llm.invoke([HumanMessage(content=planner_prompt)]).content.split("\n")

    st.session_state.research_data = {"topic": topic, "questions": []}

    for q in questions[:3]:  # Reduce questions to stay within token limits
        if not q.strip():
            continue
        search_results = google_search(q)
        sources = []

        for result in search_results.get("organic", [])[:2]:
            content = web_scraper(result["link"])
            if content:
                sources.append({"title": result.get("title", ""), "link": result["link"], "content": content})
                store_in_vector_db(content, {"title": result.get("title", ""), "link": result["link"]})

        st.session_state.research_data["questions"].append({"question": q, "sources": sources})


def generate_report():
    """Generate structured research report using Groq AI"""
    report_template = """
    # Research Report: {{ topic }}

    ## Overview
    {% for section in sections %}
    ### {{ section.title }}
    {{ section.content }}

    Sources:
    {% for source in section.sources %}
    - [{{ source.title }}]({{ source.link }})
    {% endfor %}
    {% endfor %}
    """

    sections = []
    for item in st.session_state.research_data["questions"]:
        prompt = f"""Compile research findings for: {item['question']}
        Context: {item['sources']}
        Create a comprehensive section with references."""
        section_content = llm.invoke([HumanMessage(content=prompt)]).content

        sections.append({"title": item["question"], "content": section_content, "sources": item["sources"]})

    return Template(report_template).render(topic=st.session_state.research_data["topic"], sections=sections)


def create_pdf(content):
    """Generate PDF"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in content.splitlines():
        pdf.multi_cell(0, 10, line)

    if os.path.exists("research_report.pdf"):
        os.remove("research_report.pdf")

    pdf.output("research_report.pdf")
    st.success("PDF saved as 'research_report.pdf'.")


def main():
    st.title("🤖 AI Research Agent")
    st.markdown("Enter a topic to generate a comprehensive research report.")
    topic = st.text_input("Research Topic:", placeholder="Climate change impacts on biodiversity...")

    if st.button("Start Research"):
        if topic:
            with st.spinner("🔍 Conducting research..."):
                research_agent(topic)
                report_content = generate_report()
            st.success("✅ Research complete!")
            create_pdf(report_content)
            with open("research_report.pdf", "rb") as pdf_file:
                st.download_button(label="📄 Download PDF", data=pdf_file, file_name="research_report.pdf", mime="application/pdf")
            st.markdown("### Research Preview")
            st.markdown(report_content, unsafe_allow_html=True)
        else:
            st.warning("Please enter a research topic")


if __name__ == "__main__":
    main()
