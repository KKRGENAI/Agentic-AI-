from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv
import os
load_dotenv()

# Model Initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# 1st Agent : Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a research search agent. "
            "You MUST call the web_search tool for every query. "
            "Do not answer from memory alone. "
            "In your final reply, list the best findings as bullet points and "
            "ALWAYS include the full Title and URL for every source you cite. "
            "Never drop or rewrite URLs."
        ),
    )

# 2nd Agent : Reader Agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a research reader agent. "
            "You MUST call the scrape_url tool on the most relevant http/https URLs "
            "provided in the user message (up to 3 URLs). "
            "Do not invent URLs. If no URL is present, say so clearly. "
            "After scraping, briefly summarize the extracted content and keep the source URLs."
        ),
    )


# writer chain

writer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert research writer. Write clear, structured and insightful reports. "
     "Only use facts present in the research. If a claim has no source URL, do not invent one. "
     "In Sources, list only real URLs that appear in the research."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# critic_chain

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
