import requests
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from config import *



# Initialize LLM
# llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)

# Initialize Tavily Search Tool
tavily_tool = TavilySearchResults()




# ----------------------
# Agent 1: Research Agent
# ----------------------
def search_tavily(query, max_results=5):
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "max_results": max_results,
        "include_answer": False
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        # return response.json()
        return {
            "query": query,
            "gathered_info": response.json()["results"]
        }
    else:
        print("Error:", response.status_code, response.text)
        return {
            "query": None,
            "gathered_info": None
        }


# ----------------------
# Agent 2: Drafting Agent
# ----------------------
def drafting_agent():
    llm = ChatOpenAI()
    prompt = PromptTemplate.from_template(
        " Here is the question: {query} and You are an expert research summarizer. Write clear, factual, and detailed answers based only on provided information{gathered_info}.")

    # Create a chain
    chain = prompt | llm
    return chain

if __name__ == "__main__":
    chain = drafting_agent()
    query = input("Enter your question: ")
    dic = search_tavily(query=query, max_results=5)
    agent2 = drafting_agent()
    response = agent2.invoke({"query": query, "gathered_info": dic["gathered_info"]})
    print(response.content)
