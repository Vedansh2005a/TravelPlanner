import os
import uuid
import operator
from typing import TypedDict, Annotated
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver 
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from tools.tavilytool import tavily_search
from tools.flightool import search_flights

load_dotenv()

def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("Database url not found")

    if "sslmode" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int

def flight_agent(state: TravelState):
    querys = state['user_query']
    response = search_flights(querys)
    return {
        "flight_results": response, # FIXED: Changed from flight_result
        "messages": [AIMessage(content="flight data fetched")],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def tavily_agent(state: TravelState):
    query = f"Best Hotels for {state['user_query']}"
    response = tavily_search(query)
    return {
        "hotel_results": response,
        "messages": [AIMessage(content="Hotel data found")],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def itinary_agent(state: TravelState):
    prompt = f"""User Query:{state['user_query']},
    Flight Result:{state.get("flight_results", "")},
    Hotel Result:{state.get('hotel_results', "")}
    Make the itinary practical, budget aware and easy to follow
    """
    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner"),
        HumanMessage(content=prompt)
    ])
    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state.get('flight_results', '')}

Hotels:
{state.get('hotel_results', '')}

Itinerary:
{state.get('itinerary', '')}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
"""
    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

graph = StateGraph(TravelState)
graph.add_node("flight", flight_agent)
graph.add_node("hotel", tavily_agent)
graph.add_node("itinary", itinary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight")
graph.add_edge("flight", "hotel")
graph.add_edge("hotel", "itinary")
graph.add_edge("itinary", "final_agent")
graph.add_edge("final_agent", END)

# DATABASE_URL = get_database_url()
# _conn = psycopg.connect(
#     DATABASE_URL,
#     autocommit=True,
#     row_factory=dict_row
# )
# checkpointer = PostgresSaver(_conn)
# checkpointer.setup()
checkpointer=InMemorySaver()
travel_agent = graph.compile(checkpointer)

def run_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    configur = {"configurable": {"thread_id": thread_id}}

    result = travel_agent.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        }, 
        config=configur
    )
    
    final_answer = result["messages"][-1].content
    
    # FIXED: Standardized all output keys to match app.py exactly
    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""), 
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""), 
        "llm_calls": result.get("llm_calls", 0) 
    }
