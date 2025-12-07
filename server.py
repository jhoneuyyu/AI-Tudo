import asyncio
import os
import sys
import operator
from typing import Any, Dict, List, Annotated, TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model, Field
from fastapi.middleware.cors import CORSMiddleware
import sqlite3

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

# Ensure API Key is set
# The user sets "GEMINI_API_KEY" in Render, but LangChain expects "GOOGLE_API_KEY"
if "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    print(f"✅ Loaded API Key (Length: {len(os.environ['GEMINI_API_KEY'])})")
else:
    print("❌ ERROR: GEMINI_API_KEY is missing from Helper Environment!")

app = FastAPI(title="Tudo AI API")

# Allow CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the Next.js URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "running", "message": "Welcome to Tudo AI API"}


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def create_pydantic_model_from_schema(name: str, schema: Dict[str, Any]):
    """Dynamically create a Pydantic model from a JSON schema."""
    fields = {}

    if "properties" in schema:
        for field_name, field_info in schema["properties"].items():
            typ = str
            if field_info.get("type") == "integer":
                typ = int
            elif field_info.get("type") == "boolean":
                typ = bool

            required = field_name in schema.get("required", [])
            default = ... if required else field_info.get("default", None)
            fields[field_name] = (typ, Field(default=default, description=field_info.get("description", "")))

    return create_model(f"{name}Input", **fields)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# ------------------------------------------------------------
# LOGIC
# ------------------------------------------------------------

async def query_agent(message: str, history: List[Dict[str, str]]):
    """
    Core function to query the MCP agent.
    """
    if not message:
        return "Please type something."

    # Construct conversation history
    messages = [
        SystemMessage(content="You are Tudo, an intelligent task and event manager. You have access to tools to manage a todo list. Be helpful, concise, and friendly.")
    ]
    
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
            
    messages.append(HumanMessage(content=message))

    # Parameters to start the MCP server
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["main.py"],
        env=os.environ.copy()
    )

    try:
        # Connect to MCP
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Discover Tools
                mcp_tools = await session.list_tools()
                lc_tools = []

                for tool in mcp_tools.tools:
                    schema = tool.inputSchema
                    args_model = create_pydantic_model_from_schema(tool.name, schema)

                    # Fix closure capture
                    tool_name = tool.name
                    async def wrapper(name=tool_name, **kwargs):
                        return await session.call_tool(name, arguments=kwargs)

                    lc_tool = StructuredTool.from_function(
                        func=None,
                        coroutine=wrapper,
                        name=tool.name,
                        description=tool.description or f"Tool {tool.name}",
                        args_schema=args_model
                    )
                    lc_tools.append(lc_tool)

                # Bind LLM
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
                llm = llm.bind_tools(lc_tools)

                async def call_model(state: AgentState):
                    resp = await llm.ainvoke(state["messages"])
                    return {"messages": [resp]}

                # Build Graph
                graph = StateGraph(AgentState)
                graph.add_node("agent", call_model)
                graph.add_node("tools", ToolNode(lc_tools))

                graph.set_entry_point("agent")
                graph.add_conditional_edges("agent", tools_condition)
                graph.add_edge("tools", "agent")

                workflow = graph.compile()

                # Run Workflow
                result = await workflow.ainvoke({"messages": messages})
                final_response = result["messages"][-1].content
                return final_response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"⚠️ Error: {str(e)}"

# ------------------------------------------------------------
# API MODELS & ENDPOINTS
# ------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = await query_agent(request.message, request.history)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/events")
async def get_events():
    """But exposes the event log to the frontend."""
    log_file = "event_log.log"
    if not os.path.exists(log_file):
        return {"events": []}
    
    events = []
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "EVENT FIRED:" in line:
                    # Extract the part after EVENT FIRED:
                    try:
                        timestamp = line.split(" - ")[0]
                        content = line.split("EVENT FIRED:")[1].strip()
                        # Parse key-value pairs roughly
                        data = {}
                        for part in content.split(" "):
                            if "=" in part:
                                k, v = part.split("=", 1)
                                data[k] = v
                        
                        data['timestamp'] = timestamp
                        events.append(data)
                    except:
                        continue
    except Exception as e:
        print(f"Error reading log: {e}")
        
    # Return last 50 events reversed
    return {"events": events[::-1][:50]}

@app.get("/tasks")
async def get_tasks():
    """Fetch all tasks directly from the DB for the UI."""
    db_file = "tudo.sqlite3"
    if not os.path.exists(db_file):
        return {"tasks": []}
    
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row  # To get dictionary-like access
        c = conn.cursor()
        c.execute("SELECT * FROM todos ORDER BY priority DESC, created_at DESC")
        rows = c.fetchall()
        
        tasks = []
        for row in rows:
            tasks.append(dict(row))
            
        conn.close()
        return {"tasks": tasks}
    except Exception as e:
        print(f"Error reading tasks: {e}")
        return {"tasks": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Tudo API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
