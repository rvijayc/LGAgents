import IPython, pdb
from typing import Literal, TypedDict
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.tools import tool
from agentrun import AgentRun

# initialize agent runner.
python_runner = AgentRun(
        container_name="code_runner-python_runner-1",
        cached_dependencies = ["requests", "matplotlib"]
        )

# create a tool.
@tool("run_python_tool", parse_docstring=True)
def run_python_tool(python_code: str) -> str:
    """Runs the specified python code provided and returns the standard output.

    Args:
        python_code: Python code to run provided as a string.
    """
    return python_runner.execute_code_in_container(python_code)

# create a tool node corresponding to the above tool.
python_tool_node = ToolNode([run_python_tool], name="run_python")

# load database.
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")
print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')

# initialize chat model.
llm = init_chat_model("openai:gpt-4.1")

# get and analyze tools.
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}\n")

get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")


# Example: create a predetermined tool call
def list_tables(state: MessagesState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")

    return {"messages": [tool_call_message, tool_message, response]}


# Example: force a model to create a tool call
def call_get_schema(state: MessagesState):
    # Note that LangChain enforces that all models accept `tool_choice="any"`
    # as well as `tool_choice=<string name of tool>`.
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}

generate_query_system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You also have available, a python tool ("run_python_tool") that can execute
python code and return the standard output. You can use this for any
computation and transformation of the results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
""".format(
    dialect=db.dialect,
    top_k=5,
)


def generate_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
    # We do not force a tool call here, to allow the model to
    # respond naturally when it obtains the solution.
    llm_with_tools = llm.bind_tools([run_query_tool, run_python_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}

check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
""".format(dialect=db.dialect)


def check_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }

    # Generate an artificial user message to check
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}

def route_tool(state: MessagesState) -> Literal[END, "check_query", "run_python"]:
    messages = state["messages"]
    last_message = messages[-1]
    assert isinstance(last_message, AIMessage)
    tool_calls = last_message.tool_calls
    pdb.set_trace()
    if len(tool_calls) == 0:
        return END
    elif len(tool_calls) > 1:
        calls = [ t['name'] for t in tool_calls ]
        raise NotImplementedError(f'Not handling multiple tool calls {calls}')
    elif tool_calls[0]['name'] == 'sql_db_query':
        return 'check_query'
    elif tool_calls[0]['name'] == 'run_python_tool':
        return 'run_python'
    else:
        # should not be here.
        assert False

def should_get_table_meta(state: MessagesState, config: RunnableConfig) -> Literal["generate_query", "list_tables"]:
    if config['configurable']['first_time']: # type: ignore
        return 'list_tables'
    else:
        return 'generate_query'

class ConfigSchema(TypedDict):
    first_time: bool

builder = StateGraph(MessagesState, config_schema=ConfigSchema)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_query_node, "run_query")
builder.add_node(python_tool_node, "run_python")

builder.add_conditional_edges(START, should_get_table_meta)
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    route_tool,
)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")
builder.add_edge("run_python", "generate_query")

memory = MemorySaver()
agent = builder.compile(checkpointer=memory)

# print the graph.
agent.get_graph().print_ascii()

# initial runtime configuration.
config = { 
          "configurable": {
          "thread_id": "1",
          "first_time": True  # initialize first time to false.
          }
}

def ask(question:str="Which sales agent made the most in sales in 2009?"):
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
        config=config
    ):
        step["messages"][-1].pretty_print()

    # don't run steps needed only the first time.
    config['configurable']['first_time'] = False

IPython.embed()
