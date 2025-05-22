import os
import tempfile
from typing import Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.runnables.graph_png import PngDrawer

from python_toolkit import PythonRunnerTool
from sql_toolkit import SQLDatabaseTool, list_tables, call_get_schema, check_query
from utils import show_image

react_system_prompt = """
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
"""
def react_node(state: MessagesState, config: RunnableConfig):

    # generate a system message.
    cfg = config['configurable'] # pyright: ignore[reportTypedDictNotRequiredAccess]
    sql_toolkit = cfg['sql_toolkit']
    system_message = SystemMessage(react_system_prompt.format(
        dialect=sql_toolkit.dialect,
        top_k=5
    ))
    python_toolkit = cfg['python_runner']

    # We do not force a tool call here, to allow the model to
    # respond naturally when it obtains the solution.
    run_query_tool = sql_toolkit.run_query_tool
    llm_with_tools = sql_toolkit.llm.bind_tools([run_query_tool, python_toolkit.tool()])
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}

def route_tool(state: MessagesState) -> Literal[END, "check_query", "run_python"]:  # pyright: ignore[reportInvalidTypeForm]
    messages = state["messages"]
    last_message = messages[-1]
    assert isinstance(last_message, AIMessage)
    tool_calls = last_message.tool_calls
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

def should_get_table_meta(_: MessagesState, config: RunnableConfig) -> Literal["react_node", "list_tables"]:
    if config['configurable']['first_time']: # type: ignore
        return 'list_tables'
    else:
        return 'react_node'

class SQLAgentConfigSchema(TypedDict):
    first_time: bool
    python_runner: PythonRunnerTool
    sql_toolkit: SQLDatabaseTool

class SQLAgent:

    def __init__(self, sql_db_uri: str, python_tool: PythonRunnerTool, model:str = "openai:gpt-4.1"):

        # initialization.
        self.llm = init_chat_model(model)
        self.sql_tool = SQLDatabaseTool(db_uri=sql_db_uri, llm=self.llm)
        self.python_tool = python_tool

        # graph building.
        self._build_graph()

        # initialize runtime configuration.
        self._init_config()

    def _build_graph(self):

        builder = StateGraph(MessagesState, config_schema=SQLAgentConfigSchema)
        builder.add_node(list_tables)
        builder.add_node(call_get_schema)
        builder.add_node(self.sql_tool.get_schema_tool_node)
        builder.add_node(react_node)
        builder.add_node(check_query)
        builder.add_node(self.sql_tool.run_query_tool_node)
        builder.add_node(self.python_tool.tool_node())

        builder.add_conditional_edges(START, should_get_table_meta)
        builder.add_edge("list_tables", "call_get_schema")
        builder.add_edge("call_get_schema", "get_schema")
        builder.add_edge("get_schema", "react_node")
        builder.add_conditional_edges(
            "react_node",
            route_tool,
        )
        builder.add_edge("check_query", "run_query")
        builder.add_edge("run_query", "react_node")
        builder.add_edge("run_python", "react_node")

        self.memory = MemorySaver()
        self.agent = builder.compile(checkpointer=self.memory)

    def print_ascii(self):
        self.agent.get_graph().print_ascii()

    def display_graph(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            drawer = PngDrawer()
            png_path = os.path.join(tmpdir, '_lang_graph.png')
            drawer.draw(self.agent.get_graph(), png_path)
            show_image(png_path)

    def _init_config(self):

        self.config = {
                'configurable': {
                    'thread_id': 1,
                    'first_time': True,
                    'python_runner': self.python_tool,
                    'sql_toolkit': self.sql_tool
                }
        }

    def chat(self, question:str, debug=False):

        if debug:
            for step in self.agent.stream(
                {"messages": [HumanMessage(question)]},
                stream_mode="values",
                config=self.config # type: ignore
            ):
                step["messages"][-1].pretty_print()
        else:
            result = self.agent.invoke(
                {"messages": [HumanMessage(question)]},
                config=self.config # type: ignore
            )
            result['messages'][-1].pretty_print()

        # don't run steps needed only the first time.
        self.config['configurable']['first_time'] = False

