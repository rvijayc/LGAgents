import os
import logging
import subprocess
import IPython, pdb
from typing import Literal, TypedDict, Optional
import tempfile, tarfile

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.tools import tool
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import docker
from docker import DockerClient

from external.AgentRun.agentrun import AgentRun, PIPInstallPolicy

import docker
import subprocess
import os


class PythonRunnerTool:

    def __init__(self):
        self.log = logging.getLogger('PYTHON_RUNNER')
        self.client: DockerClient = docker.from_env()
        script_path = os.path.dirname(os.path.abspath(__file__))
        self.code_runner_path = os.path.join(script_path, 'code_runner')

    def __enter__(self):
        self.log.info('Starting Python Runner Docker Image ...')
        subprocess.run(
                ['docker', 'compose', 'up', '--build', '-d'], 
                cwd=self.code_runner_path,
                check=True
                )
        # initialize the agent runner.
        self.container = self.client.containers.get('code_runner-python_runner-1')
        self.python_runner = AgentRun(
                container_name="code_runner-python_runner-1",
                cached_dependencies = [],
                install_policy=PIPInstallPolicy()
                )
        self.tool_node = ToolNode([run_python_tool], name="run_python")
        self.tmpdir = tempfile.TemporaryDirectory()
        return self

    def execute_code(self, code: str):
        return self.python_runner.execute_code_in_container(code)

    def configure(self, config: RunnableConfig):
        config['configurable']['python_runner'] = self

    def copy_file_from_container(
            self, 
            src_path: str,
            dst_folder: Optional[str]=None
            ) -> str:

        # get the archive
        stream, _ = self.container.get_archive(src_path)

        # use temporary folder itself if destination isn't specified.
        if not dst_folder:
            dst_folder = os.path.join(self.tmpdir.name)
        
        with tempfile.NamedTemporaryFile(delete_on_close=False) as tmpfp:

            # write the tar stream on to a temporary file ...
            for chunk in stream:
                tmpfp.write(chunk)
            tmpfp.close()

            # ... and extract it.
            with tarfile.open(tmpfp.name) as tar:
                tar.extractall(dst_folder)

        # return the destination file name.
        dst_path = os.path.join(dst_folder, os.path.basename(src_path))
        assert os.path.isfile(dst_path)
        return dst_path

    def runner_tool_node(self) -> ToolNode:
        return self.tool_node

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log.info('Stopping Python Runner Docker Image ...')
        subprocess.run(
                ['docker', 'compose', 'down'],
                cwd=self.code_runner_path,
                check=True
                )
        self.tmpdir.cleanup()

class PythonToolOutput(BaseModel):
    text: str = Field(description="Text output of the Python Program")
    plot_file_path: Optional[str] = Field(description="Optional plot file path (if the tool produced a plot")

# create a tool.
@tool("run_python_tool", parse_docstring=True)
def run_python_tool(python_code: str, abs_plot_file_path: Optional[str], config: RunnableConfig) -> PythonToolOutput:
    """
    Runs the specified python code provided and returns the standard output.
    This tool doesn't have any display capabilities, and hence, if you wish to
    generate a plot, store the plot/figure into a PNG file and specify
    its *absolute path* as one of the arguments to the tool.

    Args:
        python_code: Python code to run provided as a string.
        abs_plot_file_path: If the python code is generating a plot, specify
            the absolute path of the file containing the plot.
    """
    python_tool: PythonRunnerTool = config['configurable']['python_runner']
    python_runner = python_tool.python_runner
    if not python_runner:
        raise RuntimeError('Python runner is not configured by the runtime configuration!')
    output = python_runner.execute_code_in_container(python_code)
    if abs_plot_file_path:
        # copy the file locally
        png_file = python_tool.copy_file_from_container(abs_plot_file_path)
        return PythonToolOutput(
                text=output,
                plot_file_path=png_file
        )
    return PythonToolOutput(
            text=output,
            plot_file_path=None
    )

class SQLDatabaseTool:

    def __init__(self, db_uri: str, llm):

        self.db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        self.dialect = self.db.dialect
        self.llm = llm
        # get and analyze tools.
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()

        # create tool references and tool nodes.
        self.get_schema_tool = next(tool for tool in self.tools if tool.name == "sql_db_schema")
        self.get_schema_tool_node = ToolNode([self.get_schema_tool], name="get_schema")
        self.run_query_tool = next(tool for tool in self.tools if tool.name == "sql_db_query")
        self.run_query_tool_node = ToolNode([self.run_query_tool], name="run_query")
        self.list_tables_tool = next(tool for tool in self.tools if tool.name == "sql_db_list_tables")

    def configure(self, config: RunnableConfig):
        config['configurable']['sql_toolkit'] = self
        
# initialize chat model.
llm = init_chat_model("openai:gpt-4.1")

def list_tables(state: MessagesState, config: RunnableConfig):
    """
    Create a pre-determined tool call to list database table entries.
    """
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    # get the list tables tool from the runtime SQL database toolkit.
    list_tables_tool = config['configurable']['sql_toolkit'].list_tables_tool
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")

    return {"messages": [tool_call_message, tool_message, response]}

# Example: force a model to create a tool call
def call_get_schema(state: MessagesState, config: RunnableConfig):
    # Note that LangChain enforces that all models accept `tool_choice="any"`
    # as well as `tool_choice=<string name of tool>`.
    get_schema_tool = config['configurable']['sql_toolkit'].get_schema_tool
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}

class PythonToolOutputSummary(BaseModel):
    text: str = Field(description="Any text output to the user.")
    plot_path: Optional[str] = Field(description="If the python tool run produced a path, specify the path of the plot.")

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
"""
def generate_query(state: MessagesState, config: RunnableConfig):

    # generate a system message.
    sql_toolkit = config['configurable']['sql_toolkit']
    system_message = SystemMessage(generate_query_system_prompt.format(
        dialect=sql_toolkit.dialect,
        top_k=5
    ))

    # We do not force a tool call here, to allow the model to
    # respond naturally when it obtains the solution.
    run_query_tool = sql_toolkit.run_query_tool
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
"""
def check_query(state: MessagesState, config: RunnableConfig):
    sql_toolkit = config['configurable']['sql_toolkit']
    system_message = SystemMessage(check_query_system_prompt.format(
        dialect=sql_toolkit.dialect
    ))

    # Generate an artificial user message to check
    last_message = state["messages"][-1]
    assert isinstance(last_message, AIMessage)
    tool_call = last_message.tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    run_query_tool = sql_toolkit.run_query_tool
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}

def process_python_tool_output(state: MessagesState):
    """
    Allow the LLM to process the output of the python tool and return a structured response.
    """
    parser = PydanticOutputParser(pydantic_object=PythonToolOutputSummary)
    system_msg = SystemMessage(f"""
    Process the output of the previous python tool run and generate a response using this schema: {parser.get_format_instructions()}
    """)
    response = llm.invoke([system_msg] + state['messages'])
    assert isinstance(response.content, str)
    parsed = parser.parse(response.content)
    return {"messages": [response]}

def route_tool(state: MessagesState) -> Literal[END, "check_query", "run_python"]:
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

def should_get_table_meta(state: MessagesState, config: RunnableConfig) -> Literal["generate_query", "list_tables"]:
    if config['configurable']['first_time']: # type: ignore
        return 'list_tables'
    else:
        return 'generate_query'

class ConfigSchema(TypedDict):
    first_time: bool
    python_runner: Optional[PythonRunnerTool]
    sql_toolkit: Optional[SQLDatabaseTool]

sql_toolkit = SQLDatabaseTool(db_uri="sqlite:///Chinook.db", llm=llm)

with PythonRunnerTool() as python_tool:
    builder = StateGraph(MessagesState, config_schema=ConfigSchema)
    builder.add_node(list_tables)
    builder.add_node(call_get_schema)
    builder.add_node(sql_toolkit.get_schema_tool_node)
    builder.add_node(generate_query)
    builder.add_node(check_query)
    builder.add_node(sql_toolkit.run_query_tool_node)
    builder.add_node(python_tool.runner_tool_node())
    builder.add_node(process_python_tool_output)

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
    # add the python runner runtime configuration to be passed to the tool.
    python_tool.configure(config)
    sql_toolkit.configure(config)

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
