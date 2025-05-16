import os
import pdb
import tempfile
from typing import Literal, TypedDict, List
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.runnables.graph_png import PngDrawer
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from python_toolkit import PythonRunnerTool
from utils import show_image

SQLA_LIST_TABLES_PYTHON=r"""
from src.database import engine
from sqlalchemy import MetaData

# Create a metadata object
metadata = MetaData()
metadata.reflect(bind=engine)

# Get the table names
table_names = metadata.tables.keys()
print('Available tables are:', list(table_names))
"""

SQLA_GET_SCHEMA_PYTHON=r"""
from src.database import engine
from sqlalchemy import MetaData
from sqlalchemy.schema import CreateTable

# Load metadata
metadata = MetaData()
metadata.reflect(bind=engine)

# Generate CREATE TABLE statements
for table_name, table in metadata.tables.items():
    create_statement = str(CreateTable(table).compile(engine))
    print(f"Schema for '{table_name} table':\n{create_statement}\n")

"""
def _run_python_tool(code: str, config:RunnableConfig):
    tool_call = {
        "name": "run_python_tool",
        "args": {
            'python_code': code,
            'artifacts_abs_paths': []
        },
        "id": uuid4().hex,
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    python_runner = config['configurable']['python_runner'] # type: ignore
    tool_message = python_runner.tool().invoke(tool_call, config=config)
    return tool_call_message, tool_message

def list_tables(state: MessagesState, config: RunnableConfig):
    """
    Create a pre-determined tool call to list database table entries.
    """
    tool_call_message, tool_message = _run_python_tool(SQLA_LIST_TABLES_PYTHON, config)
    return {"messages": [tool_call_message, tool_message]}

def call_get_schema(state: MessagesState, config: RunnableConfig):
    """
    Force the model to query the schema of the SQL database.
    """
    tool_call_message, tool_message = _run_python_tool(SQLA_GET_SCHEMA_PYTHON, config)
    return {"messages": [tool_call_message, tool_message]}

class AnswerSchema(BaseModel):
    message: str = Field(description="""
    The answer to the user's question in markdown format. Include links to artifacts as appropriate.
    """)
    user_artifacts_abs_paths: List[str] = Field(description="""
    Local paths of artifacts (if any) produced as a part of the user's answer.
    """)

react_system_prompt ="""
You are an agent designed to interact with a SQL database. You'll do so by
writing and executing python code using a python runner tool
("run_python_tool") that is equipped with Python 3.12+.

The database is exposed to you using an SQLAlchemy Engine object which you can
import as follows:

```
from src.database import engine
```

Always include the above line when referring to the databse. You can now use
the engine object imported above to list tables, schema etc., and also to run
queries.

The python tool returns the stdout of the program and hence you can use print
statements to get the information you want. 

Given an input question, create a syntactically correct code to run, then look
at the results of the run, and return the answer. You can also use the tool to
create artifiacts such as plots to show the user.

Here is an example of code that lists all the tables in the database:

```
{example_code}
```

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

{output_instructions}
""".format(
        example_code=SQLA_LIST_TABLES_PYTHON,
        output_instructions=PydanticOutputParser(pydantic_object=AnswerSchema).get_format_instructions()
)

def react_node(state: MessagesState, config: RunnableConfig):
    """
    The main node in which we invoke the reasoning ability of the agent.
    """
    # create a system prompt.
    system_message = SystemMessage(react_system_prompt)
    # bind the python runner tool the the LLM.
    cfg = config['configurable'] # pyright: ignore[reportTypedDictNotRequiredAccess]
    llm_with_tools = cfg['llm'].bind_tools([cfg['python_runner'].tool()])
    # invoke the LLM.
    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}

def route_tool(state: MessagesState) -> Literal[END, 'run_python']:
    messages = state["messages"]
    last_message = messages[-1]
    assert isinstance(last_message, AIMessage)
    tool_calls = last_message.tool_calls
    if len(tool_calls) == 0:
        return END
    else:
        return 'run_python'

def should_get_table_meta(_: MessagesState, config: RunnableConfig) -> Literal["react_node", "list_tables"]:
    if config['configurable']['first_time']: # type: ignore
        return 'list_tables'
    else:
        return 'react_node'

class SQLAgentConfigSchema(TypedDict):
    first_time: bool
    llm: BaseChatModel
    python_runner: PythonRunnerTool

class SQLCodeAgent:

    def __init__(self, python_tool: PythonRunnerTool, model:str = "openai:gpt-4.1"):

        # initialization.
        self.llm = init_chat_model(model)
        self.python_tool = python_tool

        # graph building.
        self._build_graph()

        # initialize runtime configuration.
        self._init_config()

    def _build_graph(self):

        builder = StateGraph(MessagesState, config_schema=SQLAgentConfigSchema)

        # add nodes.
        builder.add_node(list_tables)
        builder.add_node(call_get_schema)
        builder.add_node(react_node)
        builder.add_node(self.python_tool.tool_node())

        # connect them.
        builder.add_conditional_edges(START, should_get_table_meta)
        builder.add_edge("list_tables", "call_get_schema")
        builder.add_edge("call_get_schema", "react_node")
        builder.add_conditional_edges(
            "react_node",
            route_tool,
        )
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

    def display_artifacts(self, artifacts: List[str]):
        """
        Displays artifacts generated by the Model.
        """
        for artifact in artifacts:
            if not os.path.isfile(artifact):
                raise RuntimeError(f'Artifact {artifact} is not found!')
            _, ext = os.path.splitext(artifact)
            match ext:
                case '.png': show_image(artifact)

    def artifacts(self) -> List[str]:
        """
        Returns the list of artifacts from the last AI message.
        """
        final_state = self.agent.get_state(self.config) # pyright: ignore[reportArgumentType]
        final_message = final_state.values['messages'][-1]
        answer: AnswerSchema = AnswerSchema.model_validate_json(final_message.content)
        return answer.user_artifacts_abs_paths

    def _init_config(self):

        self.config = {
                'recursion_limit': 10,
                'configurable': {
                    'thread_id': 1,
                    'first_time': True,
                    'python_runner': self.python_tool,
                    'llm': self.llm,
                }
        }

    def chat(self, question:str, quiet=False, show_artifacts=True):

        if not quiet:
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

        # don't run steps needed only the first time.
        self.config['configurable']['first_time'] = False
        
        print('================================== Final Answer ================================== ')
        final_state = self.agent.get_state(self.config) # pyright: ignore[reportArgumentType]
        final_message = final_state.values['messages'][-1]
        answer: AnswerSchema = AnswerSchema.model_validate_json(final_message.content)
        print(answer.message)
        if show_artifacts:
            self.display_artifacts(answer.user_artifacts_abs_paths)
