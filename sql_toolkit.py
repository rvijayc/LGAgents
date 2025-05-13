from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

class SQLDatabaseTool:

    def __init__(self, db_uri: str, llm):

        self.db = SQLDatabase.from_uri(db_uri)
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

def call_get_schema(state: MessagesState, config: RunnableConfig):
    """
    Force the model to query the schema of the SQL database.
    """
    # Note that LangChain enforces that all models accept `tool_choice="any"`
    # as well as `tool_choice=<string name of tool>`.
    sql_toolkit: SQLDatabaseTool = config['configurable']['sql_toolkit']
    get_schema_tool = sql_toolkit.get_schema_tool
    llm_with_tools = sql_toolkit.llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])

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
    """
    A node that employs the LLM to check the passed SQL query parameter.
    """
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
    llm_with_tools = sql_toolkit.llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}

