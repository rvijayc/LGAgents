import os
import urllib.request
import IPython

from python_toolkit import PythonRunnerTool
from sql_code_agent import SQLAgent, SQLiteAgentPolicy

# ignore any pre-installed dependencies needed by the above code.
IGNORE_DEPDENCIES=['src']
# ignore any unsafe functions (SQLAlchemy has a function called "compile").
IGNORE_UNSAFE_FUNCTIONS=['compile']

CHINOOK_DATABASE_HINTS="""
Here are some additional hints about tables and columns the database.

Table InvoiceLine:
    - Column UnitPrice is in US dollar units.

"""

class ChinookSQLitePolicy(SQLiteAgentPolicy):

    def __init__(self):
        db_file = 'Chinook_Sqlite.sqlite'
        if not os.path.isfile(db_file):
            print(f'Downloading {db_file} ...')            
            url = f'https://github.com/lerocha/chinook-database/releases/download/v1.4.5/{db_file}'
            urllib.request.urlretrieve(url, db_file)
        super().__init__(db_file)

    def database_hints(self):
        return CHINOOK_DATABASE_HINTS

def main():
    
    with PythonRunnerTool(
            ignore_dependencies=IGNORE_DEPDENCIES,
            ignore_unsafe_functions=IGNORE_UNSAFE_FUNCTIONS,
            debug=True
    ) as python_tool:
        policy = ChinookSQLitePolicy()
        agent = SQLAgent(policy, python_tool)
        IPython.embed()

if __name__ == "__main__":
    main()
