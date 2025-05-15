import IPython

from python_toolkit import PythonRunnerTool
from sql_code_agent import SQLCodeAgent

# this is the code that loads and exposes the database as a module to the LLM.
# The docker image must come preinstalled with any dependencies needed for
# this.
DATABASE_LOADER="""
import os
from sqlalchemy import create_engine
script_dir = os.path.dirname(os.path.abspath(__file__))
db = os.path.join(script_dir, 'Chinook.db')
engine = create_engine(f"sqlite:///{db}")
__all__ = [ 'engine' ]
"""
# ignore any pre-installed dependencies needed by the above code.
IGNORE_DEPDENCIES=['src']

def main():
    
    with PythonRunnerTool(ignore_dependencies=IGNORE_DEPDENCIES) as python_tool:
        python_tool.copy_file_to_container('Chinook.db', '/code/src')
        python_tool.copy_code_to_container(DATABASE_LOADER, '/code/src/database.py')
        agent = SQLCodeAgent(python_tool)
        IPython.embed()

if __name__ == "__main__":
    main()
