import IPython

from python_toolkit import PythonRunnerTool
from sql_direct_agent import SQLAgent

def main():
    
    with PythonRunnerTool() as python_tool:
        
        agent = SQLAgent('sqlite:///Chinook.db', python_tool)
        IPython.embed()

if __name__ == "__main__":
    main()
