import os
import urllib.request
import tempfile

import IPython

from langchain.chat_models import init_chat_model
from lg_agents import SQLiteAgentPolicy, SQLAgent

CHINOOK_DATABASE_HINTS="""
Here are some additional hints about tables and columns the database.

Table InvoiceLine:
    - Column UnitPrice is in millions of US dollar units.

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
    
    model = init_chat_model('openai:gpt-4.1')
    # start the runner using a context manager (to allow graceful shutdown).
    with tempfile.TemporaryDirectory() as tempdir:
        policy = ChinookSQLitePolicy()
        agent = SQLAgent(
                agentrun_url='http://localhost:8000',
                agent_policy=policy, 
                tmpdir=tempdir,
                show_artifacts=True,
                model=model
        )
        agent.chat('Generate a histogram of artists sorted by revenue for the first 15 artists', quiet=False)
        IPython.embed()

if __name__ == "__main__":
    main()
