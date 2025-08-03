from importlib.resources import files
from pathlib import Path
import subprocess

# AGP home folder.
agpdir = files('agentrun_plus')
# AGP docker compose script.
agentrun_script = agpdir.joinpath('docker-compose.sh')
subprocess.run(
        [
            str(agentrun_script),
            'dev',
            'up'
        ],
        cwd = str(agpdir)
)
    
