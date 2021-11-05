import subprocess
from threading import Thread


def run_command(command: str, thread: bool = False):
    """Default method for executing shell commands with pytest."""
    command = command.split(" ")
    if thread:
        t = Thread(target=subprocess.run, args=(command,))
        t.start()
        return t
    else:
        subprocess.run(command)
        return None
