"""Starts the Phoenix monitoring dashboard on the local machine.

Run it with:

    python run_phoenix.py

Then open http://localhost:6006 in your browser. Keep this terminal
window open the whole time you are working; closing it stops the
dashboard. Stop it on purpose by pressing Enter in this window (or
Ctrl and C if the fallback server is running).
"""

import subprocess
import sys

import phoenix as px

try:
    session = px.launch_app()
except RuntimeError:
    # On some machines, especially the first time Phoenix builds its
    # local database, the quick-start launcher gives up before the
    # dashboard finishes starting. Running the server directly does
    # the same job with no time limit.
    print()
    print("Quick start took too long; starting the server directly")
    print("instead. The dashboard is the same: http://localhost:6006")
    print("Stop it by pressing Ctrl and C in this window.")
    print()
    subprocess.run([sys.executable, "-m", "phoenix.server.main", "serve"])
    sys.exit(0)

print()
print("Phoenix is running. Open this address in your browser:")
print("    http://localhost:6006")
print()
input("Press Enter here when you want to stop Phoenix... ")
