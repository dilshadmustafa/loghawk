# Make sure to run firsttime_setup.bat before running this script to set up the environment and install dependencies.
# run the following commands in separate terminals to start the Temporal worker and pipeline

# Terminal 1
python -m loghawk.workflows.temporal.worker

# Terminal 2
python -m loghawk.workflows.temporal.start_pipeline

# Temporal Web UI is available at http://localhost:8233/namespaces/default/workflows
# Navigate to http://localhost:8233/namespaces/default/workflows
# to see Temporal workflow
