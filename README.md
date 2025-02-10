# dsp_helpers
Helpful functions for DSP

This project uses Pipenv to manage dependencies. 
Modify `venv.bat` to point to the correct python executable and 
Run `venv.bat` to setup the virtual environment.

## Setting Up the Virtual Environment

To set up the virtual environment, follow these steps:

1. **Modify the Python Path:**
    - Open the `venv.bat` file.
    - Modify the line that points to your Python installation to match your local setup:
      ```bat
      @REM modify this line to point to your python installation
      C:\Users\ksanoo\AppData\Local\Programs\Python\Python312\python.exe -m venv %VENV_DIR%
      ```

2. **Run the `venv.bat` Script:**
    - Execute the `venv.bat` script to set up the virtual environment. This script will:
        - Set necessary environment variables.
        - Create a virtual environment in the specified directory.
        - Install `pipenv` in the virtual environment.
        - Synchronize the virtual environment with the dependencies specified in the `Pipfile.lock`.
        - Activate the virtual environment.