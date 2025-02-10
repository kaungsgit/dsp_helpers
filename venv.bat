set PIPENV_CUSTOM_VENV_NAME=dsp_helpers
set PIPENV_IGNORE_VIRTUALENVS=1
set PYTHONDONTWRITEBYTECODE=0
set VENV_DIR=%~dp0\out\pipenv-bootstrap
set WORKON_HOME=%~dp0\out\virtualenv

@REM modify this line to point to your python installation
C:\Users\ksanoo\AppData\Local\Programs\Python\Python312\python.exe -m venv %VENV_DIR%

%VENV_DIR%\\Scripts\\python -m pip install pipenv=="2023.12.1"
%VENV_DIR%\\Scripts\\python -m pipenv sync
%WORKON_HOME%\\%PIPENV_CUSTOM_VENV_NAME%\\Scripts\\activate