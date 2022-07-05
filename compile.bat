@echo off
echo The Python package "pyinstaller" is required to perform this command; is it installed?
echo You may also need to activate your Python virtual environment
echo:

pyinstaller -F --paths=venv\Lib\site-packages main.py