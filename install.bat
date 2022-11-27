@echo off

echo,
echo ------------------------------------------------------------------
echo Checking python version...
echo ------------------------------------------------------------------
echo,

python --version 4>nul
if %errorlevel% neq 0 (
    echo Python is not installed or not in the PATH.
    echo Please install Python 3.x.x from http://www.python.org/download/
    echo and make sure it is in the PATH.
    echo,
    echo Press any key to exit...
    pause >nul
    exit /b 1
)


echo,
echo ------------------------------------------------------------------
echo installing requirements...
echo ------------------------------------------------------------------
echo,

python -m pip install -r requirements.txt


echo,
echo ------------------------------------------------------------------
echo Running the application
echo ------------------------------------------------------------------
echo,

echo WARNING: keep this window open to keep the application running... 

python main.py