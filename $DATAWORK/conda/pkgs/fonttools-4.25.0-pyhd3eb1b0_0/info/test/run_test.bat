



fonttools subset --help
IF %ERRORLEVEL% NEQ 0 exit /B 1
ttx -h
IF %ERRORLEVEL% NEQ 0 exit /B 1
pyftsubset --help
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
