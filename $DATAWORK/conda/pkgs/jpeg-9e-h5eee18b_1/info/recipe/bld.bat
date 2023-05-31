set PATH=%PREFIX%\cmake-bin\bin;%PATH%

REM Configure step
copy jconfig.vc jconfig.h
if errorlevel 1 exit /b 1

mkdir %c_compiler%
pushd %c_compiler%
set CXXFLAGS=
set CFLAGS=

REM Build step
cmake -G "NMake Makefiles"                     ^
      -DCMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX%  ^
      -DCMAKE_BUILD_TYPE=Release               ^
      ..

if errorlevel 1 exit /b 1
cmake --build . --config Release --target INSTALL -- VERBOSE=1
if errorlevel 1 exit /b 1
popd
