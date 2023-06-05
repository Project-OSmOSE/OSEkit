cd %SRC_DIR%

echo on

copy %RECIPE_DIR%\\CMakeLists.txt %SRC_DIR%\\CMakeLists.txt
mkdir build
cd build

REM https://github.com/conda/conda-build/issues/2850
REM set "CXXFLAGS=%CXXFLAGS:-GL=%"
REM set "CFLAGS=%CFLAGS:-GL=%"

cmake -G "NMake Makefiles" ^
	  -DCMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX% ^
	  -DCMAKE_BUILD_TYPE=Release ^
	  ..
if errorlevel 1 exit 1

nmake
if errorlevel 1 exit 1

nmake install
if errorlevel 1 exit 1
cd %SRC_DIR%

echo "copy stdbool.h %LIBRARY_PREFIX%\include ..."
copy /b stdbool.h %LIBRARY_PREFIX%\include
if errorlevel 1 exit 1

