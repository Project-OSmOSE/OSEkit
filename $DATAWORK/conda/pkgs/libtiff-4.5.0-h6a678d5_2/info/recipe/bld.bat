mkdir build_%c_compiler%
cd build_%c_compiler%

cmake -GNinja                                    ^
      -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%"  ^
      -DCMAKE_PREFIX_PATH="%LIBRARY_PREFIX%"     ^
      -DCMAKE_BUILD_TYPE=Release                 ^
      -DCMAKE_C_FLAGS="%CFLAGS% -DWIN32"         ^
      -DCMAKE_CXX_FLAGS="%CXXFLAGS% -EHsc"       ^
      -DCMAKE_SHARED_LIBRARY_PREFIX=""           ^
      ..
if errorlevel 1 exit /b 1

cmake --build . --target install --config Release
if errorlevel 1 exit /b 1

copy "%LIBRARY_PREFIX%"\bin\tiff.dll "%LIBRARY_PREFIX%"\bin\libtiff.dll
if errorlevel 1 exit /b 1
