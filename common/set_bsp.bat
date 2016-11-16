@echo OFF
echo Setting VisualStudio environment variables
pushd %VCINSTALLDIR%\bin\amd64
call vcvars64.bat

echo Setting up the environment to run OpenCL BSP for Pikes Peak...
set AOCL_BOARD_PACKAGE_ROOT=C:\altera\BSP_263944
set PATH=%ALTERAOCLSDKROOT%\bin;%PATH%
set PATH=%ALTERAOCLSDKROOT%\host\windows64\bin;%PATH%
set PATH=%AOCL_BOARD_PACKAGE_ROOT%\Software\SDK\x64\Release;%PATH%
set PATH=%QSYS_ROOTDIR%;%PATH%
set PATH=C:\Program Files (x86)\GnuWin32\bin;%PATH%

set CL_CONTEXT_COMPILER_MODE_ALTERA=3

echo ALTERAOCLSDKROOT is set to %ALTERAOCLSDKROOT%
echo AOCL_BOARD_PACKAGE_ROOT is set to %AOCL_BOARD_PACKAGE_ROOT%
echo Add to PATH: %ALTERAOCLSDKROOT%\bin
echo Add to PATH: %ALTERAOCLSDKROOT%\host\windows64\bin
echo Add to PATH: %AOCL_BOARD_PACKAGE_ROOT%\Software\SDK\x64\Release
echo Add to PATH: %QSYS_ROOTDIR%
echo Add to PATH: C:\Program Files (x86)\GnuWin32\bin

rem cd %AOCL_BOARD_PACKAGE_ROOT%
popd
:end
