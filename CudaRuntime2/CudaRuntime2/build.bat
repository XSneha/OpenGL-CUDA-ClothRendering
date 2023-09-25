cls

rc.exe resource.rc

nvcc.exe -c -o cloth.obj kernel.cu

link.exe /LIBPATH:"C:\glew-2.1.0\lib\Release\x64" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64" cloth.obj  resource.res opengl32.lib user32.lib kernel32.lib gdi32.lib

cloth.exe