cls

del cloth.obj main.obj log.txt

rc.exe resource.rc

nvcc.exe -c -o main.obj main.cu

cl.exe /c /EHsc /I"C:\freetype\freetype-2.13.2\include" /I"C:\freetype\freetype-2.13.2\" text.cpp 

link.exe /LIBPATH:"C:\glew-2.1.0\lib\Release\x64" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64"  /LIBPATH:"C:\freetype\freetype-2.13.2\lib\win64" main.obj  resource.res text.obj opengl32.lib user32.lib kernel32.lib gdi32.lib

main.exe