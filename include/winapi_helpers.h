#ifndef WINAPI_HELPERS_H
#define WINAPI_HELPERS_H

#include <windows.h>

HWND CreateWinAPIWindow(const char* title, int width, int height);
void RenderBitmap(HWND hwnd, unsigned char* bitmap, int width, int height);
void WaitForWindowToClose();

#endif // !WINAPI_HELPERS_H