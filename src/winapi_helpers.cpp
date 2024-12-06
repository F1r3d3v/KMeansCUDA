#include "winapi_helpers.h"
#include "shellscalingapi.h"

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	default:
		return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}
}

HWND CreateWinAPIWindow(const char* title, int width, int height)
{
	HINSTANCE hInstance = GetModuleHandle(NULL);

	// Set DPI awareness
	SetProcessDpiAwareness(PROCESS_SYSTEM_DPI_AWARE);

	// Register window class
	const char* CLASS_NAME = "Window Class";
	WNDCLASS wc = { };
	wc.lpfnWndProc = WindowProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = CLASS_NAME;
	RegisterClass(&wc);

	// Create window
	DWORD dwStyle = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE;

	// Calculate the required size of the window rectangle based on desired client area size
	RECT windowRect = { 0, 0, width, height };
	AdjustWindowRectEx(&windowRect, dwStyle, FALSE, 0);

	HWND hwnd = CreateWindowEx(
		0, CLASS_NAME, title,
		dwStyle, CW_USEDEFAULT, CW_USEDEFAULT,
		windowRect.right - windowRect.left,
		windowRect.bottom - windowRect.top,
		nullptr, nullptr, hInstance, nullptr);

	if (hwnd == nullptr)
		return nullptr;

	ShowWindow(hwnd, SW_SHOW);
	return hwnd;
}

void RenderBitmap(HWND hwnd, unsigned char* bitmap, int width, int height)
{
	PAINTSTRUCT ps;
	HDC hdc = BeginPaint(hwnd, &ps);

	BITMAPINFO bmi = { 0 };
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = width;
	bmi.bmiHeader.biHeight = height;
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 24;
	bmi.bmiHeader.biCompression = BI_RGB;

	StretchDIBits(hdc, 0, 0, width, height, 0, 0, width, height, bitmap, &bmi, DIB_RGB_COLORS, SRCCOPY);

	EndPaint(hwnd, &ps);
}

void WaitForWindowToClose()
{
	MSG msg = { 0 };

	// Message loop
	while (GetMessage(&msg, nullptr, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);

		// Check if WM_QUIT message is received
		if (msg.message == WM_QUIT)
			break;
	}
}