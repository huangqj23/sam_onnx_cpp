
// #pragma once
// #include <iostream>

// wchar_t* multi_Byte_To_Wide_Char(std::string& pKey)
// {
//     // string 转 char*
//     const char* pCStrKey = pKey.c_str();
//     // 第一次调用返回转换后的字符串长度，用于确认为wchar_t*开辟多大的内存空间
//     int pSize = MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, NULL, 0);
//     wchar_t* pWCStrKey = new wchar_t[pSize];
//     // 第二次调用将单字节字符串转换成双字节字符串
//     MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, pWCStrKey, pSize);
//     // 不要忘记在使用完wchar_t*后delete[]释放内存
//     return pWCStrKey;
// }

#pragma once
#include <iostream>
#include <string>
#include <locale>
#include <codecvt>

wchar_t* multi_Byte_To_Wide_Char(const std::string& pKey) {
    // 创建转换器
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    // 将 std::string 转换为 std::wstring
    std::wstring wstr = converter.from_bytes(pKey);

    // 将 std::wstring 转换为 wchar_t*
    wchar_t* wcstr = new wchar_t[wstr.size() + 1];
    std::wcscpy(wcstr, wstr.c_str());

    return wcstr;
}
