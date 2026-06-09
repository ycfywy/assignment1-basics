"""
chr(0) 会返回空字符 U+0000
__repr() 会将其转义为 '\x00'  print() 打印时会输出真实的空字符，所以通常看不见。

 当这个字符出现在文本中时，Python 字符串会正常保存它，但它通常没有可见显示效果
 ，所以 "this is a test" + chr(0) + "string" 
 打印出来看起来像 this is a teststring，但中间其实仍然包含一个空字符。

output:

'\x00'
this is a teststring

"""

print(chr(0))

print(repr(chr(0)))

print("this is a test" + chr(0) + "string")


"""

utf-8 相比于 16 32 更加节省空间
兼容ASCII
且当前互联网上都是该格式


utf-8 format
0xxxxxxx              -> 1 字节字符，ASCII
110xxxxx 10xxxxxx     -> 2 字节字符
1110xxxx 10xxxxxx 10xxxxxx -> 3 字节字符
11110xxx 10xxxxxx 10xxxxxx 10xxxxxx -> 4 字节字符

"""
test_string = "Hello! 我是马牛逼"
utf8_encoded = test_string.encode()
print(utf8_encoded)

print(type(utf8_encoded))
print(list(utf8_encoded))

print(utf8_encoded.decode('utf-8'))

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
print(decode_utf8_bytes_to_str_wrong("hello ???".encode("utf-8")))

# print(b'\xc3\x28'.decode("utf-8"))