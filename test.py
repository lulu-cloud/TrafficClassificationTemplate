# -*- coding: utf-8 -*-
# @Time : 2023/6/16 22:39
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : test.py
"""
@Description: 
"""
import secrets
from Crypto.Cipher import AES

def encrypt(plaintext, key):
    """Encrypt the plaintext using the given AES key"""
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
    return nonce + ciphertext + tag

def decrypt(ciphertext, key):
    """Decrypt the ciphertext using the given AES key"""
    nonce = ciphertext[:AES.block_size]
    tag = ciphertext[-AES.block_size:]
    ciphertext = ciphertext[AES.block_size:-AES.block_size]
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext).decode()
    try:
        cipher.verify(tag)
    except ValueError:
        raise ValueError("Invalid key or ciphertext")
    return plaintext

# 生成随机的 16 字节（128 位）AES 密钥
key = get_key()

# 加密和解密一个字符串
plaintext = "This is a secret message."
ciphertext = encrypt(plaintext, key)
decrypted_text = decrypt(ciphertext, key)

# 打印加密、解密前后的内容
print("Plaintext: ", plaintext)
print("Ciphertext: ", ciphertext)
print("Decrypted text: ", decrypted_text)
