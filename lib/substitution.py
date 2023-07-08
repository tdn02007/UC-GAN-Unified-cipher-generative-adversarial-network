### ref : https://github.com/CrypTools/CaesarCipher

import random

LETTERS = 'abcdefghijklmnopqrstuvwxyz'

def encrypt(initial):
    """ Use : encrypt("message", 98)
    => 'gymmuay'
    """
    initial = initial.lower()
    output = ""

    key = 'qwertyuiopasdfghjklzxcvbnm'

    shift = []

    for j in range(len(key)):
        x = ord(key[j]) - 97
        shift.append(x)

    cnt = 0
    for char in initial:
        if char in LETTERS:
            output += LETTERS[shift[LETTERS.index(char)]]
            cnt += 1
	
    return output

def decrypt(initial):
    """ Use : decrypt('gymmuay', 98)
    => 'message'
    """
    initial = initial.lower()
    output = ""

    key_inv = 'kxvmcnophqrszyijadlegwbuft'

    shift = []

    for j in range(len(key_inv)):
        x = ord(key_inv[j]) - 97
        shift.append(x)

    cnt = 0
    for char in initial:
        if char in LETTERS:
            output += LETTERS[shift[LETTERS.index(char)]]
            cnt += 1
	
    return output
