import lib.caeser as caeser
import lib.vigenere as vigenere
import lib.substitution as substitution

KEY = "defg"
TEXT = "Brown"

encrypt_c = caeser.encrypt(TEXT, 98)
encrypt_v = vigenere.encrypt(TEXT, KEY)
encrypt_s = substitution.encrypt(TEXT)

print(encrypt_c)
print(encrypt_v)
print(encrypt_s)
