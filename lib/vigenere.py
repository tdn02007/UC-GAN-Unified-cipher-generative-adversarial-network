### ref : https://github.com/CrypTools/VigenereCipher

LETTERS = 'abcdefghijklmnopqrstuvwxyz'

def encrypt(initial, key):

	initial = initial.lower()

	msg = []
	output = ''
	cnt = 0
	for char in initial:
		if char in LETTERS:
			shift = ord(key[cnt % len(key)]) - 97
			#print(shift)
			output += LETTERS[(LETTERS.index(char) + shift) % len(LETTERS)]
			cnt += 1

		'''
		if char in LETTERS:
			if ord(char) in range(65,91):		## ord : return ASCII number
				msg.append((ord(char) - 65,0))	## for uppercase
			elif ord(char) in range(97,123):
				msg.append((ord(char) - 97,1))	## for lowercase
			else: msg.append(char)
		'''

	#print(msg)

	#key = [ord(char) - 65 for char in key.lower()]

	#print(key)

	'''
	for i in range(len(msg)):
		if type(msg[i]) == type(''): output += msg[i]
		else:
			value   = (msg[i][0] + key[i % len(key)]) % 26
			output += chr(value + 65 + msg[i][1] * 32)	## 32 : to represent lowercase
	'''

	return output


####################### should be revised later... 
def decrypt(initial, key):

	initial = initial.lower()

	msg = []
	output = ''
	cnt = 0
	for char in initial:
		if char in LETTERS:
			shift = ord(key[cnt % len(key)]) - 97
			#print(shift, end=', ')
			output += LETTERS[(LETTERS.index(char) - shift) % len(LETTERS)]
			cnt += 1
	
	return output


	'''
	initial = initial.lower()

	msg = []
	output = ''
	for char in initial:
		if char in LETTERS:
			if ord(char) in range(65,91):
				msg.append((ord(char) - 65,0))
			elif ord(char) in range(97,123):
				msg.append((ord(char) - 97,1))
			else: msg.append(char)

	key = [ord(char) - 65 for char in key.lower()]



	for i in range(len(msg)):
		if type(msg[i]) == type(''): output += msg[i]
		else:
			value   = (msg[i][0] - key[i % len(key)]) % 26
			output += chr(value + 65 + msg[i][1] * 32)

	return output
	'''
