from __future__ import print_function
import cgt

a = cgt.scalar('a')
b = cgt.scalar('b')

y = a * b

multiply = cgt.function([a,b], y)

print(multiply(1, 2)) # 2
print(multiply(3, 3)) # 9
