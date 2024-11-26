from decimal import *

getcontext().prec = 1000

a = Decimal(2).sqrt()+Decimal(3).sqrt()
a = a**1980
b = a*10
b = b.to_integral(rounding='ROUND_DOWN')
print(b%100)

