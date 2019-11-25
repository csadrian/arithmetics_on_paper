def number_to_base(n, b=10):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def is_prime(x):
        if x < 2:
            return False
        else:
            for n in range(2, x):
                if x % n == 0:
                   return False
            return True

def primes_lt(n):
    n = int(n) + 1
    res = []
    for i in range(1, n):
        if is_prime(i):
            res.append(i)
    return res

class Symbols:
    add = 11
    sub = 12
    is_prime = 13
    is_divisible_by = 14
    factorize = 15
    div = 16
    yes = 17
    no = 18
    sqrt = 19
    base_conversion = 20
    product = 21
    eq = 22

    @classmethod
    def visual(cls, n):
        symbols = {
            cls.add: '+',
            cls.sub: '-',
            cls.is_prime: 'p?',
            cls.is_divisible_by: '|',
            cls.factorize: 'fact',
            cls.div: '/',
            -1: ' ',
            cls.yes: '✓',
            cls.no: 'X',
            cls.sqrt: '√',
            cls.base_conversion: '→b',
            cls.product: '*',
            cls.eq: '=',
        }
        for i in range(10):
            symbols[i] = str(i)
        return symbols[n]
