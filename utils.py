class Symbols:
    NUM_SYMBOLS = 38

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
    greater = 23
    sort = 24
    last_digit = 25
    remainder = 26
    question = 27
    answer = 28
    decimal = 29
    end = 30
    comma = 31
    fraction = 32
    percentage = 33
    gcd = 34
    lcm = 35
    leftbracket = 36
    rightbracket = 37

    @classmethod
    def visual(cls, n):
        symbols = {
            cls.add: '+',
            cls.sub: '-',
            cls.is_prime: 'p?',
            cls.is_divisible_by: '|',
            cls.factorize: 'fact',
            cls.div: ':',
            0: ' ',
            cls.yes: '✓',
            cls.no: 'X',
            cls.sqrt: '√',
            cls.base_conversion: '→b',
            cls.product: '*',
            cls.eq: '=',
            cls.greater: '>',
            cls.sort: 'sort',
            cls.question: 'Q:',
            cls.answer: 'A:',
            cls.decimal: '.',
            cls.end: '∎',
            cls.comma: ',',
            cls.fraction: '/',
            cls.percentage: '%',
            cls.gcd: 'gcd',
            cls.lcm: 'lcm',
            cls.leftbracket : '(',
            cls.rightbracket: ')'
        }
        for i in range(1, 11):
            symbols[i] = str(i-1)
        return symbols.get(n, '?')

    @classmethod
    def digits_to_symbols(cls, digits):
        digits = list(digits)
        for i, digit in enumerate(digits):
            if digit in range(0, 10):
                digits[i] = digit + 1
        return digits

def int_to_base(n, b=10):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def number_to_symbols(n):
    if n == 0:
        return [0]
    digits = []
    if n < 0:
        digits.append(Symbols.sub)
        n *= -1
    if '.' in str(n):
        integer_str, fractional_str = str(n).split('.')
        part1, part2 = int_to_base(int(integer_str)),\
            int_to_base(int(fractional_str))
        digits += part1
        digits.append(Symbols.decimal)
        digits += part2
    else:
        digits = int_to_base(n)
    return Symbols.digits_to_symbols(digits)

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

