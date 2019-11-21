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
