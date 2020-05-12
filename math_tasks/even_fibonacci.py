"""
Вывести все четные числа Фибоначчи, которые  меньше заданной величины
"""


def fibonacci(n: int) -> list:
    fib = [0, 1] + [0]*(n - 2)
    for i in range(2, n):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib


def even_fibonacci(greatest: int = 100) -> list:
    i = 0
    fibonacci_list = fibonacci(greatest)
    for number in fibonacci_list:
        if number % 2 == 0:
            i = number
            break
    return fibonacci_list[i:greatest:3]


if __name__ == '__main__':
    even_fibonacci()
