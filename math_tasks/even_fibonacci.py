"""
Вывести все четные числа Фибоначчи, которые  меньше заданной величины
"""


def fibonacci():
    previous_digit, current_digit = 1, 1
    yield previous_digit
    while True:
        yield current_digit
        previous_digit, current_digit = current_digit, previous_digit + current_digit


def even_fibonacci(greatest=10000):
    for digit in fibonacci():
        if digit > greatest:
            return
        if digit % 2 == 0:
            yield digit


if __name__ == '__main__':
    print(list(even_fibonacci()))
