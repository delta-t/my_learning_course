from typing import TypeVar, Iterable


num = TypeVar('num', int, float)


def sum(items: Iterable[num]) -> num:
    accum = 0
    for item in items:
        accum += item
    return accum


if __name__ == '__main__':
    print(sum([1, 2, 3, 4, 5]))
