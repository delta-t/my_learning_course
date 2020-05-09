def generate(cur: str, opened: int, closed: int, n: int) -> None:
    """

    :param cur:
    :param opened:
    :param closed:
    :param n:
    :return:
    """
    if len(cur) == 2*n:
        if opened == closed:
            print(cur)
        return

    if opened < n:
        generate(cur + '(', opened + 1, closed, n)
    if closed < opened:
        generate(cur + ')', opened, closed + 1, n)


def parens(n: int) -> None:
    """

    :param n:
    :return:
    """
    generate('', 0, 0, n)


if __name__ == '__main__':
    parens(3)
