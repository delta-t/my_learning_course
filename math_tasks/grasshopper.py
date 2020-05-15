def count_min_cost(n: int, price: list) -> int:
    """

    :param n:
    :param price:
    :return:
    """
    cost = [price[0], price[0] + price[1]] + [0]*(n - 1)
    for i in range(2, n + 1):
        cost[i] = price[i] + min(cost[i - 1], cost[i - 2])
    return cost[n]


if __name__ == '__main__':
    print(count_min_cost(3, [0, 2, 1, 2, 2]))
