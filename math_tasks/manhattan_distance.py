def manhattan_distance(point_a: tuple, point_b: tuple) -> bool:
    """

    :param point_a:
    :param point_b:
    :return:
    """
    return True if (abs(point_b[0] - point_a[0]) + abs(point_b[1] - point_a[1])) % 2 == 0 else False


if __name__ == '__main__':
    # Вводятся координаты начальной и целевой клеток шахматной доски. Определить, одинакового ли цвета обе клетки.
    x_start = int(input())
    y_start = int(input())
    x_target = int(input())
    y_target = int(input())
    # Если L1 кратна 2, значит обе клетки одинакового цвета, иначе - разного
    print("YES") if manhattan_distance((x_start, y_start), (x_target, y_target)) else print("NO")
