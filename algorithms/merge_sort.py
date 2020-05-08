def merge(array_a: list, array_b: list) -> list:
    """

    :param array_a:
    :param array_b:
    :return:
    """
    i = j = k = 0
    array_c = [0] * (len(array_a) + len(array_b))
    while i < len(array_a) and j < len(array_b):
        if array_a[i] <= array_b[j]:
            array_c[k] = array_a[i]
            i += 1
            k += 1
        else:
            array_c[k] = array_b[j]
            j += 1
            k += 1

    while i < len(array_a):
        array_c[k] = array_a[i]
        i += 1
        k += 1

    while j < len(array_b):
        array_c[k] = array_b[j]
        j += 1
        k += 1
    return array_c


def merge_sort(array_a: list):
    if len(array_a) <= 1:
        return
    middle = len(array_a) // 2
    left = array_a[:middle]
    right = array_a[middle:]
    merge_sort(left)
    merge_sort(right)
    result = merge(left, right)
    for i in range(len(array_a)):
        array_a[i] = result[i]


if __name__ == '__main__':
    print(merge([5, 6, 7], [8, 9, 10]))
