def hoar_sort(array_a: list) -> None:
    if len(array_a) <= 1:
        return
    border = array_a[0]
    left, middle, right = [], [], []
    for x in array_a:
        if x < border:
            left.append(x)
        elif x > border:
            right.append(x)
        else:
            middle.append(x)

    hoar_sort(left)
    hoar_sort(right)
    cnt = 0
    for x in left + middle + right:
        array_a[cnt] = x
        cnt += 1
