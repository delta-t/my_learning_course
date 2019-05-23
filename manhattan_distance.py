# Вводятся координаты начальной и целевой клеток шахматной доски. Определить, одинакового ли цвета обе клетки.
x_start = int(input())
y_start = int(input())
x_target = int(input())
y_target = int(input())

# Вычисление манхеттенской метрики (L1-норма)
L1 = abs(x_target - x_start) + abs(y_target - y_start)

# Если L1 кратна 2, значит обе клетки одинакового цвета, иначе - разного
print("YES") if L1 % 2 == 0 else print("NO")
