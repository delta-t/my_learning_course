'''
Пользователь вводит длину массива случайных элементов. Выводится неотсортированный массив чисел и отсортированный по возрастанию.
'''
from random import randint


def min_to_max(arr):
    for idx in range(len(arr) - 1):
        for j in range(len(arr) - idx - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

    return arr


new_array = []
N = int(input("Enter the length of array:"))
for i in range(N):
    new_array.append(randint(1, 99))

sorted_array = min_to_max(new_array.copy())
print("Array:\n", new_array, "\nFrom min to max:\n", sorted_array)
