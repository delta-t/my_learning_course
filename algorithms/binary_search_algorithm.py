'''
Пользователь вводит длину массива и число, которое необходимо найти в сгенерированном списке. 
На выходе выводится индекс соответствующего элемента или сообщение о том, что элемента в списке нет и количество итераций.
'''
# import necessary packages
from random import random

# get the length of array
N = int(input('input length of array:'))

# create an array
array = []
for i in range(N):
    array.append(int(random()*100))
array.sort()
print(array)

number = int(input())

# Binary search
low = 0
high = N - 1
step = 0
while low <= high:
    step += 1
    mid = (low + high) // 2
    if number < array[mid]:
        high = mid - 1
    elif number > array[mid]:
        low = mid + 1
    else:
        print("ID =", mid)
        break
else:
    print('No the number')

# number of the iterations
print('Step:', step)
