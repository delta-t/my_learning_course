# Input data
number = input("Enter the number:")

total_sum = 0 # Initially, sum == 0
for digit in number:
    total_sum += int(digit)
print("The total of input digits is {}.".format(total_sum))
