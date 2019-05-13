def factorial(num):
    if num == 0:
        return 1
    else:
        return num * factorial(num - 1)


n = int(input("Enter any positive integer:"))
if n >= 0:
    print("Factorial of %d is" % n, factorial(n))
else:
    print("Factorial of negative number cannot be found!")
