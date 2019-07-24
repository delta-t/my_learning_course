# The year can be evenly divided by 4, is a leap year, unless:
#   The year can be evenly divided by 100, it is not a leap year, unless:
#       The year is also evenly divisible by 400. Then it is a leap year.


def is_leap(check_year):
    leap = False

    # Write your logic here
    if not bool(check_year % 4):
        if not bool(check_year % 100):
            if not bool(check_year % 400):
                leap = True
        else:
            leap = True
    return leap


year = int(input())
print(is_leap(year))
