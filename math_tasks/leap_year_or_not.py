# The year can be evenly divided by 4, is a leap year, unless:
#   The year can be evenly divided by 100, it is not a leap year, unless:
#       The year is also evenly divisible by 400. Then it is a leap year.


def is_leap(year: int) -> bool:
    """
    :param year: The year can be evenly divided by 4, is a leap year, unless:
    The year can be evenly divided by 100, it is not a leap year, unless:
    The year is also evenly divisible by 400. Then it is a leap year.
    :return: True if the year is a leap year, else False
    """
    leap = False

    # Write your logic here
    if not bool(year % 4):
        if not bool(year % 100):
            if not bool(year % 400):
                leap = True
        else:
            leap = True
    return leap


if __name__ == '__main__':
    print('It is a leap year' if is_leap(int(input("Enter a year: "))) else 'It is not a leap year')
