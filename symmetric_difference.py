"""
Given 2 sets of integers, M and N, print their symmetric difference in ascending order.
The term symmetric difference indicates those values that exist in either M or N
but do not exist in both
"""


M, N = set(list(input().split())), set(list(input().split()))
print('\n'.join(map(str, sorted(map(int, M.difference(N).union(N.difference(M)))))))
