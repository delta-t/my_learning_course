from data_structures.heap import Heap


def heapify(arr: list) -> Heap:
    heap = Heap()

    for item in arr:
        heap.insert(item)
    return heap


def get_sorted_array(heap: Heap) -> list:
    arr = []
    while heap.size:
        arr.append(heap.extract_min())
    return arr


if __name__ == '__main__':
    tmp = heapify([5, 8, 10, 1, 4, 3, 3])
    print(tmp.values)
    print(get_sorted_array(tmp))
