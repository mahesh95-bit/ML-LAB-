# Set A – Python Assignment
# Roll No: 24161

import random


# 1. Count pairs with sum 10
def count_pairs_with_sum_ten(numbers):
    count = 0
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == 10:
                count += 1
    return count


# 2. Find range of list
def find_range_of_list(numbers):
    if len(numbers) < 3:
        return "Range determination not possible"
    return max(numbers) - min(numbers)


# 3. Matrix multiplication
def multiply_matrices(a, b):
    size = len(a)
    result = [[0] * size for _ in range(size)]

    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += a[i][k] * b[k][j]
    return result


def matrix_power(matrix, power):
    size = len(matrix)
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    for _ in range(power):
        result = multiply_matrices(result, matrix)
    return result


# 4. Highest occurring alphabet character
def highest_occurring_character(text):
    count = {}
    for ch in text.lower():
        if ch.isalpha():
            count[ch] = count.get(ch, 0) + 1
    return max(count, key=count.get), max(count.values())


# 5. Mean, Median, Mode
def calculate_mean(numbers):
    return sum(numbers) / len(numbers)


def calculate_median(numbers):
    numbers.sort()
    mid = len(numbers) // 2
    if len(numbers) % 2 == 0:
        return (numbers[mid - 1] + numbers[mid]) / 2
    return numbers[mid]


def calculate_mode(numbers):
    freq = {}
    for n in numbers:
        freq[n] = freq.get(n, 0) + 1
    max_freq = max(freq.values())
    return [n for n in freq if freq[n] == max_freq]


# Main program
if __name__ == "__main__":

    print("1.", count_pairs_with_sum_ten([2, 7, 4, 1, 3, 6]))

    print("2.", find_range_of_list([5, 3, 8, 1, 0, 4]))

    matrix = [[1, 2], [3, 4]]
    print("3. Matrix power 2:")
    for row in matrix_power(matrix, 2):
        print(row)

    ch, cnt = highest_occurring_character("hippopotamus")
    print("4.", ch, cnt)

    nums = [random.randint(1, 10) for _ in range(25)]
    print("5. Numbers:", nums)
    print("Mean:", calculate_mean(nums))
    print("Median:", calculate_median(nums))
    print("Mode:", calculate_mode(nums))
