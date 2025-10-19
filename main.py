import numpy as np
from gauss_method import solve_gauss_system
from thomas_method import solve_thomas_system
from zeidel_method import solve_zeidel_system
from utils import print_system, check_solution, print_solution


def main():

    A1 = np.array([
        [4, 3, 1, 0],
        [-2, 2, 6, 1],
        [0, 5, 2, 3],
        [0, 1, 2, 7]
    ], dtype=float)

    b1 = np.array([14, 31, 33, 45], dtype=float)

    print_system(A1, b1, "СИСТЕМА 1")

    solution1, det1, inv_A1 = solve_gauss_system(A1, b1, verbose=True)
    print_solution(solution1, "метод Гауса")
    print(f"Визначник матриці: {det1:.6f}")
    check_solution(A1, solution1, b1, "метод Гауса")

    A2 = np.array([
        [1, 2, 0],
        [2, 2, 3],
        [0, 3, 2]
    ], dtype=float)

    b2 = np.array([5, 15, 12], dtype=float)

    print_system(A2, b2, "\nСИСТЕМА 2")

    solution2 = solve_thomas_system(A2, b2, verbose=True)
    print_solution(solution2, "метод прогонки")
    check_solution(A2, solution2, b2, "метод прогонки")

    A3 = np.array([
        [4, 0, 1, 0],
        [0, 3, 0, 2],
        [1, 0, 5, 1],
        [0, 2, 1, 4]
    ], dtype=float)

    b3 = np.array([7, 14, 20, 23], dtype=float)

    print_system(A3, b3, "\nСИСТЕМА 3")

    solution3, iterations3 = solve_zeidel_system(A3, b3, epsilon=1e-6, verbose=True)
    print_solution(solution3, "метод Зейделя")
    print(f"Кількість ітерацій: {iterations3}")
    check_solution(A3, solution3, b3, "метод Зейделя")

    print("\n" + "=" * 70)
    print("ВСІ СИСТЕМИ РОЗВ'ЯЗАНІ УСПІШНО!")
    print("=" * 70)


if __name__ == "__main__":
    main()