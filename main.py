import numpy as np
from gauss_method import solve_gauss_system_complete
from thomas_method import solve_thomas_system
from zeidel_method import solve_zeidel_system
from utils import print_system


def main():
    print("=" * 70)
    print("РОЗВ'ЯЗАННЯ СИСТЕМ РІВНЯНЬ РІЗНИМИ МЕТОДАМИ")
    print("=" * 70)

    A1 = np.array([
        [4, 3, 1, 0],
        [-2, 2, 6, 1],
        [0, 5, 2, 3],
        [0, 1, 2, 7]
    ], dtype=float)

    b1 = np.array([14, 31, 33, 45], dtype=float)

    print_system(A1, b1, "СИСТЕМА 1 - Метод Гауса")

    results1 = solve_gauss_system_complete(A1, b1, verbose=True)

    print(f"\nРозв'язок системи:")
    for i, val in enumerate(results1['solution']):
        print(f"x{i + 1} = {val:.6f}")

    A2 = np.array([
        [1, 2, 0],
        [2, 2, 3],
        [0, 3, 2]
    ], dtype=float)

    b2 = np.array([5, 15, 12], dtype=float)

    print_system(A2, b2, "\nСИСТЕМА 2 - Метод прогонки")

    solution2 = solve_thomas_system(A2, b2, verbose=True)

    print(f"\nРозв'язок системи:")
    for i, val in enumerate(solution2):
        print(f"x{i + 1} = {val:.6f}")

    A3 = np.array([
        [4, 0, 1, 0],
        [0, 3, 0, 2],
        [1, 0, 5, 1],
        [0, 2, 1, 4]
    ], dtype=float)

    b3 = np.array([7, 14, 20, 23], dtype=float)

    print_system(A3, b3, "\nСИСТЕМА 3 - Метод Зейделя")

    solution3, iterations3 = solve_zeidel_system(A3, b3, epsilon=1e-6, verbose=True)

    print(f"\nРозв'язок системи:")
    for i, val in enumerate(solution3):
        print(f"x{i + 1} = {val:.6f}")
    print(f"Кількість ітерацій: {iterations3}")


if __name__ == "__main__":
    main()