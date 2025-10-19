import numpy as np


def zeidel_method(A, b, epsilon=1e-6, max_iterations=100, verbose=True):
    n = len(b)
    x = np.zeros_like(b, dtype=float)
    iteration = 0

    if verbose:
        print("Метод Зейделя:")
        print(f"Початкове наближення: {x}")
        print(f"Точність: {epsilon}")
        print(f"Максимальна кількість ітерацій: {max_iterations}")
        print("-" * 50)

    for iteration in range(1, max_iterations + 1):
        x_prev = x.copy()

        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_prev[i + 1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        error = np.max(np.abs(x - x_prev))

        if verbose:
            print(f"Ітерація {iteration}:")
            print(f"x = {[f'{val:.6f}' for val in x]}")
            print(f"Похибка: {error:.2e}")

        if error < epsilon:
            if verbose:
                print(f"Досягнуто точність {epsilon} за {iteration} ітерацій")
            break

    if iteration == max_iterations and verbose:
        print(f"Досягнуто максимальну кількість ітерацій ({max_iterations})")

    if verbose:
        print("-" * 50)

    return x, iteration


def solve_zeidel_system(A, b, epsilon=1e-6, max_iterations=100, verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print("МЕТОД ЗЕЙДЕЛЯ")
        print("=" * 70)

    solution, iterations = zeidel_method(A, b, epsilon, max_iterations, verbose)

    return solution, iterations