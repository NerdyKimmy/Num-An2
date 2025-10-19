import numpy as np


def print_system(A, b, system_name=""):
    n = len(b)

    if system_name:
        print(f"\n{system_name}")
        print("=" * 50)

    print("Система рівнянь:")
    for i in range(n):
        equation = ""
        for j in range(n):
            if A[i][j] != 0:
                if equation and A[i][j] > 0:
                    equation += " + "
                elif equation and A[i][j] < 0:
                    equation += " - "
                elif not equation and A[i][j] < 0:
                    equation += "-"

                coeff = abs(A[i][j])
                if coeff != 1:
                    equation += f"{coeff}"
                equation += f"x{j + 1}"

        equation += f" = {b[i]}"
        print(equation)


def check_solution(A, x, b, method_name=""):
    calculated_b = np.dot(A, x)
    residual = np.max(np.abs(calculated_b - b))

    if method_name:
        print(f"\nПеревірка розв'язку ({method_name}):")

    print(f"A * x = {[f'{val:.6f}' for val in calculated_b]}")
    print(f"b     = {[f'{val:.6f}' for val in b]}")
    print(f"Максимальна нев'язка: {residual:.2e}")

    return residual


def print_solution(x, method_name=""):
    if method_name:
        print(f"\nРозв'язок системи ({method_name}):")

    for i, val in enumerate(x):
        print(f"x{i + 1} = {val:.6f}")