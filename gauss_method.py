import numpy as np


def gauss_elimination(A, b, verbose=True):
    n = len(b)
    determinant = 1
    swap_count = 0

    if verbose:
        print("Початкова матриця A та вектор b:")
        print(A)
        print("b =", b)
        print("-" * 50)

    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[i], b[max_row] = b[max_row], b[i]
            swap_count += 1
            if verbose:
                print(f"Обмін рядків {i} і {max_row}")

        pivot = A[i][i]
        determinant *= pivot

        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

        if verbose:
            print(f"Ітерація {i + 1}:")
            print("Матриця A:")
            print(A)
            print("Вектор b:")
            print(b)
            print("-" * 50)

    determinant *= (-1) ** swap_count

    if verbose:
        print(f"Визначник матриці: {determinant}")

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i][i]

    return x, determinant


def find_inverse(A, verbose=True):
    n = A.shape[0]
    A_ext = np.copy(A)
    E = np.eye(n)
    A_inv = np.copy(E)

    if verbose:
        print("Початкова матриця A та одинична матриця E:")
        print(A_ext)
        print(E)
        print("-" * 50)

    for i in range(n):
        max_row = i + np.argmax(np.abs(A_ext[i:, i]))
        if max_row != i:
            A_ext[[i, max_row]] = A_ext[[max_row, i]]
            A_inv[[i, max_row]] = A_inv[[max_row, i]]
            if verbose:
                print(f"Обмін рядків {i} і {max_row}")

        pivot = A_ext[i][i]
        A_ext[i] = A_ext[i] / pivot
        A_inv[i] = A_inv[i] / pivot

        for j in range(i + 1, n):
            factor = A_ext[j][i]
            A_ext[j] = A_ext[j] - factor * A_ext[i]
            A_inv[j] = A_inv[j] - factor * A_inv[i]

        if verbose:
            print(f"Ітерація {i + 1} (прямий хід):")
            print("Матриця A:")
            print(A_ext)
            print("Обернена матриця (на даний момент):")
            print(A_inv)
            print("-" * 50)

    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            factor = A_ext[j][i]
            A_ext[j] = A_ext[j] - factor * A_ext[i]
            A_inv[j] = A_inv[j] - factor * A_inv[i]

        if verbose:
            print(f"Ітерація {n - i} (зворотній хід):")
            print("Матриця A:")
            print(A_ext)
            print("Обернена матриця (на даний момент):")
            print(A_inv)
            print("-" * 50)

    return A_inv


def solve_gauss_system(A, b, verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print("МЕТОД ГАУСА")
        print("=" * 70)

    solution, determinant = gauss_elimination(np.copy(A), b.copy(), verbose)

    # Обернена матриця
    if verbose:
        print("\nЗнаходження оберненої матриці:")
    A_inv = find_inverse(A.copy(), verbose)

    return solution, determinant, A_inv