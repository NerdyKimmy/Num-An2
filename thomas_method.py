def thomas_method(A, b, verbose=True):
    n = len(A)

    if verbose:
        print("Метод прогонки:")
        print("-" * 50)

    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and abs(A[i][j]) > 1e-10:
                if verbose:
                    print(f"Увага: Матриця не є строго тридіагональною (елемент A[{i}][{j}] = {A[i][j]})")

    a = [0] * n
    b_diag = [0] * n
    c = [0] * n
    d = b.copy()

    for i in range(n):
        b_diag[i] = A[i][i]
        if i > 0:
            a[i] = A[i][i - 1]
        if i < n - 1:
            c[i] = A[i][i + 1]

    if verbose:
        print(f"Нижня діагональ a: {a}")
        print(f"Головна діагональ b: {b_diag}")
        print(f"Верхня діагональ c: {c}")
        print(f"Вектор d: {d}")
        print("-" * 50)

    alpha = [0] * n
    beta = [0] * n

    alpha[0] = -c[0] / b_diag[0]
    beta[0] = d[0] / b_diag[0]

    for i in range(1, n - 1):
        denominator = b_diag[i] + a[i] * alpha[i - 1]
        alpha[i] = -c[i] / denominator
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denominator

    denominator = b_diag[n - 1] + a[n - 1] * alpha[n - 2]
    beta[n - 1] = (d[n - 1] - a[n - 1] * beta[n - 2]) / denominator

    if verbose:
        print("Прямий хід:")
        print(f"alpha: {[f'{x:.6f}' for x in alpha]}")
        print(f"beta: {[f'{x:.6f}' for x in beta]}")
        print("-" * 50)

    x = [0] * n
    x[n - 1] = beta[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x


def solve_thomas_system(A, b, verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print("МЕТОД ПРОГОНКИ")
        print("=" * 70)

    solution = thomas_method(A, b, verbose)

    return solution