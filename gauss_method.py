import numpy as np


def gauss_elimination_with_pm(A, b, verbose=True):
    n = len(b)
    A = A.copy().astype(float)
    b = b.copy().astype(float)

    P_matrices = []
    M_matrices = []  ь

    if verbose:
        print("ПОЧАТКОВА СИСТЕМА:")
        print("A =")
        print(A)
        print(f"b = {b}")
        print("=" * 60)

    for i in range(n):
        if verbose:
            print(f"\n--- КРОК {i + 1} ---")

        P_i = np.eye(n)

        max_row = i + np.argmax(np.abs(A[i:, i]))
        if max_row != i:
            if verbose:
                print(f"Обмін рядків {i + 1} ↔ {max_row + 1}")

            P_i[[i, max_row]] = P_i[[max_row, i]]

            A = P_i @ A
            b = P_i @ b

        P_matrices.append(P_i)

        if verbose:
            print(f"Матриця перестановок P_{i + 1}:")
            print(P_i)
            print("Матриця A після перестановки:")
            print(A)

        M_i = np.eye(n)
        pivot = A[i, i]

        if verbose:
            print(f"Головний елемент: a_{i + 1}{i + 1} = {pivot:.4f}")

        for j in range(i + 1, n):
            factor = A[j, i] / pivot
            M_i[j, i] = -factor

            if verbose:
                print(f"m_{j + 1}{i + 1} = {factor:.4f}")

        if verbose:
            print(f"Матриця перетворення M_{i + 1}:")
            print(M_i)

        A = M_i @ A
        b = M_i @ b
        M_matrices.append(M_i)

        if verbose:
            print("Матриця A після перетворення:")
            print(A)
            print(f"Вектор b: {b}")

    P_total = np.eye(n)
    for P_i in P_matrices:
        P_total = P_i @ P_total

    M_total = np.eye(n)
    for M_i in reversed(M_matrices):
        M_total = M_i @ M_total

    if verbose:
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТ ПРЯМОГО ХОДУ:")
        print("Верхня трикутна матриця U:")
        print(A)
        print(f"Перетворений вектор b: {b}")
        print(f"Загальна матриця перестановок P:")
        print(P_total)
        print(f"Загальна матриця перетворень M:")
        print(M_total)

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        if verbose:
            print(f"x[{i + 1}] = ({b[i]:.4f} - {np.dot(A[i, i + 1:], x[i + 1:]):.4f}) / {A[i, i]:.4f} = {x[i]:.4f}")

    det = np.prod(np.diag(A)) * np.linalg.det(P_total)

    return x, det, A, P_total, M_total, P_matrices, M_matrices


def find_inverse_gauss(A, verbose=True):
    n = A.shape[0]
    A_inv = np.zeros((n, n))

    if verbose:
        print("\n" + "=" * 60)
        print("ЗНАХОДЖЕННЯ ОБЕРНЕНОЇ МАТРИЦІ МЕТОДОМ ГАУСА")
        print("Розв'язуємо A * X = I")
        print("Для кожного стовпця одиничної матриці:")

    for col in range(n):
        b_col = np.zeros(n)
        b_col[col] = 1

        if verbose:
            print(f"\n--- Стовпець {col + 1} ---")
            print(f"b = {b_col}")

        x_col, _, _, _, _, _, _ = gauss_elimination_with_pm(A.copy(), b_col, verbose=False)

        A_inv[:, col] = x_col

        if verbose:
            print(f"Розв'язок (стовпець {col + 1} оберненої матриці): {x_col}")

    if verbose:
        print(f"\nОБЕРНЕНА МАТРИЦЯ A⁻¹:")
        print(A_inv)

    return A_inv


def verify_solution(A, x, b, method_name=""):
    Ax = A @ x
    residual = np.max(np.abs(Ax - b))

    print(f"\n{method_name} - ПЕРЕВІРКА:")
    print(f"A * x = {Ax}")
    print(f"b     = {b}")
    print(f"Максимальна нев'язка: {residual:.2e}")

    return residual


def verify_inverse(A, A_inv):
    I_calculated = A @ A_inv
    I_expected = np.eye(A.shape[0])
    error = np.max(np.abs(I_calculated - I_expected))

    print(f"\nПЕРЕВІРКА ОБЕРНЕНОЇ МАТРИЦІ:")
    print("A * A⁻¹ =")
    print(I_calculated)
    print(f"Максимальна похибка: {error:.2e}")

    return error


def solve_gauss_system_complete(A, b, verbose=True):
    if verbose:
        print("=" * 70)
        print("МЕТОД ГАУСА З МАТРИЦЯМИ ПЕРЕТВОРЕНЬ")
        print("=" * 70)

    solution, determinant, U, P_total, M_total, P_list, M_list = gauss_elimination_with_pm(A, b, verbose)

    A_inv = find_inverse_gauss(A, verbose)

    residual_solution = verify_solution(A, solution, b, "Метод Гауса")
    residual_inverse = verify_inverse(A, A_inv)

    return {
        'solution': solution,
        'determinant': determinant,
        'inverse': A_inv,
        'U': U,
        'P_total': P_total,
        'M_total': M_total,
        'P_matrices': P_list,
        'M_matrices': M_list
    }


