import math

# Объемы (площади) фундаментальных областей для Z3 и Z4
# Для T^2/Z3: Im(tau) = sqrt(3)/2, площадь = Im(tau) / 3
V3 = (math.sqrt(3) / 2) / 3  # = sqrt(3)/6

# Для T^2/Z4: Im(tau) = 1, площадь = Im(tau) / 4
V4 = 1.0 / 4.0  # = 1/4

# Количество фиксированных точек
N3 = 3
N4 = 2  # для Z4 на торе обычно 2 точки порядка 4

# Эффективный фактор I, если он возникает из разности вкладов
I_derived = N3 * V3 - N4 * V4
I_theory = (math.sqrt(3) - 1) / 2

print(f"I (derived from N3*V3 - N4*V4) = {I_derived:.6f}")
print(f"I (theoretical, (sqrt(3)-1)/2) = {I_theory:.6f}")
print(f"Difference = {abs(I_derived - I_theory):.2e}")

# Проверим альтернативные комбинации
I_alt1 = (N3 * V3 + N4 * V4) / 2
print(f"Alternative 1 ((N3*V3+N4*V4)/2) = {I_alt1:.6f}")

I_alt2 = (N4 * V4) / (N3 * V3)
print(f"Alternative 2 (N4*V4)/(N3*V3) = {I_alt2:.6f}")

I_alt3 = 1.0 / (1 + N3 * V3)
print(f"Alternative 3 (1/(1+N3*V3)) = {I_alt3:.6f}")