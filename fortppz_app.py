import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import math
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Калькулятор A/B тестов",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Калькулятор A/B тестов</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Параметры теста")

    test_type = st.selectbox(
        "Тип A/B теста:",
        [
            "t-test (независимые выборки)",
            "t-test (парный)",
            "Mann-Whitney U",
            "Chi-square",
            "Z-test для пропорций"
        ],
        index=0
    )

    st.subheader("Статистические параметры")
    alpha = st.slider("Уровень значимости (alpha)", 0.01, 0.20, 0.05, 0.01)
    power = st.slider("Мощность теста (1-beta)", 0.70, 0.99, 0.80, 0.01)
    beta = 1 - power

    st.subheader("Размер выборки")
    sample_size = st.slider("Размер выборки (n)", 10, 1000, 100, 10)

    st.subheader("Дополнительно")
    allocation_ratio = st.slider(
        "Соотношение размеров групп B/A (r = n_B / n_A)",
        0.1, 5.0, 1.0, 0.1,
        help="Если r=1 - равные размеры; если r=2 - группа B в 2 раза больше A."
    )

    if test_type.startswith("t-test"):
        st.subheader("Параметры для t-test")
        effect_size = st.slider("Размер эффекта (Cohen's d)", 0.05, 2.0, 0.5, 0.05)
        alternative = st.radio("Тип гипотезы", ["Двусторонняя", "Односторонняя (A < B)", "Односторонняя (A > B)"])
        df_input = st.slider("Степени свободы (для визуализации)", 1, 200, 30, 1)
    elif test_type == "Chi-square":
        st.subheader("Параметры для chi-square test")
        df_input = st.slider("Степени свободы", 1, 20, 3, 1)
        alternative = "Двусторонняя"
        effect_size_chi = st.slider("Размер эффекта (w)", 0.1, 0.8, 0.3, 0.05,
                                    help="Размер эффекта w: 0.1 - малый, 0.3 - средний, 0.5 - большой")
    elif test_type == "Z-test для пропорций":
        st.subheader("Параметры для Z-test пропорций")
        p1 = st.slider("Ожидаемая пропорция в группе A (p1)", 0.01, 0.99, 0.5, 0.01)
        p2 = st.slider("Ожидаемая пропорция в группе B (p2)", 0.01, 0.99, 0.6, 0.01)
        alternative = st.radio("Тип гипотезы", ["Двусторонняя", "Односторонняя (A < B)", "Односторонняя (A > B)"])
    else:
        st.subheader("Параметры для непараметрического теста")
        alternative = st.radio("Тип гипотезы", ["Двусторонняя", "Односторонняя (A < B)", "Односторонняя (A > B)"])
        effect_size_mw = st.slider("Размер эффекта (r)", 0.1, 0.8, 0.3, 0.05,
                                   help="Размер эффекта r (корреляция рангов): 0.1 - малый, 0.3 - средний, 0.5 - большой")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Визуализация распределений", "Расчет размера выборки", "Проверка по наблюдаемым данным"])

with tab1:
    st.subheader("График распределения и критические области")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if test_type.startswith("t-test"):
            df = df_input
            try:
                x_min = stats.t.ppf(0.0005, df)
                x_max = stats.t.ppf(0.9995, df)
                if not np.isfinite(x_min) or not np.isfinite(x_max):
                    raise Exception
            except Exception:
                x_min, x_max = -8, 8

            x = np.linspace(x_min, x_max, 1200)
            y = stats.t.pdf(x, df)
            if alternative == "Двусторонняя":
                t_left = stats.t.ppf(alpha / 2, df)
                t_right = stats.t.ppf(1 - alpha / 2, df)
                ax.fill_between(x, 0, y, where=(x <= t_left), alpha=0.6, color='red')
                ax.fill_between(x, 0, y, where=(x >= t_right), alpha=0.6, color='red')
                ax.axvline(t_left, color='red', linestyle='--')
                ax.axvline(t_right, color='red', linestyle='--', label=f'критические: {t_left:.2f}, {t_right:.2f}')
            elif alternative == "Односторонняя (A < B)":
                t_left = stats.t.ppf(alpha, df)
                ax.fill_between(x, 0, y, where=(x <= t_left), alpha=0.6, color='red')
                ax.axvline(t_left, color='red', linestyle='--', label=f'критическое: {t_left:.2f}')
            else:
                t_right = stats.t.ppf(1 - alpha, df)
                ax.fill_between(x, 0, y, where=(x >= t_right), alpha=0.6, color='red')
                ax.axvline(t_right, color='red', linestyle='--', label=f'критическое: {t_right:.2f}')

            ax.plot(x, y, linewidth=2, label=f't-распределение (df={df})')
            ax.set_title("t-распределение и области отклонения")
            ax.set_xlabel("t")
            ax.set_ylabel("Плотность")
            ax.legend()
            ax.grid(alpha=0.3)

        elif test_type == "Chi-square":
            df = df_input
            try:
                x_max = stats.chi2.ppf(0.9995, df)
                if not np.isfinite(x_max):
                    raise Exception
            except Exception:
                x_max = max(20, df * 3)

            x = np.linspace(0.0001, x_max, 1200)
            y = stats.chi2.pdf(x, df)
            chi2_crit = stats.chi2.ppf(1 - alpha, df)
            ax.plot(x, y, linewidth=2, label=f'chi-square (df={df})')
            ax.fill_between(x, 0, y, where=(x >= chi2_crit), alpha=0.6, color='red')
            ax.axvline(chi2_crit, color='red', linestyle='--', label=f'критическое: {chi2_crit:.2f}')
            ax.set_xlabel("chi-square")
            ax.set_ylabel("Плотность")
            ax.set_title("Распределение Chi-square")
            ax.legend()
            ax.grid(alpha=0.3)

        elif test_type == "Z-test для пропорций":
            x_min = stats.norm.ppf(0.0005)
            x_max = stats.norm.ppf(0.9995)
            x = np.linspace(x_min, x_max, 1200)
            y = stats.norm.pdf(x)
            if alternative == "Двусторонняя":
                z_l = stats.norm.ppf(alpha / 2)
                z_r = stats.norm.ppf(1 - alpha / 2)
                ax.fill_between(x, 0, y, where=(x <= z_l), alpha=0.6, color='red')
                ax.fill_between(x, 0, y, where=(x >= z_r), alpha=0.6, color='red')
                ax.axvline(z_l, color='red', linestyle='--')
                ax.axvline(z_r, color='red', linestyle='--', label=f'критические: {z_l:.2f}, {z_r:.2f}')
            elif alternative == "Односторонняя (A < B)":
                z_l = stats.norm.ppf(alpha)
                ax.fill_between(x, 0, y, where=(x <= z_l), alpha=0.6, color='red')
                ax.axvline(z_l, color='red', linestyle='--', label=f'критическое: {z_l:.2f}')
            else:
                z_r = stats.norm.ppf(1 - alpha)
                ax.fill_between(x, 0, y, where=(x >= z_r), alpha=0.6, color='red')
                ax.axvline(z_r, color='red', linestyle='--', label=f'критическое: {z_r:.2f}')

            ax.plot(x, y, linewidth=2)
            ax.set_title(f"Нормальное распределение (Z), p1={p1}, p2={p2}")
            ax.set_xlabel("Z")
            ax.set_ylabel("Плотность")
            ax.legend()
            ax.grid(alpha=0.3)

        elif test_type == "Mann-Whitney U":
            n1 = max(1, int(sample_size / (1 + allocation_ratio)))
            n2 = max(1, sample_size - n1)

            mean_u = n1 * n2 / 2
            std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

            x_min = mean_u - 4 * std_u
            x_max = mean_u + 4 * std_u
            x = np.linspace(x_min, x_max, 1200)
            y = stats.norm.pdf(x, mean_u, std_u)

            if alternative == "Двусторонняя":
                u_left = stats.norm.ppf(alpha / 2, mean_u, std_u)
                u_right = stats.norm.ppf(1 - alpha / 2, mean_u, std_u)
                ax.fill_between(x, 0, y, where=(x <= u_left), alpha=0.6, color='red')
                ax.fill_between(x, 0, y, where=(x >= u_right), alpha=0.6, color='red')
                ax.axvline(u_left, color='red', linestyle='--')
                ax.axvline(u_right, color='red', linestyle='--', label=f'критические: {u_left:.2f}, {u_right:.2f}')
            elif alternative == "Односторонняя (A < B)":
                u_left = stats.norm.ppf(alpha, mean_u, std_u)
                ax.fill_between(x, 0, y, where=(x <= u_left), alpha=0.6, color='red')
                ax.axvline(u_left, color='red', linestyle='--', label=f'критическое: {u_left:.2f}')
            else:
                u_right = stats.norm.ppf(1 - alpha, mean_u, std_u)
                ax.fill_between(x, 0, y, where=(x >= u_right), alpha=0.6, color='red')
                ax.axvline(u_right, color='red', linestyle='--', label=f'критическое: {u_right:.2f}')

            ax.plot(x, y, linewidth=2)
            ax.set_title(f"Приближенное распределение U-статистики (n1={n1}, n2={n2})")
            ax.set_xlabel("U")
            ax.set_ylabel("Плотность")
            ax.legend()
            ax.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Ошибка построения графика: {str(e)}")

with tab2:
    st.subheader("Расчет необходимого размера выборки")

    if test_type.startswith("t-test"):
        d = effect_size
        if alternative == "Двусторонняя":
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)

        r = allocation_ratio
        nA = math.ceil(((z_alpha + z_beta) ** 2 * (1 + 1.0 / r)) / (d ** 2))
        nB = math.ceil(r * nA)

        st.success(f"""
            Рекомендуемый размер выборки (t-test):
            - n_A = {nA:,}
            - n_B = {nB:,}  (r = {r})
            - Общий: {nA + nB:,}
            - Cohen's d = {d:.2f}
            - Мощность: {power:.2f}, alpha = {alpha:.2f}
            """)

    elif test_type == "Z-test для пропорций":
        effect_prop = abs(p2 - p1)
        if effect_prop <= 0:
            st.warning("p1 и p2 равны - эффект = 0. Введите разные пропорции.")
        else:
            if alternative == "Двусторонняя":
                z_alpha = stats.norm.ppf(1 - alpha / 2)
            else:
                z_alpha = stats.norm.ppf(1 - alpha)
            z_beta = stats.norm.ppf(power)
            r = allocation_ratio

            p_pool = (p1 + p2) / 2.0
            term1 = z_alpha * math.sqrt((1 + 1.0 / r) * p_pool * (1 - p_pool))
            term2 = z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / r)
            nA = math.ceil((term1 + term2) ** 2 / (effect_prop ** 2))
            nB = math.ceil(r * nA)

            st.success(f"""
                    Рекомендуемый размер выборки (Z-пропорции):
                    - n_A = {nA:,}
                    - n_B = {nB:,}  (r = {r})
                    - Общий: {nA + nB:,}
                    - p1 = {p1:.2f}, p2 = {p2:.2f}, эффект = {effect_prop:.2f}
                    """)

    elif test_type == "Chi-square":
        if alternative == "Двусторонняя":
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)

        w = effect_size_chi

        if df_input == 1:
            n_total = math.ceil(((z_alpha + z_beta) ** 2) / (w ** 2))
            nA = math.ceil(n_total / 2)
            nB = math.ceil(n_total / 2)
        else:
            n_total = math.ceil(((z_alpha + z_beta) ** 2) / (w ** 2) * (df_input + 1))
            n_per_cell = math.ceil(n_total / (df_input + 1))
            nA = n_per_cell * (df_input + 1) // 2
            nB = n_per_cell * (df_input + 1) // 2

        st.success(f"""
            Рекомендуемый размер выборки (Chi-square):
            - n_A = {nA:,}
            - n_B = {nB:,}
            - Общий: {nA + nB:,}
            - Размер эффекта w = {w:.2f}
            - Степени свободы = {df_input}
            - Мощность: {power:.2f}, alpha = {alpha:.2f}
            """)

    elif test_type == "Mann-Whitney U":
        if alternative == "Двусторонняя":
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)
        r_effect = effect_size_mw
        d_equivalent = 2 * r_effect / math.sqrt(1 - r_effect ** 2)

        r = allocation_ratio
        nA = math.ceil(
            ((z_alpha + z_beta) ** 2 * (1 + 1.0 / r)) / (d_equivalent ** 2) * 1.05)
        nB = math.ceil(r * nA)
        st.success(f"""
            Рекомендуемый размер выборки (Mann-Whitney U):
            - n_A = {nA:,}
            - n_B = {nB:,}  (r = {r})
            - Общий: {nA + nB:,}
            - Размер эффекта r = {r_effect:.2f}
            - Эквивалентный Cohen's d ≈ {d_equivalent:.2f}
            - Мощность: {power:.2f}, alpha = {alpha:.2f}
            """)

with tab3:
    st.subheader("Проверка гипотезы по вашим наблюдаемым данным")
    st.markdown("Введите фактические наблюдения / сводные статистики для получения статистики теста и p-value.")

    if test_type == "t-test (независимые выборки)":
        colA, colB = st.columns(2)
        with colA:
            st.markdown("Группа A")
            meanA = st.number_input("Среднее A", value=0.0, step=0.1, format="%.2f")
            sdA = st.number_input("Стандартное отклонение A", value=1.0, step=0.1, format="%.2f")
            nA = st.number_input("n_A", min_value=2, value=30, step=1)
        with colB:
            st.markdown("Группа B")
            meanB = st.number_input("Среднее B", value=0.5, step=0.1, format="%.2f")
            sdB = st.number_input("Стандартное отклонение B", value=1.0, step=0.1, format="%.2f")
            nB = st.number_input("n_B", min_value=2, value=30, step=1)

        if st.button("Рассчитать t-статистику и p-value (Welch)"):
            se = math.sqrt(sdA ** 2 / nA + sdB ** 2 / nB)
            t_stat = (meanA - meanB) / se
            df_w = (sdA ** 4 / (nA ** 2) + sdB ** 4 / (nB ** 2)) / (
                    (sdA ** 4) / ((nA ** 2) * (nA - 1)) + (sdB ** 4) / ((nB ** 2) * (nB - 1)))
            if alternative == "Двусторонняя":
                pval = 2 * stats.t.sf(abs(t_stat), df=df_w)
            elif alternative == "Односторонняя (A < B)":
                pval = stats.t.cdf(t_stat, df=df_w)
            else:
                pval = 1 - stats.t.cdf(t_stat, df=df_w)

            st.write(f"t = {t_stat:.2f}, df ≈ {df_w:.2f}, p-value = {pval:.2f}")
            st.write("Интерпретация: " + ("Отвергаем H0" if pval < alpha else "Не отвергаем H0"))

    elif test_type == "t-test (парный)":
        st.markdown("Введите разности (A - B) через запятую или введите сводные статистики.")
        diffs_text = st.text_area(
            "Список разностей (через запятую), либо оставьте пустым и введите разницу средних между парами наблюдений, стандартное отклонение между парами наблюдений  и размер выборки",
            value="")
        if diffs_text.strip() != "":
            try:
                diffs = np.array([float(x.strip()) for x in diffs_text.split(",") if x.strip() != ""])
                n = len(diffs)
                mean_diff = diffs.mean()
                sd_diff = diffs.std(ddof=1)
            except Exception as e:
                st.error("Ошибка чтения чисел: " + str(e))
                diffs = None
                n = 0
        else:
            mean_diff = st.number_input("Средняя разность (mean A - B)", value=0.0, format="%.2f")
            sd_diff = st.number_input("Стандартное отклонение разностей", value=1.0, format="%.2f")
            n = st.number_input("n (пар)", min_value=2, value=30, step=1)

        if st.button("Рассчитать парный t-test"):
            se = sd_diff / math.sqrt(n)
            t_stat = mean_diff / se
            df = n - 1
            if alternative == "Двусторонняя":
                pval = 2 * stats.t.sf(abs(t_stat), df)
            elif alternative == "Односторонняя (A < B)":
                pval = stats.t.cdf(t_stat, df)
            else:
                pval = 1 - stats.t.cdf(t_stat, df)
            st.write(f"t = {t_stat:.2f}, df = {df}, p-value = {pval:.2f}")
            st.write("Интерпретация: " + ("Отвергаем H0" if pval < alpha else "Не отвергаем H0"))

    elif test_type == "Z-test для пропорций":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Группа A")
            succA = st.number_input("Успехи в A (kA)", min_value=0, value=50, step=1)
            nA = st.number_input("n_A", min_value=1, value=100, step=1)
        with col2:
            st.markdown("Группа B")
            succB = st.number_input("Успехи в B (kB)", min_value=0, value=60, step=1)
            nB = st.number_input("n_B", min_value=1, value=100, step=1)

        if st.button("Рассчитать Z-test для пропорций"):
            pA_hat = succA / nA
            pB_hat = succB / nB
            p_pool = (succA + succB) / (nA + nB)
            se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / nA + 1 / nB))
            if se_pool == 0:
                st.error("SE = 0, невозможно рассчитать тест.")
            else:
                z = (pA_hat - pB_hat) / se_pool
                if alternative == "Двусторонняя":
                    pval = 2 * stats.norm.sf(abs(z))
                elif alternative == "Односторонняя (A < B)":
                    pval = stats.norm.cdf(z)
                else:
                    pval = 1 - stats.norm.cdf(z)
                st.write(f"p̂A = {pA_hat:.2f}, p̂B = {pB_hat:.2f}")
                st.write(f"Z = {z:.2f}, p-value = {pval:.2f}")
                st.write("Интерпретация: " + ("Отвергаем H0" if pval < alpha else "Не отвергаем H0"))

    elif test_type == "Chi-square":
        st.markdown("Введите таблицу 2x2 наблюдаемых частот (целые числа).")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Группа A**")
            c11 = st.number_input("Успех в A", min_value=0, value=30, step=1, key="c11")
            c12 = st.number_input("Неудача в A", min_value=0, value=20, step=1, key="c12")
        with col2:
            st.markdown("**Группа B**")
            c21 = st.number_input("Успех в B", min_value=0, value=10, step=1, key="c21")
            c22 = st.number_input("Неудача в B", min_value=0, value=40, step=1, key="c22")

        if st.button("Рассчитать chi-square test"):
            table = np.array([[c11, c12], [c21, c22]])
            chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
            st.write(f"χ² = {chi2:.2f}, df = {dof}, p-value = {p:.4f}")
            st.write("Ожидаемые частоты (матрица):")
            st.write(pd.DataFrame(expected.round(2),
                                  columns=["Успех", "Неудача"],
                                  index=["Группа A", "Группа B"]))
            st.write("Интерпретация: " + ("Отвергаем H0" if p < alpha else "Не отвергаем H0"))
            if expected.min() < 5:
                st.warning("В некоторых ячейках ожидаемая частота < 5 - рассмотрите точный критерий Фишера.")

    elif test_type == "Mann-Whitney U":
        st.markdown("Введите наблюдения для A и B через запятую (пример: 1.2, 1.5, 2.1).")
        a_text = st.text_area("Наблюдения A", value="1,2,3,4,5")
        b_text = st.text_area("Наблюдения B", value="2,3,4,5,6")

        if st.button("Рассчитать Mann-Whitney U"):
            try:
                A = np.array([float(x.strip()) for x in a_text.split(",") if x.strip() != ""])
                B = np.array([float(x.strip()) for x in b_text.split(",") if x.strip() != ""])
                u_stat, pval = stats.mannwhitneyu(A, B, alternative=(
                    'two-sided' if alternative == "Двусторонняя" else (
                        'less' if alternative == "Односторонняя (A < B)" else 'greater')))
                st.write(f"U = {u_stat:.2f}, p-value = {pval:.4f}")
                st.write("Интерпретация: " + ("Отвергаем H0" if pval < alpha else "Не отвергаем H0"))
            except Exception as e:

                st.error("Ошибка чтения данных: " + str(e))
