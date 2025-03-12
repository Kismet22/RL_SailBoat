import statsmodels.api as sm


# 定义成功次数和总试验次数
successes = 915  # 成功次数
n = 1000          # 总试验次数

# 计算成功率
p = successes / n
print(f"成功率: {p*100:.2f}%")

# 计算置信区间
# 使用 'wilson' 方法，这是一个常用且较为准确的方法
# alpha = 0.05 95%置信区间
# alpha = 0.01 99%置信区间
"""""""""
ci_low, ci_upp = sm.stats.proportion_confint(successes, n, alpha=0.01, method='wilson')
print(f"95% 置信区间 (Wilson): [{ci_low*100:.2f}%, {ci_upp*100:.2f}%]")
"""

# 也可以使用其他方法，如 'normal'（正态近似
ci_low_normal, ci_upp_normal = sm.stats.proportion_confint(successes, n, alpha=0.05, method='normal')
print(f"±{(ci_upp_normal - p) * 100:.2f}%")
print(f"95% 置信区间 (Normal): [{ci_low_normal*100:.2f}%, {ci_upp_normal*100:.2f}%]")

ci_low_normal, ci_upp_normal = sm.stats.proportion_confint(successes, n, alpha=0.01, method='normal')
print(f"99% 置信区间 (Normal): [{ci_low_normal*100:.2f}%, {ci_upp_normal*100:.2f}%]")

"""""""""
# 'agresti_coull'近似
ci_low_agresti, ci_upp_agresti = sm.stats.proportion_confint(successes, n, alpha=0.01, method='agresti_coull')
print(f"95% 置信区间 (Agresti-Coull): [{ci_low_agresti*100:.2f}%, {ci_upp_agresti*100:.2f}%]")
"""