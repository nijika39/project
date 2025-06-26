library(readxl)
library(moments)
library(tseries)
library(ggplot2)
library(MASS)
library(olsrr)
library(car)

# 描述性统计+正态性检验函数
descriptive_stats <- function(x, plot_qq = FALSE, digits = 3) {
  # 移除缺失值
  x <- na.omit(x)
  
  # 描述性统计
  stats_list <- list(
    "观测数" = length(x),
    "均值" = mean(x),
    "中位数" = median(x),
    "标准差" = sd(x),
    "最小值" = min(x),
    "最大值" = max(x),
    "偏度" = moments::skewness(x),
    "峰度" = moments::kurtosis(x)
  )
  
  # Jarque-Bera正态性检验
  jb_test <- tseries::jarque.bera.test(x)
  stats_list[["JB p值"]] <- jb_test$p.value
  
  # shapiro正态性检验(小样本)
  shapiro_test <- shapiro.test(x)
  stats_list[["shapiro p值"]] <- shapiro_test$p.value
  
  # 保留digits位小数
  formatted_stats <- lapply(stats_list, function(val) {
    if (is.numeric(val)) round(val, digits) else val
  })
  
  cat("\n描述性统计结果:\n")
  print(data.frame(统计量 = names(formatted_stats), 值 = unlist(formatted_stats)), 
        row.names = FALSE)
  
  if (plot_qq) {
    qqnorm(x, main = "正态Q-Q图")
    qqline(x, col = "red")
  }
  
  invisible(stats_list)
}

# 读入数据
data <- read_excel("睡眠质量数据.xlsx")
data <- na.omit(data)

# 一、描述性统计结果
x <- data$失眠指数
descriptive_stats(x, plot_qq = TRUE, digits = 2)

# 二、t检验
t_x1 <- data$失眠指数[data$性别 == "男"]
t_x2 <- data$失眠指数[data$性别 == "女"]

# 描述性统计结果
descriptive_stats(t_x1, plot_qq = TRUE, digits = 2)
descriptive_stats(t_x2, plot_qq = TRUE, digits = 2)

# t检验
t_test_result <- t.test(t_x1, t_x2)
t_test_result

# 方差齐性检验
var.test(t_x1, t_x2)

# 三、方差分析
# 获取睡眠质量指数
var_x1 <- data$失眠指数[data$年级 == "大一/大二"]
var_x2 <- data$失眠指数[data$年级 == "大三/大四"]
var_x3 <- data$失眠指数[data$年级 == "研究生及以上"]

# 描述性统计结果
descriptive_stats(var_x1, plot_qq = TRUE, digits = 2)
descriptive_stats(var_x2, plot_qq = TRUE, digits = 2)
descriptive_stats(var_x3, plot_qq = TRUE, digits = 2)

# 创建数据框
variance_analyse_data <- data.frame(
  values = c(var_x1, var_x2, var_x3),
  group = factor(rep(c("x1", "x2", "x3"), times = c(length(var_x1), length(var_x2), length(var_x3))))
)

# 方差分析
anova_result <- aov(values ~ group, data = variance_analyse_data)
summary(anova_result)

# 方差齐性检验
# Bartlett检验(要求数据正态分布)
bartlett.test(values ~ group, data = variance_analyse_data)

# Levene检验(对正态性要求较低)
leveneTest(values ~ group, data = variance_analyse_data)

# 多重比较检验
tukey_result <- TukeyHSD(anova_result)
tukey_result

# 可视化结果
plot(tukey_result, las = 1)

# 多元回归分析
# 转换分类变量
data$性别 <- factor(data$性别)
data$年级 <- factor(data$年级, levels = c("大一/大二", "大三/大四", "研究生及以上"))

# 标准化数据
连续变量 <- c("每天的学习时间", "每天的体育锻炼时间", "每天使用手机时间",
              "学业压力", "宿舍条件", "咖啡因摄入")

data[连续变量] <- lapply(data[连续变量], function(x) {
  as.numeric(x)
})
data[连续变量] <- scale(data[连续变量])

x1 <- data$性别
x2 <- data$年级
x3 <- data$每天的学习时间
x4 <- data$每天的体育锻炼时间
x5 <- data$每天使用手机时间
x6 <- data$学业压力
x7 <- data$宿舍条件
x8 <- data$咖啡因摄入
y <- data$失眠指数

# 初始回归模型
init_model <- lm(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8, data = data)

# 检验多重共线性
vif(init_model)

# 设置 trace=2 打印详细信息（包括 AIC、变量剔除/加入）
final_model <- stepAIC(init_model, direction = "both", trace = 2)

# 最终模型摘要
summary(final_model)

# 提取系数值
coefficients <- coef(final_model)
coefficients

# 正态性检验
jarque.bera.test(residuals(final_model))

# 残差分析
# 绘制残差图
par(mfrow = c(2, 2))
plot(final_model)

# 进一步残差诊断
residualPlots(final_model)

# 检验非线性关系 - 若p值显著，则可能存在非线性
residualPlots(final_model, 
              terms = ~.,       # 所有变量
              tests = TRUE,    # 显示假设检验
              fitted = FALSE)   # 不显示拟合值图

# 基于olsrr方法进行多元回归分析
# 运行双向逐步回归
final_result <- ols_step_both_p(
  init_model,
  pent = 0.05,     # 变量进入的 p 值阈值
  prem = 0.1,     # 变量剔除的 p 值阈值
  details = TRUE
)

# 最终模型
final_model <- final_result$model
summary(final_model)