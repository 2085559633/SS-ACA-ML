# install.packages(c("ggpubr","corrplot",
#                    "glmnet","caret","CBCgrps",
#                    "tidyverse","rms"))
library(corrplot) 
library(caret) 
library(CBCgrps)
library(nortest)
library(tidyverse)
library(pROC)
library(openxlsx)
library(Classprob)
library(mice)
library(zoo)
library(VIM)
head(tidyr, 20)
tidyr2 <- tidyr %>%
  mutate(
    all_text = str_c(
      coalesce(zhusu, ""),
      coalesce(zhusu1, ""),
      coalesce(other, ""),
      coalesce(other1, ""),
      sep = "；"
    )
  )

# 2) 定义关键词/正则（可按需要继续扩展同义写法）
kw <- list(
  kougan   = "(口干|口咸|口渴)",
  yangan   = "(眼干)",
  `il d`     = "(间质性肺(炎|病)|间质性肺炎|ILD)",        # ILD 也算
  leinuo   = "(雷诺|Raynaud)",
  wbc_low  = "(白细胞(减少|减低|降低)|白细胞低)",
  plt_low  = "(血小板(减少|减低|降低)|血小板低)",
  pbc      = "(胆汁性肝硬化|原发性胆汁性肝硬化|PBC)"
)

# 3) 生成二分类变量：命中=1，否则=0
tidyr_flag <- tidyr2 %>%
  mutate(
    dry_mouth = as.integer(str_detect(all_text, kw$kougan)),
    dry_eye   = as.integer(str_detect(all_text, kw$yangan)),
    ild       = as.integer(str_detect(all_text, kw$`il d`)),
    raynaud   = as.integer(str_detect(all_text, kw$leinuo)),
    wbc_low   = as.integer(str_detect(all_text, kw$wbc_low)),
    plt_low   = as.integer(str_detect(all_text, kw$plt_low)),
    pbc       = as.integer(str_detect(all_text, kw$pbc))
  ) %>%
  select(-all_text)  

write.csv(tidyr_flag,"tidyr_flag.csv")

library(readxl)
data <- read_excel("data.xlsx")
aggr(data)

# 计算每列的缺失值比例
missing_percentage <- colMeans(is.na(data)) * 100

# 选择缺失值比例小于等于20%的列
selected_columns <- names(missing_percentage[missing_percentage <= 30])
data <- data[,selected_columns]
# 缺失值插补
dataall <- mice(data, #数据集
                method = "rf", #采用随机森林插补
                m=5, # 5次插补
                printFlag = FALSE #不显示历史记录
)
data1 <- complete(dataall, action = 2)
aggr(data1)
write.csv(data1,"datamice.csv")
data <- data1
#设置因子型
library(janitor)
data <- clean_names(data)
colnames(data)[1] <- "Group"
ordata <- data
colnames(data)
data[,1:17] <- lapply(data[,1:17],as.factor)

##数据集划分####
set.seed(3456)
trainIndex <- createDataPartition(data$Group, p = .7, 
                                  list = FALSE, 
                                  times = 1)
train <- data[trainIndex,]
test  <- data[-trainIndex,]

write.xlsx(train, "train.xlsx",rowNames=F,colNames=T)
write.xlsx(test, "test.xlsx",rowNames=F,colNames=T)

train <- as.data.frame(train)#有时候需要
tab1 <-twogrps(train, gvar = "Group",ShowStatistic = T)
write.xlsx(tab1$Table, "tab1train.xlsx",rowNames=F,colNames=F)

test <- as.data.frame(test)#有时候需要
tab1 <-twogrps(test, gvar = "Group",ShowStatistic = T)
write.xlsx(tab1$Table, "tab1test.xlsx",rowNames=F,colNames=F)

###训练集验证集tab1####
test$group <- "test"
train$group <- "train"
all <- rbind(test,train)


all <- as.data.frame(all)#有时候需要
tab1 <-twogrps(all, gvar = "group",ShowStatistic = T)
write.xlsx(tab1$Table, "tab1testtrian.xlsx",rowNames=F,colNames=F)
tab1 <-twogrps(all, gvar = "Group",ShowStatistic = T)
write.xlsx(tab1$Table, "tab1.xlsx",rowNames=F,colNames=F)

#minmax标准化转换####
data2 = train%>%
  mutate_if(.predicate = is.numeric,
            .funs = min_max_scale)%>%
  as.data.frame()
#z转化矩阵
library(glmnet)
set.seed(123) #random number generator
x <- data.matrix(data2[, -1])
y <- data2[, 1]
y<-as.numeric(unlist(y))
###lasso####
lasso <- glmnet(x, y, family = "binomial",nlambda = 1000, alpha = 1)
print(lasso)
plot(lasso, xvar = "lambda", label = TRUE,lwd=2)
pdf("LASSO1路径系数图.pdf",width = 5,height = 4)
plot(lasso, xvar = "lambda", label = TRUE,lwd=2)
dev.off()

tmp <- as_tibble(as.matrix(coef(lasso)), rownames = "coef") %>% 
  pivot_longer(cols = -coef, 
               names_to = "variable", 
               names_transform = list(variable = parse_number), 
               values_to = "value") %>% 
  group_by(variable) %>% 
  mutate(lambda = lasso$lambda[variable + 1], 
         norm = sum(if_else(coef == "(Intercept)", 0, abs(value))))

tmp[tmp=='(Intercept)'] <- NA
tmp1=na.omit(tmp)
plasso <- ggplot(tmp1, aes(log(lambda),value,color=coef,group=coef))+
  geom_line(linewidth=1.2)+
  labs(x="Log Lambda",y="Coefficients")+
  theme_classic()
plasso
ggsave("LASSO1(路径系数图2).pdf", plot =plasso, width = 8.3, height = 5.7)

#交叉验证
lasso.cv = cv.glmnet(x, y,alpha = 1,nfolds =5,family="binomial")
plot(lasso.cv)
pdf("LASSO2交叉验证曲线.pdf",width = 5,height = 4)
plot(lasso.cv)
dev.off()
sink("lasso筛选结果.txt")
cat(
  "以下为LASSO筛选的结果相关参数，使用glmnet包完成。\n",
  "请最大化浏览窗口查看。对于此技术不了解的可自行学习。\n",
  "=====================================\n",
  "①以下是左边那条虚线相关参数，即lambda值可以使得MSE最小\n",
  "=====================================\n"
)

print(paste("lambda.min =", lasso.cv$lambda.min)) # minimum
coef(lasso.cv, s = "lambda.min")
cat(
  "=====================================\n",
  "②以下是右边那条虚线相关参数，此时的lambda值可以使得MSE在最小MSE的1倍标准误区间内，
  但是同时可以使模型的复杂度降低\n",
  "=====================================\n"
)
print(paste("lambda.1se =", lasso.cv$lambda.1se)) # one standard error away
coef(lasso.cv, s = "lambda.1se")
sink()
#提取变量
Coefficients <- coef(lasso.cv, s = lasso.cv$lambda.min)
Active.Index <- which(Coefficients != 0)
Active.Coefficients <- Coefficients[Active.Index]
Active.Index
Active.Coefficients
lassonames <- row.names(Coefficients)[Active.Index]
reformulate(lassonames)
lassonames[1] <- "Group"
lassonames

train1 <- train[,lassonames]
test1 <- test[,lassonames]

write.xlsx(train1, "train1.xlsx",rowNames=F,colNames=T)
write.xlsx(test1, "test1.xlsx",rowNames=F,colNames=T)


###ROC####
rocdata <- ordata
rocdata$Group <- as.factor(rocdata$Group)
reformulate(colnames(data),"y")
roc.list <- roc(Group ~ ssa_ro + ssa_ro52 + ssb + jo1 + u1rnp + scl70 + ds_dna + 
                  amam2 + sex + dry_mouth + dry_eye + ild + raynaud + wbc_low + 
                  plt_low + pbc + age + weight + height + esspri + wbc + ne + 
                  ly + rbc + hgb + plt + esr + alt + ast + tp + glo + alb + 
                  tbil + ibil + crea + ggt + alp + crp + ig_a + ig_m + ig_g + 
                  c3 + c4 + essdai, data = rocdata)

g.list <- ggroc(roc.list,alpha = 0.8, linetype = 1, size = 0.8)
rocall <- g.list+theme_classic()+ 
  theme (axis.text = element_text (size = 15))+
  theme(axis.title.x=element_text(vjust=2, size=15,face = "plain"))+
  theme(axis.title.y=element_text(vjust=2, size=15,face = "plain"))
rocall
ggsave("ROCall.pdf", plot =rocall, width = 8.2, height = 6.7)

#提取AUC####
auc <- sapply(roc.list,"[",9)
auc <- as.data.frame(auc)
auc <- t(auc)
auc <- as.data.frame(auc)
AUC <- arrange(auc,desc(V1),by_group=FALSE)
write.csv(AUC,"aucall.csv")

##根据数值大小而出现渐变颜色的柱状图####

library(ggplot2)
library(viridis)
signif(AUC$V1,digit=3) 
AUC$X <- rownames(AUC)
AUC$X1 <- gsub('.{4}$', '', AUC$X)

aucbar <- ggplot(AUC, aes(x = V1, y = reorder(X1, V1),fill=V1)) +
  geom_col(width = 0.7) +
  scale_fill_viridis(begin = 0, end = 0.85, option = "D") +
  labs(x = "AUC") +
  ylab("")+
  geom_text(aes(label = signif(V1,digit=3) ),
            size=2.5,vjust=0.3,hjust=-0.2,color="#2878B5")+
  scale_x_continuous(limits = c(0, 0.8))+
  theme_classic()+theme(legend.position = 'none')
aucbar
###长款图
ggsave("AUCall长.pdf", plot =aucbar, width = 3.8, height = 7.3)
###宽图
ggsave("AUCall宽.pdf", plot =aucbar, width = 7, height = 7)


###相关性共线性分析：####
numdata = ordata[,-1]
colnames(numdata)
#相关性矩阵
M<-cor(numdata)
M
write.csv(M,"相关性分析r值.csv")
#相关性的显著性检验
testRes<-cor.mtest(numdata,conf.level = 0.95)
testRes
write.csv(testRes,"相关性分析p值.csv")

par(mfrow=c(2,3))
#相关性热图####
corrplot(M,tl.col = "black",method ='circle')
dev.off()
pdf("相关性热图.pdf",width = 12,height = 12)
corrplot(M,tl.col = "black",method ='circle')
corrplot(
  M,
  method='color',
  type = 'upper',
  add = T ,
  tl.pos = "n",
  cl.pos = "n",
  diag = F,
  p.mat = testRes$p,
  sig.level = c(0.001,0.01,0.05),
  pch.cex = 1.5,
  insig = 'label_sig'
)
dev.off()



# =========================
# 0) 依赖
# =========================
library(corrplot)

# 如果你还没定义 cor.mtest，请确保你已经有该函数
# corrplot 官方示例里 cor.mtest 是自定义函数（不是包自带）
# 你原来能跑通说明你环境里已有，这里不重复定义。


# =========================
# 1) 变量名映射：把 numdata 的列名改成图中的“标准名”
# =========================
name_map <- c(
  ssa_ro    = "SSA/Ro60",
  ssa_ro52  = "SSA/Ro52",
  ssb       = "SSB",
  jo1       = "JO1",
  u1rnp     = "U1RNP",
  scl70     = "SCL70",
  ds_dna    = "DS-DNA",
  amam2     = "AMAM2",
  sex       = "SEX",
  dry_mouth = "DRY-MOUTH",
  dry_eye   = "DRY-EYE",
  ild       = "ILD",
  raynaud   = "Raynaud",
  wbc_low   = "WBC-LOW",
  plt_low   = "PLT-LOW",
  pbc       = "PBC",
  age       = "Age",
  weight    = "Weight",
  height    = "Hight",     # 若你想要标准拼写，把这里改成 "Height"
  esspri    = "ESSPRI",
  wbc       = "WBC",
  ne        = "NE",
  ly        = "LY",
  rbc       = "RBC",
  hgb       = "HGB",
  plt       = "PLT",
  esr       = "ESR",
  alt       = "ALT",
  ast       = "AST",
  tp        = "TP",
  glo       = "GLO",
  alb       = "ALB",
  tbil      = "TBIL",
  ibil      = "IBIL",
  crea      = "CREA",
  ggt       = "GGT",
  alp       = "ALP",
  crp       = "CRP",
  ig_a      = "IgA",
  ig_m      = "IgM",
  ig_g      = "IgG",
  c3        = "C3",
  c4        = "C4",
  essdai    = "ESSDAI"
)

# 检查：有没有列名没被映射到（防止漏变量导致 NA 列名）
missing <- setdiff(colnames(numdata), names(name_map))
if (length(missing) > 0) {
  stop("以下变量没有映射到图中标准名，请补充 name_map：\n  ",
       paste(missing, collapse = ", "))
}

# 执行重命名（保持原列顺序）
colnames(numdata) <- unname(name_map[colnames(numdata)])


# =========================
# 2) 相关性矩阵 + 显著性检验
# =========================
M <- cor(numdata, use = "pairwise.complete.obs", method = "pearson")

write.csv(M, "相关性分析r值.csv", row.names = TRUE)

testRes <- cor.mtest(numdata, conf.level = 0.95)

# 你原来 write.csv(testRes, ...) 是写不出去的（testRes 是 list）
# 一般写 p 值矩阵即可：
write.csv(testRes$p, "相关性分析p值.csv", row.names = TRUE)


# =========================
# 3) 相关性热图（PDF）
# =========================
pdf("相关性热图.pdf", width = 12, height = 12)

# 底图：circle
corrplot(
  M,
  method = "circle",
  tl.col = "black",
  tl.cex = 0.8,   # 标签多时建议调小
  tl.srt = 45     # 旋转一点更不挤
)

# 上层叠加：upper + 显著性标记
corrplot(
  M,
  method = "color",
  type = "upper",
  add = TRUE,
  tl.pos = "n",
  cl.pos = "n",
  diag = FALSE,
  p.mat = testRes$p,
  sig.level = c(0.001, 0.01, 0.05),
  pch.cex = 1.5,
  insig = "label_sig"
)

dev.off()

