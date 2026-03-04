## =========================================================
##  6-model evaluation pipeline:
##  - Brier score
##  - DCA (rmda) : 6 models in one plot (train), one plot (val)
##  - Calibration curve (ggplot2) : 6 models in one plot (train), one plot (val)
##  - Threshold optimization (multiple criteria) + export tables
##  - Export PDF + PNG (600 dpi)
## =========================================================

## -------------------------
## 0) Packages
## -------------------------
pkgs <- c("tidyverse", "rmda")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if(length(to_install) > 0) install.packages(to_install, repos = "https://cloud.r-project.org")
library(tidyverse)
library(rmda)

## -------------------------
## 1) I/O paths
## -------------------------
train_path <- "all_train_results.csv"
val_path   <- "all_val_results.csv"

out_dir <- "model_eval_outputs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## -------------------------
## 2) Read data
## -------------------------
train <- read.csv(train_path, stringsAsFactors = FALSE)
val   <- read.csv(val_path, stringsAsFactors = FALSE)

## Your files look like:
## train: y_train + 6 probability columns
## val:   y_val   + 6 probability columns
outcome_train <- "y_train"
outcome_val   <- "y_val"

## The 6 model probability columns (edit here if your names change)
model_cols <- c("rf_proba", "lr_proba", "gbdt_proba", "ada_proba", "xgb_proba", "lgbm_proba")

## Pretty names for plots/tables
model_map <- c(
  rf_proba   = "Random Forest",
  lr_proba   = "Logistic Regression",
  gbdt_proba = "GBDT",
  ada_proba  = "AdaBoost",
  xgb_proba  = "XGBoost",
  lgbm_proba = "LightGBM"
)

## Basic sanity checks
stopifnot(outcome_train %in% names(train), outcome_val %in% names(val))
stopifnot(all(model_cols %in% names(train)), all(model_cols %in% names(val)))

## Ensure outcomes are 0/1 numeric
train[[outcome_train]] <- as.integer(train[[outcome_train]])
val[[outcome_val]]     <- as.integer(val[[outcome_val]])

## Clamp probabilities to avoid logit issues / numeric edge cases
clamp_prob <- function(p, eps = 1e-15) pmin(pmax(p, eps), 1 - eps)
for(m in model_cols){
  train[[m]] <- clamp_prob(train[[m]])
  val[[m]]   <- clamp_prob(val[[m]])
}

## -------------------------
## 3) Metrics: Brier score
## -------------------------
brier_score <- function(y, p){
  mean((p - y)^2)
}

brier_tbl <- bind_rows(
  tibble(
    dataset = "Train",
    model   = model_cols,
    brier   = sapply(model_cols, \(m) brier_score(train[[outcome_train]], train[[m]]))
  ),
  tibble(
    dataset = "Validation",
    model   = model_cols,
    brier   = sapply(model_cols, \(m) brier_score(val[[outcome_val]], val[[m]]))
  )
) %>%
  mutate(model_name = unname(model_map[model])) %>%
  select(dataset, model, model_name, brier) %>%
  arrange(dataset, brier)

write.csv(brier_tbl, file = file.path(out_dir, "brier_scores.csv"), row.names = FALSE)

## -------------------------
## 4) Calibration curve (ggplot2)
##    - binning (default: deciles)
##    - plot observed vs predicted (6 models on one figure)
## -------------------------
calibration_bins <- function(df, y_col, p_col, n_bins = 10){
  df2 <- df %>%
    transmute(
      y = .data[[y_col]],
      p = .data[[p_col]]
    ) %>%
    mutate(bin = ntile(p, n_bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(p),
      obs_rate  = mean(y),
      n         = n(),
      .groups   = "drop"
    )
  df2
}

make_calibration_long <- function(df, y_col, model_cols, n_bins = 10){
  purrr::map_dfr(model_cols, function(m){
    calibration_bins(df, y_col = y_col, p_col = m, n_bins = n_bins) %>%
      mutate(model = m, model_name = unname(model_map[m]))
  })
}

plot_calibration <- function(df, y_col, dataset_label, n_bins = 10){
  cal_long <- make_calibration_long(df, y_col = y_col, model_cols = model_cols, n_bins = n_bins)
  
  ggplot(cal_long, aes(x = mean_pred, y = obs_rate, color = model_name)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", linewidth = 0.6) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
      title = paste0("Calibration Curve (", dataset_label, ")"),
      x = "Mean predicted probability",
      y = "Observed event rate",
      color = "Model"
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.position = "right"
    )
}

cal_train_plot <- plot_calibration(train, y_col = outcome_train, dataset_label = "Train", n_bins = 10)
cal_val_plot   <- plot_calibration(val,   y_col = outcome_val,   dataset_label = "Validation", n_bins = 10)

## Export calibration plots
ggsave(file.path(out_dir, "calibration_train.pdf"), cal_train_plot, width = 7.5, height = 6)
ggsave(file.path(out_dir, "calibration_train.png"), cal_train_plot, width = 7.5, height = 6, dpi = 600)
ggsave(file.path(out_dir, "calibration_validation.pdf"), cal_val_plot, width = 7.5, height = 6)
ggsave(file.path(out_dir, "calibration_validation.png"), cal_val_plot, width = 7.5, height = 6, dpi = 600)

## -------------------------
## 5) Decision Curve Analysis (rmda)
##    - IMPORTANT: use rmda package as requested
##    - predictors are already "fitted risks" (probabilities)

## 强制把预测概率列转成 numeric（很多 CSV 会读成字符）
for(m in model_cols){
  train[[m]] <- as.numeric(train[[m]])
  val[[m]]   <- as.numeric(val[[m]])
}

## 去掉缺失
train <- train %>% tidyr::drop_na(all_of(c(outcome_train, model_cols)))
val   <- val   %>% tidyr::drop_na(all_of(c(outcome_val, model_cols)))

## outcome 确保 0/1
train[[outcome_train]] <- as.integer(train[[outcome_train]])
val[[outcome_val]]     <- as.integer(val[[outcome_val]])

## clamp
clamp_prob <- function(p, eps = 1e-15) pmin(pmax(p, eps), 1 - eps)
for(m in model_cols){
  train[[m]] <- clamp_prob(train[[m]])
  val[[m]]   <- clamp_prob(val[[m]])
}
## -------------------------
run_dca <- function(df, y_col, model_cols,
                    thresholds = seq(0.01, 0.99, by = 0.01)){
  dca_list <- purrr::map(model_cols, function(m){
    form <- as.formula(paste0(y_col, " ~ ", m))
    rmda::decision_curve(
      formula = form,
      data = df,
      family = binomial(link = "logit"),
      thresholds = thresholds,
      fitted.risk = TRUE,
      study.design = "cohort"
    )
  })
  names(dca_list) <- unname(model_map[model_cols])
  dca_list
}

dca_train <- run_dca(train, y_col = outcome_train, model_cols = model_cols,
                     thresholds = seq(0, 1, by = 0.01))
dca_val   <- run_dca(val,   y_col = outcome_val,   model_cols = model_cols,
                     thresholds = seq(0, 1, by = 0.01))

## rmda::plot_decision_curve returns a ggplot object (can be saved by ggsave)
plot_dca <- function(dca_list, dataset_label){
  rmda::plot_decision_curve(
    x = dca_list,
    curve.names = names(dca_list),
    confidence.intervals = FALSE,
    standardize = FALSE
  ) +
    ggtitle(paste0("Decision Curve Analysis (", dataset_label, ")")) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.position = "right"
    )
}

plot_dca(dca_train, "Train")
plot_dca(dca_val,   "Validation")

## -------------------------
## =========================================================
## Threshold optimization + Bootstrap (1000) 95% CI
## =========================================================
## Threshold optimization + Bootstrap(1000) 95%CI
## - Avoid name collision in unnest_wider()
## - Format: 0.812 (0.765–0.854), 3 decimals
## =========================================================

library(tidyverse)

## ---------- core metrics at threshold ----------
conf_metrics <- function(y, p, thr){
  pred <- ifelse(p >= thr, 1L, 0L)
  
  TP <- sum(pred == 1L & y == 1L)
  TN <- sum(pred == 0L & y == 0L)
  FP <- sum(pred == 1L & y == 0L)
  FN <- sum(pred == 0L & y == 1L)
  
  sens <- ifelse((TP + FN) == 0, NA_real_, TP / (TP + FN))
  spec <- ifelse((TN + FP) == 0, NA_real_, TN / (TN + FP))
  ppv  <- ifelse((TP + FP) == 0, NA_real_, TP / (TP + FP))
  npv  <- ifelse((TN + FN) == 0, NA_real_, TN / (TN + FN))
  acc  <- (TP + TN) / (TP + TN + FP + FN)
  youden <- sens + spec - 1
  f1 <- ifelse(is.na(ppv) | is.na(sens) | (ppv + sens) == 0, NA_real_, 2 * ppv * sens / (ppv + sens))
  dist01 <- sqrt((1 - spec)^2 + (1 - sens)^2)
  
  tibble(
    threshold = thr,
    TP = TP, TN = TN, FP = FP, FN = FN,
    sensitivity = sens,
    specificity = spec,
    PPV = ppv,
    NPV = npv,
    accuracy = acc,
    F1 = f1,
    youden = youden,
    dist01 = dist01
  )
}

threshold_grid_eval <- function(df, y_col, p_col, thresholds = seq(0.01, 0.99, by = 0.01)){
  y <- df[[y_col]]
  p <- df[[p_col]]
  purrr::map_dfr(thresholds, \(t) conf_metrics(y, p, t))
}

## ---------- pick best ----------
pick_best <- function(tbl, criterion){
  if(criterion == "max_youden"){
    tbl %>% filter(!is.na(youden)) %>% slice_max(youden, n = 1, with_ties = FALSE)
  } else if(criterion == "min_dist01"){
    tbl %>% filter(!is.na(dist01)) %>% slice_min(dist01, n = 1, with_ties = FALSE)
  } else if(criterion == "max_f1"){
    tbl %>% filter(!is.na(F1)) %>% slice_max(F1, n = 1, with_ties = FALSE)
  } else if(criterion == "max_accuracy"){
    tbl %>% slice_max(accuracy, n = 1, with_ties = FALSE)
  } else if(criterion == "max_sens_spec_ge_0.80"){
    cand <- tbl %>% filter(!is.na(specificity), specificity >= 0.80, !is.na(sensitivity))
    if(nrow(cand) == 0) return(tibble())
    cand %>% slice_max(sensitivity, n = 1, with_ties = FALSE)
  } else if(criterion == "max_spec_sens_ge_0.80"){
    cand <- tbl %>% filter(!is.na(sensitivity), sensitivity >= 0.80, !is.na(specificity))
    if(nrow(cand) == 0) return(tibble())
    cand %>% slice_max(specificity, n = 1, with_ties = FALSE)
  } else if(criterion == "max_sensitivity"){
    tbl %>% filter(!is.na(sensitivity)) %>% slice_max(sensitivity, n = 1, with_ties = FALSE)
  } else if(criterion == "max_specificity"){
    tbl %>% filter(!is.na(specificity)) %>% slice_max(specificity, n = 1, with_ties = FALSE)
  } else {
    stop("Unknown criterion: ", criterion)
  }
}

## ---------- bootstrap CI: return CI ONLY (avoid duplicate names) ----------
boot_ci_only <- function(y, p, thr, n_boot = 1000, conf = 0.95, seed = 2026){
  set.seed(seed)
  
  n <- length(y)
  alpha <- (1 - conf) / 2
  
  metric_names <- c("sensitivity", "specificity", "PPV", "NPV", "accuracy", "F1", "youden")
  
  boot_mat <- sapply(seq_len(n_boot), function(i){
    idx <- sample.int(n, size = n, replace = TRUE)
    m <- conf_metrics(y[idx], p[idx], thr) %>%
      select(all_of(metric_names))
    as.numeric(m[1, ])
  })
  
  ## sapply returns matrix with rownames = metric_names
  if(is.null(dim(boot_mat))){
    # edge case: n_boot=1
    boot_mat <- matrix(boot_mat, nrow = length(metric_names))
    rownames(boot_mat) <- metric_names
  }
  
  ci_low  <- apply(boot_mat, 1, quantile, probs = alpha,     na.rm = TRUE)
  ci_high <- apply(boot_mat, 1, quantile, probs = 1 - alpha, na.rm = TRUE)
  
  ## return CI columns only
  out <- tibble(
    sensitivity_CI_low  = unname(ci_low["sensitivity"]),
    sensitivity_CI_high = unname(ci_high["sensitivity"]),
    specificity_CI_low  = unname(ci_low["specificity"]),
    specificity_CI_high = unname(ci_high["specificity"]),
    PPV_CI_low          = unname(ci_low["PPV"]),
    PPV_CI_high         = unname(ci_high["PPV"]),
    NPV_CI_low          = unname(ci_low["NPV"]),
    NPV_CI_high         = unname(ci_high["NPV"]),
    accuracy_CI_low     = unname(ci_low["accuracy"]),
    accuracy_CI_high    = unname(ci_high["accuracy"]),
    F1_CI_low           = unname(ci_low["F1"]),
    F1_CI_high          = unname(ci_high["F1"]),
    youden_CI_low       = unname(ci_low["youden"]),
    youden_CI_high      = unname(ci_high["youden"])
  )
  out
}

## ---------- formatting ----------
fmt_ci <- function(est, low, high, digits = 3){
  ifelse(is.na(est) | is.na(low) | is.na(high),
         NA_character_,
         sprintf(paste0("%.", digits, "f (%.", digits, "f\u2013%.", digits, "f)"), est, low, high))
}

add_formatted_ci_columns <- function(tbl, digits = 3){
  tbl %>%
    mutate(
      sensitivity_fmt = fmt_ci(sensitivity, sensitivity_CI_low, sensitivity_CI_high, digits),
      specificity_fmt = fmt_ci(specificity, specificity_CI_low, specificity_CI_high, digits),
      PPV_fmt         = fmt_ci(PPV, PPV_CI_low, PPV_CI_high, digits),
      NPV_fmt         = fmt_ci(NPV, NPV_CI_low, NPV_CI_high, digits),
      accuracy_fmt    = fmt_ci(accuracy, accuracy_CI_low, accuracy_CI_high, digits),
      F1_fmt          = fmt_ci(F1, F1_CI_low, F1_CI_high, digits),
      youden_fmt      = fmt_ci(youden, youden_CI_low, youden_CI_high, digits),
      threshold       = round(threshold, digits),
      sensitivity     = round(sensitivity, digits),
      specificity     = round(specificity, digits),
      PPV             = round(PPV, digits),
      NPV             = round(NPV, digits),
      accuracy        = round(accuracy, digits),
      F1              = round(F1, digits),
      youden          = round(youden, digits)
    )
}

## ---------- main optimize ----------
optimize_thresholds <- function(df, y_col, dataset_label,
                                model_cols, model_map,
                                thresholds = seq(0.01, 0.99, by = 0.01),
                                out_dir = ".",
                                export_full = FALSE,
                                add_boot_ci = TRUE,
                                n_boot = 1000, conf = 0.95, seed = 2026,
                                digits = 3){
  
  criteria <- c(
    "max_youden",
    "min_dist01",
    "max_f1",
    "max_accuracy",
    "max_sens_spec_ge_0.80",
    "max_spec_sens_ge_0.80",
    "max_sensitivity",
    "max_specificity"
  )
  
  full_tbl <- purrr::map_dfr(model_cols, function(m){
    threshold_grid_eval(df, y_col = y_col, p_col = m, thresholds = thresholds) %>%
      mutate(
        dataset = dataset_label,
        model = m,
        model_name = unname(model_map[m])
      )
  })
  
  best_tbl <- purrr::map_dfr(model_cols, function(m){
    t1 <- full_tbl %>% filter(model == m)
    purrr::map_dfr(criteria, function(cr){
      best <- pick_best(t1, cr)
      if(nrow(best) == 0){
        return(tibble(
          dataset = dataset_label,
          model = m,
          model_name = unname(model_map[m]),
          criterion = cr,
          threshold = NA_real_
        ))
      }
      best %>%
        mutate(
          dataset = dataset_label,
          model = m,
          model_name = unname(model_map[m]),
          criterion = cr
        ) %>%
        select(dataset, model, model_name, criterion, everything())
    })
  })
  
  ## bootstrap CI -> ONLY CI cols, safe to unnest
  if(add_boot_ci){
    y <- df[[y_col]]
    
    best_tbl <- best_tbl %>%
      rowwise() %>%
      mutate(
        .ci = list({
          if(is.na(threshold)){
            tibble(
              sensitivity_CI_low = NA_real_, sensitivity_CI_high = NA_real_,
              specificity_CI_low = NA_real_, specificity_CI_high = NA_real_,
              PPV_CI_low = NA_real_, PPV_CI_high = NA_real_,
              NPV_CI_low = NA_real_, NPV_CI_high = NA_real_,
              accuracy_CI_low = NA_real_, accuracy_CI_high = NA_real_,
              F1_CI_low = NA_real_, F1_CI_high = NA_real_,
              youden_CI_low = NA_real_, youden_CI_high = NA_real_
            )
          } else {
            p <- df[[model]]
            boot_ci_only(y = y, p = p, thr = threshold, n_boot = n_boot, conf = conf, seed = seed)
          }
        })
      ) %>%
      ungroup() %>%
      tidyr::unnest_wider(.ci)
  }
  
  best_tbl_fmt <- add_formatted_ci_columns(best_tbl, digits = digits) %>%
    select(
      dataset, model_name, model, criterion, threshold,
      sensitivity_fmt, specificity_fmt, PPV_fmt, NPV_fmt, accuracy_fmt, F1_fmt, youden_fmt,
      TP, TN, FP, FN,
      sensitivity, sensitivity_CI_low, sensitivity_CI_high,
      specificity, specificity_CI_low, specificity_CI_high,
      PPV, PPV_CI_low, PPV_CI_high,
      NPV, NPV_CI_low, NPV_CI_high,
      accuracy, accuracy_CI_low, accuracy_CI_high,
      F1, F1_CI_low, F1_CI_high,
      youden, youden_CI_low, youden_CI_high
    )
  
  ## export
  write.csv(best_tbl,
            file = file.path(out_dir, paste0("threshold_optimized_", tolower(dataset_label), "_best_by_criteria_raw.csv")),
            row.names = FALSE)
  
  write.csv(best_tbl_fmt,
            file = file.path(out_dir, paste0("threshold_optimized_", tolower(dataset_label), "_best_by_criteria_fmt.csv")),
            row.names = FALSE)
  
  if(export_full){
    write.csv(full_tbl,
              file = file.path(out_dir, paste0("threshold_optimized_", tolower(dataset_label), "_FULL_grid.csv")),
              row.names = FALSE)
  }
  
  list(full = full_tbl, best_raw = best_tbl, best_fmt = best_tbl_fmt)
}


th_grid <- seq(0.01, 0.99, by = 0.01)

thr_train <- optimize_thresholds(
  train, y_col = outcome_train, dataset_label = "Train",
  model_cols = model_cols, model_map = model_map,
  thresholds = th_grid,
  out_dir = out_dir,
  export_full = FALSE,
  add_boot_ci = TRUE, n_boot = 1000, conf = 0.95, seed = 2026,
  digits = 3
)

thr_val <- optimize_thresholds(
  val, y_col = outcome_val, dataset_label = "Validation",
  model_cols = model_cols, model_map = model_map,
  thresholds = th_grid,
  out_dir = out_dir,
  export_full = FALSE,
  add_boot_ci = TRUE, n_boot = 1000, conf = 0.95, seed = 2026,
  digits = 3
)

best_all_raw <- bind_rows(thr_train$best_raw, thr_val$best_raw)
best_all_fmt <- bind_rows(thr_train$best_fmt, thr_val$best_fmt)

write.csv(best_all_raw, file = file.path(out_dir, "threshold_optimized_best_by_criteria_ALL_raw.csv"), row.names = FALSE)
write.csv(best_all_fmt, file = file.path(out_dir, "threshold_optimized_best_by_criteria_ALL_fmt.csv"), row.names = FALSE)


## =========================================================
## DeLong test: Compare GBDT vs other models (Train & Val)
## Output: CSV (separated by dataset)
## =========================================================

if(!requireNamespace("pROC", quietly = TRUE)){
  install.packages("pROC", repos = "https://cloud.r-project.org")
}
library(pROC)
library(dplyr)

## 基准模型列（GBDT）
gbdt_col <- "gbdt_proba"
stopifnot(gbdt_col %in% model_cols)

## DeLong 检验函数：GBDT vs others
delong_vs_gbdt <- function(df, y_col, model_cols, base_col = "gbdt_proba",
                           model_map = NULL, dataset_label = "Train"){
  
  y <- df[[y_col]]
  y <- as.integer(y)
  
  ## pROC 要求二分类，这里确保只含 0/1
  stopifnot(all(y %in% c(0L, 1L)))
  
  roc_base <- pROC::roc(response = y, predictor = df[[base_col]],
                        ci = FALSE, quiet = TRUE, direction = "<")
  
  others <- setdiff(model_cols, base_col)
  
  res <- lapply(others, function(m){
    roc_m <- pROC::roc(response = y, predictor = df[[m]],
                       ci = FALSE, quiet = TRUE, direction = "<")
    
    ## DeLong test
    rt <- pROC::roc.test(roc_base, roc_m, method = "delong")
    
    tibble(
      dataset = dataset_label,
      base_model = if(!is.null(model_map)) unname(model_map[base_col]) else base_col,
      compare_model = if(!is.null(model_map)) unname(model_map[m]) else m,
      auc_base = as.numeric(pROC::auc(roc_base)),
      auc_compare = as.numeric(pROC::auc(roc_m)),
      auc_diff = auc_base - auc_compare,
      z = unname(rt$statistic),
      p_value = unname(rt$p.value)
    )
  })
  
  bind_rows(res) %>%
    mutate(
      auc_base = round(auc_base, 3),
      auc_compare = round(auc_compare, 3),
      auc_diff = round(auc_diff, 3),
      z = round(z, 3),
      p_value = signif(p_value, 3)
    )
}

## 训练集
delong_train <- delong_vs_gbdt(
  df = train, y_col = outcome_train,
  model_cols = model_cols, base_col = gbdt_col,
  model_map = model_map, dataset_label = "Train"
)
write.csv(delong_train, file = file.path(out_dir, "delong_gbdt_vs_others_train.csv"), row.names = FALSE)

## 验证集
delong_val <- delong_vs_gbdt(
  df = val, y_col = outcome_val,
  model_cols = model_cols, base_col = gbdt_col,
  model_map = model_map, dataset_label = "Validation"
)
write.csv(delong_val, file = file.path(out_dir, "delong_gbdt_vs_others_validation.csv"), row.names = FALSE)