# analysis/dataload.R

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(magrittr)
  library(fs)
})

stop_if_missing <- function(df, cols, df_name = "data") {
  missing <- setdiff(cols, names(df))
  if (length(missing) > 0) {
    stop(
      sprintf(
        "%s is missing required columns: %s",
        df_name,
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  invisible(df)
}

message_glue <- function(...) {
  message(paste0(...))
}

find_latest_sweep_dir <- function(outputs_dir = "final_outputs", pattern = "dyad_sweep") {
  if (!dir_exists(outputs_dir)) {
    stop(sprintf("outputs_dir not found: %s", outputs_dir), call. = FALSE)
  }
  
  candidates <- dir_ls(outputs_dir, type = "directory") %>%
    keep(~ str_detect(path_file(.x), fixed(pattern)))
  
  if (length(candidates) == 0) {
    stop(
      sprintf("No sweep folders found in '%s' matching pattern '%s'", outputs_dir, pattern),
      call. = FALSE
    )
  }
  
  candidates %>% sort() %>% tail(1) %>% as.character()
  
}


resolve_sweep_dir <- function(
    sweep_dir = NULL,
    outputs_dir = "final_outputs",
    pattern = "dyad_sweep"
) {
  if (!is.null(sweep_dir)) {
    if (!dir_exists(sweep_dir)) {
      stop(sprintf("Provided sweep_dir not found: %s", sweep_dir), call. = FALSE)
    }
    return(as.character(sweep_dir))
  }
  find_latest_sweep_dir(outputs_dir = outputs_dir, pattern = pattern)
}

load_conditions <- function(sweep_dir, file_name, show_path = TRUE) {
  path <- path(sweep_dir, file_name)
  if (!file_exists(path)) {
    stop(sprintf("Missing file: %s", path), call. = FALSE)
  }
  if (show_path) message_glue("Reading: ", path)
  
  df <- readr::read_csv(path, show_col_types = FALSE)
  
  if (nrow(df) == 0) {
    stop(sprintf("File exists but is empty: %s", path), call. = FALSE)
  }
  df
}

load_dyad_conditions <- function(
    sweep_dir = NULL,
    outputs_dir = "final_outputs",
    pattern = "dyad_sweep"
) {
  sweep_dir <- resolve_sweep_dir(sweep_dir, outputs_dir, pattern)
  df <- load_conditions(sweep_dir, "dyad_sweep_conditions.csv")
  
  stop_if_missing(
    df,
    cols = c("k", "delta_omega", "n_runs", "r_mean", "plv_mean", "anti_mean"),
    df_name = "dyad conditions"
  )
  
  list(
    sweep_dir = sweep_dir,
    conditions = df
  )
}

load_triad_conditions <- function(
    sweep_dir = NULL,
    outputs_dir = "final_outputs",
    pattern = "triad_sweep"
) {
  sweep_dir <- resolve_sweep_dir(sweep_dir, outputs_dir, pattern)
  df <- load_conditions(sweep_dir, "triad_sweep_conditions.csv")
  
  stop_if_missing(
    df,
    cols = c(
      "preset", "delta_omega_tri", "k_strong", "k_weak", "n_runs",
      "r_mean", "plv_pairmean_mean", "coalition_mean", "leader_dom_mean"
    ),
    df_name = "triad conditions"
  )
  
  list(
    sweep_dir = sweep_dir,
    conditions = df
  )
}

load_classroom_conditions <- function(
    sweep_dir = NULL,
    outputs_dir = "final_outputs",
    pattern = "classroom_sweep"
) {
  sweep_dir <- resolve_sweep_dir(sweep_dir, outputs_dir, pattern)
  df <- load_conditions(sweep_dir, "classroom_sweep_conditions.csv")
  
  stop_if_missing(
    df,
    cols = c(
      "k_ts", "k_st", "k_ss", "n_runs",
      "r_mean", "ts_plv_mean", "ss_plv_mean", "teacher_dom_mean"
    ),
    df_name = "classroom conditions"
  )
  
  list(
    sweep_dir = sweep_dir,
    conditions = df
  )
}

print_basic_checks <- function(df, keys) {
  message_glue("Rows: ", nrow(df))
  message_glue("Unique keys: ", df %>% distinct(across(all_of(keys))) %>% nrow())
  
  if ("n_runs" %in% names(df)) {
    message_glue("n_runs range: ", paste(range(df$n_runs, na.rm = TRUE), collapse = " to "))
  }
  invisible(df)
}