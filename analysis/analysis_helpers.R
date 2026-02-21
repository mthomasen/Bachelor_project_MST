
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(scales)
  library(fs)
  library(viridisLite)
  library(knitr)
})

ensure_dir <- function(path) {
  if (!dir_exists(path)) dir_create(path, recurse = TRUE)
  invisible(path)
}

analysis_fig_dir <- function(sweep_dir) {
  out <- file.path("analysis", "figures", basename(sweep_dir))
  ensure_dir(out)
  out
}

save_fig <- function(p, filename, sweep_dir, width = 7, height = 5, dpi = 300) {
  fig_dir <- analysis_fig_dir(sweep_dir)
  out_path <- file.path(fig_dir, filename)
  ggsave(out_path, plot = p, width = width, height = height, dpi = dpi)
  message("Saved figure: ", out_path)
  invisible(out_path)
}

theme_thesis <- function() {
  theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold"),
      legend.position = "right"
    )
}

# convenience for consistent heatmaps
heatmap_tile <- function(df, x, y, fill, title = NULL, xlab = NULL, ylab = NULL) {
  ggplot(df, aes(x = {{ x }}, y = {{ y }}, fill = {{ fill }})) +
    geom_tile() +
    scale_fill_viridis_c(option = "C", na.value = "grey90") +
    labs(title = title, x = xlab, y = ylab, fill = deparse(substitute(fill))) +
    theme_thesis()
}
