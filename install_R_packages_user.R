# Install required R packages for the pipeline (user library)

# Use user library
dir.create(Sys.getenv("R_LIBS_USER"), showWarnings = FALSE, recursive = TRUE)
.libPaths(Sys.getenv("R_LIBS_USER"))

packages <- c(
  "tidyverse",
  "yaml",
  "readxl",
  "e1071",
  "caret",
  "yardstick",
  "doParallel",
  "rstatix",
  "rmcorr",
  "afex",
  "emmeans",
  "effectsize",
  "ARTool",
  "performance",
  "stringr",
  "ggplot2",
  "glue",
  "rlang",
  "broom",
  "purrr"
)

repos <- "https://cran.r-project.org/"
options(repos = c(CRAN = repos))

installed <- rownames(installed.packages())
to_install <- setdiff(packages, installed)

if (length(to_install) > 0) {
  cat("Installing packages:", paste(to_install, collapse=", "), "\n")
  install.packages(to_install, lib = Sys.getenv("R_LIBS_USER"))
  cat("Installation complete!\n")
} else {
  cat("All R packages already installed.\n")
}
