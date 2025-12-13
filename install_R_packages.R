# Install required R packages for the pipeline
packages <- c(
  "tidyverse",
  "yaml",
  "readxl",
  "e1071",
  "caret",
  "yardstick",
  "doParallel",
  "rstatix",
  "rmcorr"
)

repos <- "https://cran.r-project.org/"
options(repos = c(CRAN = repos))

installed <- rownames(installed.packages())
to_install <- setdiff(packages, installed)

if (length(to_install) > 0) {
  install.packages(to_install)
} else {
  message("All R packages already installed.")
}
