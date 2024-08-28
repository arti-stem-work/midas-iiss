#To avoid below error Created Personal library
#Installing package into ‘/usr/local/lib/R/site-library’
#(as ‘lib’ is unspecified)
#Warning in install.packages(c("hexView")) :
#'lib = "/usr/local/lib/R/site-library"' is not writable
#Error in install.packages(c("hexView")) : unable to install packages
#Execution halted

# install_hexView.R

# Create personal library if it does not exist
personal_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(personal_lib)) {
  dir.create(personal_lib, recursive = TRUE)
}
.libPaths(c(personal_lib, .libPaths()))

# Function to install packages
install_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    tryCatch({
      install.packages(pkg, lib = personal_lib)
    }, error = function(e) {
      message("Error installing package: ", pkg)
      message("Error: ", e$message)
    })
  }
}

# Install hexView package
install_package("hexView")

# Load the package
if (!requireNamespace("hexView", quietly = TRUE)) {
  stop("Package hexView could not be installed.")
}
library(hexView)
