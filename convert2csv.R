install.packages("hexView")
args <- commandArgs(trailingOnly = TRUE)
opus.file.path <- args[1]
output.path <- args[2]
file <- file.path(output.path, "Spectra.csv")

# Debugging output
cat("opus.file.path:", opus.file.path, "\n")
cat("output.path:", output.path, "\n")
cat("file:", file, "\n")

# Ensure the output directory exists
if (!dir.exists(output.path)) {
  dir.create(output.path, recursive = TRUE)
}

# Source the R script and handle any errors
source.path <- '.'  # Assuming read.opus.R is in the current directory
source(paste0(source.path, '/read.opus.R'), chdir = TRUE, echo = TRUE)

# List files and process them
lst <- as.list(list.files(path=opus.file.path, pattern=".[0-9]$", full.names=TRUE))
spectra <- c()
for (i in 1:length(lst)) {
  spec <- opus(lst[[i]], speclib="ICRAF", plot.spectra=TRUE, print.progress=TRUE)
  spectra <- rbind(spectra, spec)
}

# View part of spectra
spectra[,1:8]

# Write to CSV
tryCatch({
  write.table(spectra, file=file, sep=",", row.names=FALSE)
}, error = function(e) {
  cat("Error in writing file:", e$message, "\n")
})
