#To avoid below error Created Personal library
#Installing package into ‘/usr/local/lib/R/site-library’
#(as ‘lib’ is unspecified)
#Warning in install.packages(c("hexView")) :
#'lib = "/usr/local/lib/R/site-library"' is not writable
#Error in install.packages(c("hexView")) : unable to install packages
#Execution halted

dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)  # create personal library
.libPaths(Sys.getenv("R_LIBS_USER"))  # add to the path

install.packages("hexView")  # install like always
library(hexView) 
