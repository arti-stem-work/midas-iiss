#Author: Andrew Sila
#Email: asila@cgiar.org
#Purpose: Function for reading Bruker OPUS files from MPA, HTS-xt and Alpha
#Date: 30/March/2014
#Version: Alpha (Testing)

#To use the function you are required to change the file paths at the bottom of this file: 
#1. opus.folder=Folder with raw spectra to be converted
#2. converted.folder=Folder where converted tables will be kept
#3. run the last line to get conversions

	
require(hexView) #Needs readRaw function from this library
opus <- function(file.name, sp=NULL, codes=c("ZFF","RES","SNM","DAT","LWN","FXV","LXV","NPT","MXY","MNY","END","TIM"), plot.spectra=FALSE, print.progress=FALSE, speclib="ICRAF")
	{
	spec.opts <- new.env(hash=TRUE)
	spec.env <- function(MIR = c(390, 7500), NIRS = c(3900, 12500), NIRP = c(4000,10000), VISNIR1 = c(420, 960), VISNIR2 = c(1020, 1770), VISNIR3 = c(1830, 2480), icraf.htsxt = c(3578, 7497.964, 599.76), icraf.alpha = c(2542, 3998.12872, 399.387991),icraf.alpha.znse = c(1714, 3996.4810, 499.8151), icraf.mpa = c(2307, 12493.2, 3598.69), CO2.band = c(2350.8,2379.8), signif.digit=5, attributes = c("ORCCNS", "PHIHOX", "ALUM3S", "ECAM3S", "EXKM3S", "EMGM3S", "ENAM3S", "EXB", "NITCNS", "SNDLDF"), mdnames = c("Instrument_name", "Instrument_URL", "Laboratory_name", "Laboratory_contact", "Laboratory_URL", "Material_class", "Wavenumber_conversion", "Wavenlength_unit", "Location_error"), show.env = FALSE){
   	pl.lst <- list(
     MIR = MIR,
     NIRS = NIRS,
     NIRP = NIRP,
     VISNIR1 = VISNIR1,
     VISNIR2 = VISNIR2,
     VISNIR3 = VISNIR3,
     icraf.htsxt = icraf.htsxt,
     icraf.alpha = icraf.alpha,
     icraf.mpa = icraf.mpa,
     icraf.alpha.znse = icraf.alpha.znse,
     CO2.band = CO2.band,
     signif.digit = signif.digit,
     attributes = attributes,
     mdnames = mdnames
    )
   x <- lapply(names(pl.lst), function(x){ assign(x, pl.lst[[x]], envir=spec.opts) })
   if(show.env){
     return(pl.lst)
   				}
}
	spec.env()
		if(!(speclib=="ICRAF"|speclib=="New")){ stop("'speclib' must be one of the following: 'ICRAF' or 'New'") }
   		if(file.exists(file.name)){
    	## Read metainfo
	  	try( pa <- hexView::readRaw(file.name, offset = 0, nbytes = file.info(file.name)$size, human = "char", size = 1, endian = "little"), silent=TRUE )
      	if(!class(.Last.value)[1]=="try-error"){
        		pr <- pa$fileRaw
	   		## Get source of instrument
		  	ins <- grepRaw("INS", pr, all=TRUE)
		 	ins <- readRaw(file.name, offset = ins[length(ins)]+7, nbytes = 3, human = "char", size = 1, endian = "little")
		 	ins <- blockString(ins)
		 	## Get source of infrared to know if NIR or MIR
		    	src <- grepRaw("SRC", pr, all=TRUE)
		    	src <- readRaw(file.name, offset = src[length(src)]+4, nbytes = 3, human = "char", size = 1, endian = "little")
		    	src <- blockString(src)
  		  	instr.range <- tolower(paste(ins, src, sep="-"))
  		  	## Get Beam Splitter
  		 	bms <- grepRaw("BMS", pr, all=TRUE)
		    	bms <- readRaw(file.name, offset = bms[length(bms)]+4, nbytes = 4, human = "char", size = 1, endian = "little")
		    	bms <- blockString(bms)
         		## Wavenumbers for MIR spectra from Tensor are assigned prefix "m", MIR spectra from Alpha prefixed "a"; for NIR MPA "n"
        		if(instr.range=="ten-off"){ instr.range="ten-mir"} ## AS: Old ten-mir written as tensor-27
				pref <- ifelse(instr.range=="ten-mir", "m", ifelse(instr.range=="alp-mir", "a", ifelse(instr.range=="mpa-nir", "n", "X")))
			  	if(speclib=="ICRAF"){ 
          			if(instr.range=="ten-mir"){
					  wb <- rev(seq(get("icraf.htsxt", spec.opts)[3], get("icraf.htsxt", spec.opts)[2], (get("icraf.htsxt", spec.opts)[2]-get("icraf.htsxt", spec.opts)[3])/(get("icraf.htsxt", spec.opts)[1]-1)))
				  								}
				  	if(instr.range=="alp-mir"){
				  		if(bms!="ZnSe"){
						wb <- rev(seq(get("icraf.alpha", spec.opts)[3], get("icraf.alpha", spec.opts)[2], (get("icraf.alpha", spec.opts)[2]-get("icraf.alpha", spec.opts)[3])/(get("icraf.alpha", spec.opts)[1]-1)))
          								}
          				if(bms=="ZnSe"){
          				wb <- rev(seq(get("icraf.alpha.znse", spec.opts)[3], get("icraf.alpha.znse", spec.opts)[2], (get("icraf.alpha.znse", spec.opts)[2]-get("icraf.alpha.znse", spec.opts)[3])/													(get("icraf.alpha.znse", spec.opts)[1]-1)))}
 				 						}

			    			if(instr.range=="mpa-nir"){
						wb <- rev(seq(get("icraf.mpa", spec.opts)[3], get("icraf.mpa", spec.opts)[2], (get("icraf.mpa", spec.opts)[2]-get("icraf.mpa", spec.opts)[3])/(get("icraf.mpa", spec.opts)[1]-1)))
 				  								  }
	 	    						}
	 	     if(!(instr.range=="ten-mir"|instr.range=="alp-mir"|instr.range=="mpa-nir"|instr.range=="ten-off"|bms=="ZnSe")){ stop("Unknown file format. See '?read.opus' for more info.") }  
					if(speclib=="New"){
      	  				if(instr.range=="ten-mir"){ pref="m" }
			 			if(instr.range=="alp-mir"){ pref="a"}
						if(instr.range=="mpa-nir"){ pref="n"}
						if(bms=="ZnSe"){ pref="a"}
		    							}
			## Get positions where the following parameters are found in the file  
     			z <- grepRaw(codes[1],pr,all=TRUE)[1]+5
  			re <- grepRaw(codes[2],pr,all=TRUE)[1]+5
  	 		snm <- grepRaw(codes[3],pr,all=TRUE)[1]+7
  	 		dat <- grepRaw(codes[4],pr,all=TRUE)[1]+7
  	 		lwn <- grepRaw(codes[5],pr,all=TRUE)[1]+7
  	 		fx <- grepRaw(codes[6],pr,all=TRUE)[3]+7
     			lx <- grepRaw(codes[7],pr,all=TRUE)[3]+7
     			npt0 <- grepRaw(codes[8],pr,all=TRUE)[2]+3
     			npt1 <- grepRaw(codes[8],pr,all=TRUE)[3]+7
     			mxy <- grepRaw(codes[9],pr,all=TRUE)[1]+7 
     			mny <- grepRaw(codes[10],pr,all=TRUE)[3]+7 
     			end <- grepRaw(codes[11],pr,all=TRUE)+11
     			tim <- grepRaw(codes[12],pr,all=TRUE)+11
			## calculate end and start of each block:
 			offs <- sapply(5:10, function(x){end[x]})
     			byts <- diff(offs)
  	 		ZFF <- readRaw(file.name, offset=z, nbytes=4, human="int", size=2)[[5]][1]
  	 		RES <- readRaw(file.name, offset=re, nbytes=4, human="int", size=2)[[5]][1]
  			snm.lab.material <- blockString(readRaw(file.name, offset = snm, nbytes = 22, human = "char", size = 1, endian = "little"))
  			if(!nzchar(snm.lab.material)){
  		 	  SSN <- ""
  		  	  Material <- ""
  		  	  warning("Product name not found inside OPUS file...")
  		      								}
  		    	else {
            		if(!length(grep(snm.lab.material, pattern=";"))==0){
            		snm.lab.material <- as.vector(strsplit(snm.lab.material,";"))[[1]]
           			SSN <- paste0(snm.lab.material[2], snm.lab.material[1])
           			Material <- snm.lab.material[3]
   			  															} 
   			  			else {
       		 				 if(!length(grep(snm.lab.material, pattern="_"))==0){
          	  					SSN <- sub("_", "", snm.lab.material)
           						Material <- ""          
    																			}
    						else {
    							if(!length(snm.lab.material)==0){ 
         		 	  		SSN <- snm.lab.material
         		 			Material <- ""          
    				  											}
       																	}
    		  				}   
   					}
		## Set three SSN first three characters to lower
		SSN <- paste0(tolower(substr(SSN,1,3)), substr(SSN,4,20))
  	 	Scandate <- blockString(readRaw(file.name, offset = dat, nbytes = 10, human = "char", size = 1, endian = "little"))
  	 	Scantime <- blockString(readRaw(file.name, offset = tim[2]-4, nbytes = 8, human = "char", size = 1, endian = "little"))
  	 	Scandate <- paste(Scandate,Scantime)
  		LWN <- readRaw(file.name, offset=lwn, nbytes=8, human="real", size=8)[[5]][1]
  	 	## Combine the above parameters
		spectrum.meta <- c(SSN, Material, Scandate, ZFF, RES, LWN)
		## Get number of data points for each spectra data block   		
  	 	NPT0 <- readRaw(file.name, offset=npt0, nbytes=12, human="int", size=4)[[5]][2]
  	 	NPT1 <- readRaw(file.name, offset=npt1, nbytes=4, human="int", size=4)[[5]][1]
  	 	fxv <- readRaw(file.name, offset=fx, nbytes=16, human="real", size=8)[[5]][1] ## fxv:	Frequency of first point
  	 	lxv <- readRaw(file.name, offset=lx, nbytes=16, human="real", size=8)[[5]][1] ## lxv:	Frequency of last point
  	 	Wavenumbers <- rev(seq(lxv, fxv, (fxv-lxv)/(NPT1-1)))
  	 	## Read all through all the data blocks inside the OPUS file:
  	 	nbytes1 <- NPT0*4 ## initial parameters
  	 	smxa <- c()
  	 	smna <- c()
  	 	nbytes.f <- NPT1*4
  	 	#Start reading available data blocks
          		mxsa<-NULL
			for (f in 1:3){
				offs.f<-offs[f]
				opus.p <- readRaw(file.name,width=NULL,offset=offs.f-4,nbytes=nbytes.f,human="real",size=4,endian="little") 
      			spectra <- opus.p[[5]]
      			mxs<-max(spectra)
      			mxsa<-c(mxsa,mxs)
						}
		#Determine valid data block to read
			fs<-which(mxsa==max(mxsa))
			if (!length(fs)<1){
				offs.f<-offs[fs]
				opus.p <- readRaw(file.name,width=NULL,offset=offs.f-4,nbytes=nbytes.f,human="real",size=4,endian="little") 
      			spectra <- opus.p[[5]]
  						  }
			## Make compatible to ICRAF spectra:
			if(speclib=="ICRAF"){
			spectra <- spline(Wavenumbers, spectra, xout=wb, method="natural")$y
			Wavenumbers <- wb
      		   }
			## Specify if graphics showing spectra being converted is displayed
			if(plot.spectra==TRUE){
			plot(Wavenumbers, spectra, ylab="Absorabance", xlab=expression("Wavenumbers cm"^-1), type="l")   
			mtext(paste("File source: ", getwd(),file.name,sep="/"), side=3,line=2,cex=1)
 	    			}
			## Print progress of conversion
			if(print.progress==TRUE){
			message(paste("Converting ", file.name, " file", sep=""))
				}
	    	## Add meta ID
  	  	#if(missing(MID)){ MID <- paste(Sys.getenv(c("USERDNSDOMAIN"))[[1]], instr.range, sep="_") } 
		## create data.frames:
		samples <- data.frame(SAMPLEID=spectrum.meta[1], Material=spectrum.meta[2], Zero.Filing=spectrum.meta[4], Resolution=spectrum.meta[5], LWN=spectrum.meta[6], DateTime=as.POSIXct(spectrum.meta[3], format="%d/%m/%Y %H:%M:%S "))
		ab <- data.frame(as.list(spectra, signif.digit))
		#spectrum.meta<-c(paste0(spectrum.meta[2],spectrum.meta[1]),spectrum.meta[-c(1:2)])#When no material type is not included
     		spec<-c(spectrum.meta,spectra)
          names(spec)<-c("SAMPLEID","Material","Datetime","Zero.filling","Resolution","LWN",paste0(pref,round(Wavenumbers,1)))
          #names(spec)<-c("SAMPLEID","Datetime","Zero.filling","Resolution","LWN",paste0(pref,round(Wavenumbers,1)))

    			}
    out<-spec
  } else {
	    warning(paste("File",file.name,"does not exist"))
	}
	}
