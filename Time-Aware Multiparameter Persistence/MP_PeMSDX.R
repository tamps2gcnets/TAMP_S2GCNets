############################################################################
# This code computes the Multiparameter Persistence, thus, returning 
# multipersistence grids of Beti values (0, 1, ...)
# The output matrices are input for Python code in which we build the 
# Dynamic Euler-Poincare Surface (DEPS)
#
# It runs on multiple cores
# Notes: Below code runs on PeMSD03/PeMSD04/PeMSD08 transportation networks
############################################################################

library(igraph) 
library(TDA)
library(reticulate) 
library(foreach)
library(doParallel)
rm(list = ls())
setwd('.../R_MP') 
options(java.parameters = "-Xmx200g")
np <- import("numpy")  

# 3 Graph ML datasets
dataPath <- ".../R_MP/Datasets/"
outputPath=".../R_MP/MultiFeature/Grids/"   

# max dimension of the homological features to be computed. (e.g. 0 for connected components, 1 for connected components and loops, 2 for connected components, loops, voids, etc.)
maxDimension <- 1

# upper limit on the simplex dimension, size of the cliques to find
maxClique <-2

# node count thresholds to ignore a graph
minNodeCount <- 4

# how many digits to consider in the grid
featureSensitivity =1

nodeFeatures <- c("closeness","betweenness","hub","degree")#,"eccentricity","authority")

outputFile <- "MPGrid"

# threshold to ignore large graphs
maxNodeCount <- 6000

GRIDCOUNT <- 50

# ****** Multicore !!!!!!
numberCoresLoop = 2 

source("nodeFunctions.R")


# function to find the secondary grid id of the value in a row.
get2ndGridId<-function(min,step,birth,death){
  if(birth>death){
    message("birth is later than death?",birth,">",death)
    return(-1)
  }
  gridIdStart<-0
  gridIdEnd<-0
  if(birth<=min){
    gridIdStart<-1
  } else{
    gridIdStart<-1+ceiling((birth-min)/step)
  }
  
  # if the hole has not died, it will have INF as value
  if(!is.finite(death)){
    gridIdEnd<-2+GRIDCOUNT
  }else{
    gridIdEnd<-1+ceiling((death-min)/step)
  }
  if(gridIdStart>gridIdEnd){
    message("An error occurred in the 2nd grid: ",gridIdStart,">",gridIdEnd)
    message(min,"  ",step,"  ",birth,"  ",death,"  ")
    return(-1)
  }
  return(gridIdStart:gridIdEnd)
}

###
# typeFiltrationF2: Type of filtration for feature F2
# 1: Sublevel -> previous
# 2: Power Filtration -> addition
# 
computePersistentGrid<-function(indexLow, indexTop, dataset, dataAlias, feature1, feature2, typeFiltrationF2=1){
  
  # Name of output file
  if(typeFiltrationF2==1){ # Sublevel filtration
    whichOutputFile <- paste0(outputPath,dataAlias,'_',outputFile,'_',feature1,'_sublevel_',feature2,'_sublevel.txt')
  } else if(typeFiltrationF2==2){ # Power filtration
    whichOutputFile <- paste0(outputPath,dataAlias,'_',outputFile,'_',feature1,'_sublevel_',feature2,'_power.txt')
  } 
  
  if(dataset=='ethereum'){ # ETHEREUM
    if(feature2=='transaction'){
      dataset_folder = 'ETHEREUM/ReducedDataset_TRAN100/EdgeList_W/';   
      nameSummary = paste0(dataPath,'ETHEREUM/ReducedDataset_TRAN100/ETH100summary.txt')
    } else if(feature2 == 'volume'){
      dataset_folder = 'ETHEREUM/ReducedDataset_VOL100/EdgeList_W/';   
      nameSummary = paste0(dataPath,'ETHEREUM/ReducedDataset_VOL100/ETH100summary.txt')
    }
    tblSummary = read.table(nameSummary, header = FALSE, sep = "", dec = ".") # To load the summary... 
    nameNetworks <- as.character(tblSummary[,6])
    idToken <- match(dataAlias, nameNetworks)
    NDays <- as.numeric(tblSummary[,5]) # NDays = NGraphs
    graphs <- 1:NDays[idToken]
    
  } else if((dataset=='PEMS04_05') || (dataset=='PEMS04_03') || (dataset=='PEMS04_01') || (dataset=='PEMS04_005')  || (dataset=='PEMS04_001') || (dataset=='PEMS08_05') || (dataset=='PEMS08_03') || (dataset=='PEMS08_01') || (dataset=='PEMS08_005') || (dataset=='PEMS08_001') || (dataset=='PEMS03_01') || (dataset=='PEMS03_005') || (dataset=='PEMS03_001')){ # TRANSPORTATION NETWORKS
    datasetSplit = strsplit(dataset,'_') 
    matNets <- np$load(paste0("Datasets/PEMS0X/",datasetSplit[[1]][1],"_networks_list_",datasetSplit[[1]][2],".npy"), allow_pickle=TRUE) 
    graphs = 1:length(matNets)
    
  } else { # OTHER DATASETS (FROM CUNEYT)
    edgeFile <- paste0(dataPath, dataset, "edges")
    graphFile <- paste0(dataPath, dataset, "graph_idx")
    graphIdData <-
      read.table(graphFile,
                 quote = "\"",
                 comment.char = "",
                 sep = ",")
    allData <-
      read.table(edgeFile,
                 quote = "\"",
                 comment.char = "",
                 sep = ",")
    colnames(allData) <- c("source", "target")
    # I am converting graph vertex ids to strings that starts with V
    # This is done to avoid orig vertex ids where igraph given internal ids
    allData$source<-paste0("v",allData$source)
    allData$target<-paste0("v",allData$target)
    graphs<-unique(graphIdData$V1)
  }
  
  bettiFrame<-data.frame()
  message("Processing ",dataAlias," for feature ",feature1," and ",feature2,": ",length(graphs)," graphs.")
  time_start = Sys.time()# To measure time
  
  ########
  cl_cores<-makeCluster(numberCoresLoop, type = "FORK")
  registerDoParallel(cores = cl_cores) # Register to parallel 
  print("Numer of cores (Parallel): ")
  getDoParWorkers()
  ########
    
    ##############
    bettiFrame_Block <- foreach(graphId = graphs[indexLow:indexTop]) %dopar% {
      
      bettiFrame_Net<-data.frame() # Internal bettiFrames
      
      if(dataset=='ethereum'){ # ETHEREUM
        nameFileNet <- paste0(dataPath,dataset_folder,'W',dataAlias,as.character(graphId),'.txt')
        
        edgeData = tryCatch(
          expr = { #*** The file was correctly read 
            P = read.table(nameFileNet) # Edge List 
            if(typeFiltrationF2 == 1){ # Sublevel
              P = P[1:2] # Only takes first two columns 
              colnames(P) <- c("source", "target") 
              P$source<-paste0("v", P$source) 
              P$target<-paste0("v", P$target)
            }else if(typeFiltrationF2 == 2){ # Power filtration
              # Takes three columns 
              colnames(P) <- c("source", "target", "weight") 
              P$source<-paste0("v", P$source) 
              P$target<-paste0("v", P$target)
            }
            P
            #return(P)
          },
          error = function(e){ #*** There was an error while reading the file (e.g. empty file, no lines)
            P = data.frame(t(c(1,2,0))) # Fake graph
            colnames(P) <- c("source", "target", "weight")
            P
            #return(4)
          }
        )
        
      } else if((dataset=='PEMS04_05') || (dataset=='PEMS04_03') || (dataset=='PEMS04_01') || (dataset=='PEMS04_005') || (dataset=='PEMS04_001') || (dataset=='PEMS08_05') || (dataset=='PEMS08_03') || (dataset=='PEMS08_01') || (dataset=='PEMS08_005') || (dataset=='PEMS08_001') || (dataset=='PEMS03_01') || (dataset=='PEMS03_005') || (dataset=='PEMS03_001')){ # TRANSPORTATION NETWORKS
        if(typeFiltrationF2 == 1){ # Sublevel
          P = matNets[[graphId]][,1:2] # Only takes first two columns 
          P = as.data.frame(P) # Convert to dataframe
          colnames(P) <- c("source", "target")
        }else if(typeFiltrationF2 == 2){ # Power filtration
          # Takes three columns 
          P = matNets[[graphId]] # 
          P = as.data.frame(P) # Convert to dataframe
          colnames(P) <- c("source", "target", "weight")
        }
        P$source<-paste0("v", P$source) 
        P$target<-paste0("v", P$target)
        edgeData = P
      } else { # OTHER DATASETS (previous)  ### It only works with -> typeFiltrationF2 = 1
        thisGraphNodes <- paste0("v",which(graphIdData$V1 == graphId))
        edgeData <-
          allData[allData$source %in% thisGraphNodes |
                    allData$target %in% thisGraphNodes, ]
      }
      
      graph <- graph.data.frame(edgeData, directed = FALSE) # Build the graph 
      nodeCount <- vcount(graph) 
      
      # if the graph is not too small nor too big, compute filtration
      if (nodeCount > minNodeCount && nodeCount < maxNodeCount) {
        # below we use sub level filtrations.
        #sublevel filtration
        # we will compute activation values of nodes based on two features
        
        #message(graph)
        epsilon <- 1e-10 
        
        ########  Feature F1 - Always sublevel filtration  ########
        f1Values = computeNodeVals(feature1, graph, dataset=dataset, dataPath=dataPath)
        # create a 10 step grid for feature1
        grid1Id<-0
        f1Vals <- f1Values[is.finite(f1Values)]
        min<-min(f1Vals)
        max<-max(f1Vals)
        # we cannot do multi persistence if our max and min values are Nan or Inf
        if(!is.finite(min)|!is.finite(max)){
          next
        }else if(((max-min)/GRIDCOUNT)<epsilon){
          max <- max + 0.0001
          #min <- min - 0.0001
        }
        
        feature1StepVal = (max-min)/GRIDCOUNT
        val1<-min
        
        ########  Feature F2 - Sublevel filtration or Power filtration  ########
        if(typeFiltrationF2 == 1){ ##### Sublevel filtration
          f2Values = computeNodeVals(feature2, graph, dataset=dataset, dataPath=dataPath)
          
        } else if(typeFiltrationF2 == 2){ ##### Power filtration
          distMatrixPowerAll <- shortest.paths(graph, v=V(graph), to=V(graph))
          f2Values = unique(c(distMatrixPowerAll))
          f2Values = f2Values[2:length(f2Values)] # To delete the first 0 value
        } 
        
        # create a grid for feature2
        gridId2<-0
        f2Vals <- f2Values[is.finite(f2Values)]
        min2<-min(f2Vals)
        max2<-max(f2Vals)
        # we cannot do multi persistence if our max and min values are Nan or Inf
        if(!is.finite(min2)|!is.finite(max2)){
          next
        }else if(((max2-min2)/GRIDCOUNT)<epsilon){
          max2 <- max2 + 0.0001
        }
        feature2StepVal = (max2-min2)/GRIDCOUNT
        
        # Verifying that 'step' is not to small
        #epsilon <- 1e-10
        if(feature1StepVal< epsilon|feature2StepVal< epsilon){
          # we cannot do multi persistence for this graph.
          message("max value is equal to min in ",dataAlias," graph ", graphId,
                  ":",feature1," and ",feature2)
          next;
        }
        
        # at this point, not all f1 and f2 values are NaN or Inf
        # we will convert NaN values to min-(step/2) 
        # and Inf values to max+(step/2)
        f1Values[is.na(f1Values)]<-min-(feature1StepVal/2)
        f1Values[!is.finite(f1Values)]<-max+(feature1StepVal/2)
        
        f2Values[is.na(f2Values)]<-min2-feature2StepVal/2
        f2Values[!is.finite(f2Values)]<-max2+feature2StepVal/2
        
        while(grid1Id<=GRIDCOUNT){
          # choose nodes that are active within the grid1
          grid1Id<-grid1Id+1
          vertices<-names(f1Values[f1Values<=val1])
          filteredGraph = induced_subgraph(graph,vertices)
          
          # We will compute multi persistence only if some nodes are activated
          if(vcount(filteredGraph)>0){
            
            ########  Feature F2 - Sublevel filtration or Power filtration  ########
            if(typeFiltrationF2 == 1){ ##### Sublevel filtration
              # choose the feature 2 values of active nodes  
              filteredF2Values<-f2Values[vertices]
              # for maxClique=3 below means we are adding 0,1,2 simplices (nodes,edges,triangles) to our complex
              cmplx <- cliques(as.undirected(filteredGraph), min = 0, max = maxClique)
              # use sublevel=T for sublevel, sublevel=F for superlevel filtration
              # F.values are node values. At these values, node appear in the complex,
              # and their edges to other active nodes also activate ...
              FltRips <- funFiltration(FUNvalues = filteredF2Values,
                                       cmplx = cmplx,
                                       sublevel = T) # Construct filtration using F.values
              
              #extract the persistence diagram
              # if there is a single activated vertex, the code below will give
              # a warning message.
              persistenceDiagram <-
                filtrationDiag(filtration = FltRips, maxdimension = maxDimension)$diagram
              
            } else if(typeFiltrationF2 == 2){ ##### Power filtration
              # Compute matrix of the distances between each node (geodesic distance / short path)
              distMatrixPower_FilteredGraph <- shortest.paths(filteredGraph, v=V(filteredGraph), to=V(filteredGraph))
              #diag(distMatrixPower_FilteredGraph) <- diagMatDis # Replace using the minimum value
              # To find scale
              valsDMP = unique(c(distMatrixPower_FilteredGraph))
              maxScale = max(valsDMP[is.finite(valsDMP)]) #+feature2StepVal/2
              # Rips Filtration (it computes complexes)
              FltRips <- ripsFiltration(X = distMatrixPower_FilteredGraph, maxdimension = maxDimension,
                                        maxscale = maxScale, dist = "arbitrary", printProgress = FALSE)
              
              # Persistence Diagram of Power Filtration
              persistenceDiagram <- filtrationDiag(filtration = FltRips, maxdimension = maxDimension)$diagram
            }
            
            bArray<-array(0,dim=c(2,GRIDCOUNT+3))
            
            for(rowIndex in 1:nrow(persistenceDiagram)) {
              row <- persistenceDiagram[rowIndex,]
              # R indexes start from 1. We will add 1 to b0 and b1 
              bettiNum<-1+as.integer(row[["dimension"]])
              birth<-row[["Birth"]]
              death<-row[["Death"]]
              gridId2<-get2ndGridId(min2,feature2StepVal,birth,death)
              if(gridId2!=-1){
                #message(bettiNum,"   ",min(gridId2),max(gridId2))
                gridId2[gridId2>(GRIDCOUNT+2)] = GRIDCOUNT+2 # Out of bound...
                bArray[bettiNum,gridId2]<-(1+bArray[bettiNum,gridId2])
              }
            }
            sumOfHoles <- sum(bArray)
            #  if we had any holes, we will write them to a file
            if(sumOfHoles>0){ 
              # write the betti number as the last value of the signature 
              bArray[1,GRIDCOUNT+3]<-0
              bArray[2,GRIDCOUNT+3]<-1
              pd2 <-cbind(GraphId = graphId,dataset = dataAlias,
                          Grid1=grid1Id,filVal=val1,f1=feature1,
                          f2=feature2,bArray)
              #*** FOR PARALLEL
              bettiFrame_Net <- rbind(bettiFrame_Net,pd2)
            }
          } #End if(vcount(filteredGraph)>0)
          
          val1<-val1+feature1StepVal
        } #End While
      }else {
        message("Ignoring ",dataAlias," graph ",graphId," Node count:",nodeCount)
      }
      
      bettiFrame_Net
    } # End FOR
    
    ########
    stopCluster(cl_cores)
    registerDoSEQ() # To go back to sequential
    print("Numer of cores (sequential): ")
    getDoParWorkers()
    ########
    
    
    ### Fill out the Frame (Append9)
    #print(bettiFrame_Block)  
    for (ik in 1:length(bettiFrame_Block)) {
      bettiFrame <- rbind(bettiFrame, bettiFrame_Block[[ik]])
    }
    
    ############## 
    # Print Status
    print(paste0(dataAlias,": ",GRIDCOUNT," - ",feature1," - ",feature2, '   <->   ',indexLow, ' to ' ,indexTop))
    time_final = Sys.time() 
    print(time_final-time_start)
    write.table(
      bettiFrame, ##** ORIGINAL
      file = whichOutputFile,
      sep = "\t",
      row.names = FALSE,
      col.names = FALSE,
      #append = F, ### OVERWRITE 
      append = T, # append
      quote = FALSE
    )
    
  
} ### END FUNCTION

###########################################
features<-c("degree","betweenness", "transaction", "volume")#"authority","eccentricity"
lenBlock = 2000
# degree and betweeness only works on sublevel filtration 
# transaction and volume only works on power filtration

######  RUNS ON ALL PEMS EXPERIMENTS  ###### 
lisSC <- c()
lisSC <- c(lisSC, 'PEMS03_001')
lisSC <- c(lisSC, 'PEMS04_001')
lisSC <- c(lisSC, 'PEMS08_001') 
NCToken <- length(lisSC); # Number of Tokens

# Name of output file (TIME)
whichOutputFile_Time <- paste0(outputPath, 'Time_', outputFile, '_', GRIDCOUNT, 'x', GRIDCOUNT, '_F1_F2.csv')  
timeFrame<-data.frame()

# To all datasets
for (nomNetGraph in lisSC) {
#for (nomToken in lisSC[1:3]) { # FIRST 3...    
#for (nomToken in lisSC[4:6]) { # NEXT 3....    
  dataset = nomNetGraph  
  dataAlias = nomNetGraph  
  datasetSplit = strsplit(dataset,'_') 
  if(datasetSplit[[1]][1]=='PEMS04'){
    totalNets = 16992  
    #totalNets = 7
  }else if(datasetSplit[[1]][1]=='PEMS08'){
    totalNets = 17856  
    #totalNets = 7
  }else if(datasetSplit[[1]][1]=='PEMS03'){
    totalNets = 26208  
    #totalNets = 7
  }
  
  # *** Sequence of index per blocks
  seqBlocks = seq(1, totalNets, lenBlock) 
  if(seqBlocks[length(seqBlocks)]<totalNets){
    seqBlocks <- c(seqBlocks, totalNets+1)
  }
  
  # *** To run by blocks
  time_start = Sys.time() # To measure time
  feature1 = features[[1]]
  feature2 = features[[2]]
  whichOutputFile <- paste0(outputPath,dataAlias,'_',outputFile,'_',feature1,'_sublevel_',feature2,'_sublevel.txt')
  #Check if a previous result file  exists
  if(file.exists(whichOutputFile)) {
    file.remove(whichOutputFile)
  }
  for (iB in 1:(length(seqBlocks)-1)) {
    computePersistentGrid(seqBlocks[iB], seqBlocks[iB+1]-1, dataset, dataAlias, feature1=feature1, feature2=feature2, typeFiltrationF2 = 1)
  }
  time_final = Sys.time() # To measure time
  print(time_final-time_start)
  # Add time to DataFrame
  rowTime = cbind(TS=paste0(nomNetGraph,'_',feature1,'_',feature2), Time=as.numeric(difftime(time_final, time_start, units = "secs")))
  timeFrame<-rbind(timeFrame, rowTime)

  # *** To run by blocks
  time_start = Sys.time() # To measure time
  feature1 = features[[1]]
  feature2 = features[[3]]
  whichOutputFile <- paste0(outputPath,dataAlias,'_',outputFile,'_',feature1,'_sublevel_',feature2,'_power.txt')
  #Check if a previous result file  exists
  if(file.exists(whichOutputFile)) {
    file.remove(whichOutputFile)
  }
  for (iB in 1:(length(seqBlocks)-1)) {
    computePersistentGrid(seqBlocks[iB], seqBlocks[iB+1]-1, dataset, dataAlias, feature1=feature1, feature2=feature2, typeFiltrationF2 = 2)
  }
  time_final = Sys.time() # To measure time
  print(time_final-time_start)
  # Add time to DataFrame
  rowTime = cbind(TS=paste0(nomNetGraph,'_',feature1,'_',feature2), Time=as.numeric(difftime(time_final, time_start, units = "secs")))
  timeFrame<-rbind(timeFrame, rowTime)
  
  # *** To run by blocks
  time_start = Sys.time() # To measure time
  feature1=features[[2]]
  feature2=features[[3]]
  whichOutputFile <- paste0(outputPath,dataAlias,'_',outputFile,'_',feature1,'_sublevel_',feature2,'_power.txt')
  #Check if a previous result file  exists
  if(file.exists(whichOutputFile)) {
    file.remove(whichOutputFile)
  }
  for (iB in 1:(length(seqBlocks)-1)) {
    computePersistentGrid(seqBlocks[iB], seqBlocks[iB+1]-1, dataset, dataAlias, feature1=feature1, feature2=feature2, typeFiltrationF2 = 2)
  }
  time_final = Sys.time() # To measure time
  print(time_final-time_start)
  # Add time to DataFrame
  rowTime = cbind(TS=paste0(nomNetGraph,'_',feature1,'_',feature2), Time=as.numeric(difftime(time_final, time_start, units = "secs")))
  timeFrame<-rbind(timeFrame, rowTime)
  
} # End FOR

# Save Times
write.table(
  timeFrame,
  file = whichOutputFile_Time,
  sep = ",",
  row.names = FALSE,
  col.names = FALSE,
  append = F,
  quote = FALSE
)


