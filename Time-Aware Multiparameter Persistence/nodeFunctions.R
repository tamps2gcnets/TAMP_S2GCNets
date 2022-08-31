############################################################################
# This source code includes functions for main computations
############################################################################

require(igraph)

loadRicci<-function(dataset,graph, dataPath, method){
  datFile <- paste0(dataPath,dataset, method)
  dat <-
    read.table(datFile,
               quote = "\"",
               comment.char = "",
               sep = "\t")
  return(dat[dat$V2%in%V(graph),2:3])
}

# Function to compute node function on an igraph object.
# Some functions (e.g., degree) are discrete
# some (e.g., closeness) are continues in [0,1].
# inputs:
# 1 - feature: node feature that will computed (e.g., degree)
# 2 - thisGraph: an igraph object
# outputs:
# a vector of node feature values
computeNodeVals <- function(feature, thisGraph, dataset="",dataPath="") {
  if (feature == "degree") {
    nodeValues <- apply(as_adjacency_matrix(thisGraph), 1, sum)
  }else if (feature == "authority") {
    # for undirected graphs, hub scores are equal to authorithy scores
    nodeValues = authority_score(thisGraph)$vector
  }else if (feature == "closeness") {
    nodeValues = closeness(thisGraph,normalized=TRUE)
  }else if (feature == "betweenness") {
    nodeValues = betweenness(thisGraph,normalized=TRUE)
  }else if (feature == "eccentricity") {
    nodeValues = eccentricity(thisGraph)
  }else if (feature == "ricci") {
    valTable = loadRicci(dataset,thisGraph,dataPath,"ricci")
    nodeValues<-dplyr::pull(valTable, V3)
    names(nodeValues)<-dplyr::pull(valTable, V2)
  }else if (feature == "forman") {
    valTable = loadRicci(dataset,thisGraph,dataPath,"forman")
    nodeValues<-dplyr::pull(valTable, V3)
    names(nodeValues)<-dplyr::pull(valTable, V2)
  }else if (feature == "hub") {
    # for undirected graphs, hub scores are equal to authorithy scores
    nodeValues = hub_score(thisGraph)$vector
  }else {
    message(nodeFeature," has not ben implemented as a node function in computeNodeVals()")
  }
  return(nodeValues)
}