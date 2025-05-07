class RunData():
    def __init__(self, run_id, data):
        self.run_id = run_id
        self.data   = data    #list of fit1, node1, fit2, node2

class ConfigData():
    def __init__(self, problemType="Discrete Problem",
                 partitionStrategy= "Shannon entropy",
                 dPartitioning=0,
                 dCSize=50,
                 dVSize=25,
                 dDistance="hamming",
                 dMinBound=-100,
                 cMaxBound=100,
                 cDimension=3,
                 cHypercube=0,
                 cClusterSize=50,
                 cVolumeSize=50,
                 cDistance="euclidian"
                 ):
        self.problemType       = problemType
        self.partitionStrategy = partitionStrategy
        self.dPartitioning     = dPartitioning
        self.dCSize            = dCSize
        self.dVSize            = dVSize
        self.dDistance         = dDistance
        self.dMinBound         = dMinBound
        self.cMaxBound         = cMaxBound
        self.cDimension       = cDimension
        self.cHypercube        = cHypercube
        self.cClusterSize      = cClusterSize
        self.cVolumeSize       = cVolumeSize
        self.cDistance         = cDistance

