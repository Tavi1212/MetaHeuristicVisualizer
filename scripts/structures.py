class RunData():
    def __init__(self, run_id, data):
        self.run_id = run_id
        self.data   = data

class ConfigData():
    def __init__(self, problemType="Discrete Problem",
                 objectiveType = "minimization",
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
                 cDistance="euclidean"
                 ):
        self.problemType       = problemType
        self.objectiveType     = objectiveType
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

    def toDict(self):
        return {
            "problemType":       self.problemType,
            "objectiveType":     self.objectiveType,
            "partitionStrategy": self.partitionStrategy,
            "dPartitioning":     self.dPartitioning,
            "dCSize":            self.dCSize,
            "dVSize":            self.dVSize,
            "dDistance":         self.dDistance,
            "dMinBound":        self.dMinBound,
            "cMaxBound":        self.cMaxBound,
            "cDimension":       self.cDimension,
            "cHypercube":       self.cHypercube,
            "cClusterSize":     self.cClusterSize,
            "cVolumeSize":      self.cVolumeSize,
            "cDistance":        self.cDistance
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            problemType=d.get("problemType", "Discrete Problem"),
            objectiveType=d.get("objectiveType", "minimization"),
            partitionStrategy=d.get("partitionStrategy", "Shannon entropy"),
            dPartitioning=d.get("dPartitioning", 0),
            dCSize=d.get("dCSize", 50),
            dVSize=d.get("dVSize", 25),
            dDistance=d.get("dDistance", "hamming"),
            dMinBound=d.get("dMinBound", -100),
            cMaxBound=d.get("cMaxBound", 100),
            cDimension=d.get("cDimension", 3),
            cHypercube=d.get("cHypercube", 0),
            cClusterSize=d.get("cClusterSize", 50),
            cVolumeSize=d.get("cVolumeSize", 50),
            cDistance=d.get("cDistance", "euclidean"),
        )
