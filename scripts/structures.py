class RunData():
    def __init__(self, run_id, data):
        self.run_id = run_id
        self.data   = data

class ConfigData():
    def __init__(self, problemType="discrete",
                 objectiveType = "minimization",
                 partitionStrategy= "shannon",
                 dPartitioning=0,
                 dCSize=50,
                 dVSize=25,
                 dDistance="hamming",
                 cMinBound=-100,
                 cMaxBound=100,
                 cDimension=3,
                 cHypercube=0,
                 cClusterSize=50,
                 cVolumeSize=50,
                 cDistance="euclidean",
                 advanced = None
                 ):
        self.problemType       = problemType
        self.objectiveType     = objectiveType
        self.partitionStrategy = partitionStrategy
        self.dPartitioning     = dPartitioning
        self.dCSize            = dCSize
        self.dVSize            = dVSize
        self.dDistance         = dDistance
        self.cMinBound         = cMinBound
        self.cMaxBound         = cMaxBound
        self.cDimension        = cDimension
        self.cHypercube        = cHypercube
        self.cClusterSize      = cClusterSize
        self.cVolumeSize       = cVolumeSize
        self.cDistance         = cDistance
        self.advanced = advanced if advanced else AdvancedOptions()

    def toDict(self):
        return {
            "problemType":       self.problemType,
            "objectiveType":     self.objectiveType,
            "partitionStrategy": self.partitionStrategy,
            "dPartitioning":     self.dPartitioning,
            "dCSize":            self.dCSize,
            "dVSize":            self.dVSize,
            "dDistance":         self.dDistance,
            "cMinBound":        self.cMinBound,
            "cMaxBound":        self.cMaxBound,
            "cDimension":       self.cDimension,
            "cHypercube":       self.cHypercube,
            "cClusterSize":     self.cClusterSize,
            "cVolumeSize":      self.cVolumeSize,
            "cDistance":        self.cDistance,
            "advanced": self.advanced.to_dict() if self.advanced else {}
        }

    @classmethod
    def from_dict(cls, d):
        adv = d.get("advanced", {})
        return cls(
            problemType=d.get("problemType", "discrete"),
            objectiveType=d.get("objectiveType", "minimization"),
            partitionStrategy=d.get("partitionStrategy", "shannon"),
            dPartitioning=d.get("dPartitioning", 0),
            dCSize=d.get("dCSize", 50),
            dVSize=d.get("dVSize", 25),
            dDistance=d.get("dDistance", "hamming"),
            cMinBound=d.get("cMinBound", -100),
            cMaxBound=d.get("cMaxBound", 100),
            cDimension=d.get("cDimension", 3),
            cHypercube=d.get("cHypercube", 0),
            cClusterSize=d.get("cClusterSize", 50),
            cVolumeSize=d.get("cVolumeSize", 50),
            cDistance=d.get("cDistance", "euclidean"),
            advanced=AdvancedOptions.from_dictionary(adv)
        )

class AdvancedOptions():
    def __init__(self,
                 best_solution = "",
                 nr_of_runs    = -1,
                 vertex_size   = -1,
                 arrow_size    = -1,
                 tree_layout   = False):
        self.best_solution = best_solution
        self.nr_of_runs    = nr_of_runs
        self.vertex_size   = vertex_size
        self.arrow_size    = arrow_size
        self.tree_layout   = tree_layout

    def to_dict(self):
        return {
            "best_solution" : self.best_solution,
            "nr_of_runs"    : self.nr_of_runs,
            "vertex_size"   : self.vertex_size,
            "arrow_size"    : self.arrow_size,
            "tree_layout"   : self.tree_layout
        }

    @classmethod
    def from_dictionary(cls, d):
        return cls(
            best_solution = d.get("best_solution", ""),
            nr_of_runs    = d.get("nr_of_runs", -1),
            vertex_size   = d.get("vertex_size", -1),
            arrow_size    = d.get("arrow_size", -1),
            tree_layout   = d.get("tree_layout", False)
        )


