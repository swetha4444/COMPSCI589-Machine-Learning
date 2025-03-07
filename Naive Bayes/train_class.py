class TrainClassObject:
    def __init__(self,className,freqMap,totalCount,priorProb):
        self.className = className
        self.freqMap = freqMap
        self.totalCount = totalCount
        self.priorProb = priorProb
    
    