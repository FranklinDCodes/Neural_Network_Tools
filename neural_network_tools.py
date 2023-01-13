import random
import datetime
import numpy as np
from json import dumps, loads
from sql_tools import db
from os import path
import excel_tools

LEAK = 0.01

# LEAKY RELU
def activate(val):

    # leaky ReLU
    global LEAK
    if val > 0:
        return val
    else:
        return val*LEAK

# NEURON
class Neuron:

    def __init__(self, layer, position, connections, prevLayer, bias, rounding):
        # connections to previous layer
        # [position, weight]
        self.connections = connections
        self.bias = bias
        self.layer = layer
        self.position = position
        self.value = None
        self.previousLayer = prevLayer
        self.activation = None
        self.rounding = rounding
        # placeholder for caclulating derivative
        self.derMult = 0

    def calculateValue(self):
        vals = 0

        #value before activation
        for i in self.connections:
            vals += np.double(self.previousLayer.neurons[i[0]].value * i[1])
        
        # add bias
        raw = vals + self.bias

        # activation
        if raw >= 0:
            self.activation = 1
        else:
            self.activation = LEAK
        self.value = activate(raw)

        # new value
        self.value = round(self.value, self.rounding)
        return self.value
    
    def caluclateDerMult(self):

        # calculate it's derivative with respect to some x value in previous layer
        xMults = np.array([self.previoiusLayer[i[0]].xMult for i in self.connections], dtype=np.float64)
        weights = np.array([i[1] for i in self.connections], dtype=np.float64)
        weightMults = xMults*weights
        self.xMult = sum(weightMults)
        return self.xMult

# INPUT NEURON
class INeuron:

    def __init__(self, value, position):
        self.layer = 0
        self.value = value
        self.position = position
        self.derMult = 0

# INPUT NEURON LAYER
class ILayer:

    def __init__(self, values):
        self.neurons = []
        self.id = 0
        ind = 0
        for i in values:
            n = INeuron(i, ind)
            self.neurons.append(n)
            ind += 1

# NEURON LAYER
class Layer:
    
    def __init__(self, id, size, prevLayer, weightRange, biasRange, rounding, connectionsList=None, biasList=None):
        self.id = id
        self.neurons = []
        self.previousLayer = prevLayer

        con2 = []
        con3 = []

        # list with indexes of previous layer neurons
        connectionTemplate = [i for i in range(len(prevLayer.neurons))]
        if connectionsList is None:
            
            # generate random values for weights and biases
            for ii in range(size):
                connections = []

                for i in connectionTemplate:
                    
                    # random weight
                    weight = round(random.uniform(-float(weightRange/2), float(weightRange/2)), rounding)

                    # add to connections for neuron
                    connections.append([i, weight])
                    con2.append([ii, weight, i])

                # random bias
                bias = round(random.uniform(-float(biasRange/2), float(biasRange/2)), rounding)

                # create neuron
                n = Neuron(id, ii, connections, prevLayer, bias, rounding)
                con3.append([ii, bias])

                # add neuron
                self.neurons.append(n)
        else:

            # loading prexisting values
            ind = 0
            for i in range(size):
                try:

                    # try to load appropriate weight list and bias for the index of neuron
                    n = Neuron(id, i, connectionsList[ind], prevLayer, biasList[ind], rounding)
                except:

                    # sometimes if program was closed while savings, weights are lost
                    # in this case go to backup databases
                    print(f"{id}, {i}")

                # add values to con2 and con3
                for ii in connectionsList[ind]:
                    con2.append([i, ii[1], ii[0]])
                con3.append([i, biasList[ind]])

                # add neuron to list
                self.neurons.append(n)
                ind += 1

        # use con2 and con3 to add all the variables for this layer to the dict containing all the variables
        self.varIndex = {(self.id, i[0], i[2]):i[1] for i in con2}
        self.varIndex.update({(self.id, i[0], None):i[1] for i in con3})
        
# NEURAL NETWORK
class Network:

    def __init__(self, name, hiddenLayerCount=None, inputSize=None, outputSize=None, hiddenLayerSizes=None, weightRange=None, biasRange=None, highDataMode=False, dataAccuracy=4):
        
        # database connection
        self.dataBase = db(path.join("databases", f"{name}.db"))

        # tell sql_tools not to print commands executed
        self.dataBase.printCmds = False

        # see if a network already exists with this database 
        if len(self.dataBase.table) > 0:

            # load existing network from database
            self.existingInit()
        else:

            # create new network
            self.newInit(name, hiddenLayerCount, inputSize, outputSize, hiddenLayerSizes, weightRange, biasRange, highDataMode, dataAccuracy)
        print(f"\n{self.name} NEURAL NETWORK INITIALIZED\n")

    def system(self):
        # prints out values in all layers
        for ii in self.layers:
            print(f"LAYER {ii.id}")
            if ii.id != 0:
                for i in ii.neurons:
                    print(f"{i.layer}, {i.position} | {round(i.value, 8)} | {i.bias} | {[q[1] for q in i.connections]}")
            else:
                for i in ii.neurons:
                    print(f"{i.layer}, {i.position} | {round(i.value, 8)}")
        print()
    
    def systemOut(self):
        # prints out values in output layer
        print('OUTPUT LAYER')
        for neur in self.outputLayer.neurons:
            print(f"{neur.layer}, {neur.position + 1} | {round(neur.value, 8)}")
            

    def newInit(self, name, hiddenLayerCount, inputSize, outputSize, hiddenLayerSizes, weightRange, biasRange, highData, rounding):
        # blank input
        startInp = [None for i in range(inputSize)]

        # init layer 1
        self.inputLayer = ILayer(startInp)
        self.hiddenLayers = []

        # init hidden layers
        ind = 1
        for i in range(hiddenLayerCount):
            prevLay = self.inputLayer
            if self.hiddenLayers.__len__() > 0:
                prevLay = self.hiddenLayers[ind - 2]
            nl = Layer(ind, hiddenLayerSizes[ind - 1], prevLay, weightRange, biasRange, rounding)
            self.hiddenLayers.append(nl)
            ind += 1
        
        # init output layer
        self.outputLayer = Layer(hiddenLayerCount + 1, outputSize, self.hiddenLayers[-1], weightRange, biasRange, rounding)
        
        # define layers
        self.layers = [self.inputLayer]
        self.layers.extend(self.hiddenLayers)
        self.layers.append(self.outputLayer)

        # define layers with equations (all but input)
        self.eqLayers = self.hiddenLayers
        self.eqLayers.append(self.outputLayer)

        # define allEqNeurons in eqLayers
        self.allEqNeurons = []
        for i in self.eqLayers:
            self.allEqNeurons.extend(i.neurons)

        # define allNeurons in Layers
        self.allNeurons = []
        for i in self.layers:
            self.allNeurons.extend(i.neurons)
        
        # define allNeurons listed attributes
        self.allNeuronsData = [(l.layer, l.position, l.connections) for l in self.allEqNeurons]

        # var index
        self.varIndex = {}
        for i in self.layers:
            if i.id == 0:
                continue
            self.varIndex.update(i.varIndex)

        # IF NEW NETWORK
        # table info
        self.name = name
        self.weightRange = weightRange
        self.biasRange = biasRange
        self.rounding = rounding
        self.timesTrained = 0

        # 1 row: name, hidden layer count, sizes for hidden layers, input size, output size, bias range, weight range, times trained, record structure?, dataAccuracy (rounding)
        self.dataBase.newTable("Info", ("name", "hlc", "hls", "inps", "os", "br", "wr", "tt", "rs", "da"))
        self.dataBase.table["Info"].insert((name, hiddenLayerCount, dumps(hiddenLayerSizes), inputSize, outputSize, biasRange, weightRange, 0, str(highData), str(self.rounding)))
        self.recordStructure = highData
        self.dataBase.newTable("RecordedNeuralValues", ("position", "bias", "weights"))

        # table neurons
        self.dataBase.newTable("Neurons", ("LayerPosition", "Bias", "Weights"))
        layer = 0
        for i in self.layers:
            pos = 0
            if layer == 0:
                for n in i.neurons:
                    ins = (dumps((layer, pos)), "NULL", "NULL")
                    self.dataBase.table["Neurons"].insert(ins)
                    pos += 1
            else:
                for n in i.neurons:
                    ins = (dumps((layer, pos)), n.bias, dumps(n.connections))
                    self.dataBase.table["Neurons"].insert(ins)
                    pos += 1
            layer += 1

        # log starting structure
        self.updateNeurons(True)
    
    def existingInit(self):

        # get basic info
        info = self.dataBase.table["Info"].getAll()[0]
        # [name, hidden layer count, sizes for hidden layers, input size, output size, bias range, weight range, times trained]
        self.name = info[0]
        hiddenLayerCount = info[1]
        hiddenLayerSizes = loads(info[2])
        outputSize = info[4]
        biasRange = info[5]
        weightRange = info[6]
        self.timesTrained = info[7]
        self.recordStructure = info[8] == "True"
        self.rounding = int(info[9])
        self.weightRange = weightRange
        self.biasRange = biasRange

        # get neurons
        ns = self.dataBase.table["Neurons"].getAll()
        neurons = []
        # "Layer/Position", "Bias", "Weights"
        for i in ns:
            neuron = [w for w in i]
            neuron[0] = loads(neuron[0])
            if neuron[0][0] != 0:
                neuron[2] = loads(neuron[2])
                neurons.append(neuron)

        # init layer 1
        self.inputLayer = ILayer([0 for i in range(info[3])])
        self.hiddenLayers = []

        # init hidden layers
        ind = 1
        for ii in range(hiddenLayerCount):
            prevLay = self.inputLayer
            if self.hiddenLayers.__len__() > 0:
                prevLay = self.hiddenLayers[ind - 2]
            connectionLst = [i[2] for i in neurons if i[0][0] == ind]
            biasLst = [i[1] for i in neurons if i[0][0] == ind]
            nl = Layer(ind, hiddenLayerSizes[ind - 1], prevLay, weightRange, biasRange, self.rounding, connectionLst, biasLst)
            self.hiddenLayers.append(nl)
            ind += 1

        # init output layer
        connectionLst = [i[2] for i in neurons[-outputSize:]]
        biasLst = [i[1] for i in neurons[-outputSize:]]
        self.outputLayer = Layer(hiddenLayerCount + 1, outputSize, self.hiddenLayers[-1], weightRange, biasRange, self.rounding, connectionLst, biasLst)

        # define layers
        self.layers = [self.inputLayer]
        for i in self.hiddenLayers:
            self.layers.append(i)
        self.layers.append(self.outputLayer)

        # define layers with equations (all but input)
        self.eqLayers = self.hiddenLayers
        self.eqLayers.append(self.outputLayer)

        # define allEqNeurons in eqLayers
        self.allEqNeurons = []
        for i in self.eqLayers:
            self.allEqNeurons.extend(i.neurons)

        # define allNeurons in Layers
        self.allNeurons = []
        for i in self.layers:
            self.allNeurons.extend(i.neurons)
        
        # define allNeurons listed attributes
        self.allNeuronsData = [(l.layer, l.position, l.connections) for l in self.allEqNeurons]

        # define variable index {(dependant neur layer, dependant neur id, inependant neur id if applicable else None): value, ...}
        self.varIndex = {}
        for i in self.layers:
            if i.id == 0:
                continue
            self.varIndex.update(i.varIndex)
        
        # calculate neruons
        inpCurrentValues = []
        for i in self.layers[0].neurons:
            if i.value is not None:
                inpCurrentValues.append(float(i.value))

        # run through current values
        if len(inpCurrentValues) == len(self.layers[0].neurons):
            run = self.forwardPass(inpCurrentValues)

    def updateNeurons(self, recordLog=False):
        
        # empty current neuron structure
        self.dataBase.table["Neurons"].empty()

        layer = 0
        for i in self.layers:
            pos = 0
            if layer == 0:
                for n in i.neurons:
                    
                    # Only recording the amount of input layer neurons
                    ins = (dumps((layer, pos)), "NULL", "NULL") 
                    self.dataBase.table["Neurons"].insert(ins)
                    if self.recordStructure and recordLog:
                        self.dataBase.table["RecordedNeuralValues"].insert(ins)
                    pos += 1
            else:
                for n in i.neurons:

                    # recording all the info for regular neurons
                    ins = (dumps((layer, pos)), n.bias, dumps(n.connections))
                    self.dataBase.table["Neurons"].insert(ins)

                    # record current values if keeping a log
                    if self.recordStructure and recordLog:
                        self.dataBase.table["RecordedNeuralValues"].insert(ins)
                    pos += 1
            layer += 1
        
        info = [i for i in self.dataBase.table["Info"].getAll()[0]]
        info[7] = self.timesTrained
        self.dataBase.table["Info"].empty()
        self.dataBase.table["Info"].insert(info)

    def forwardPass(self, values, targets=[]):
        # set input layer values
        ind = 0
        for i in self.inputLayer.neurons:
            i.value = values[ind]
            ind += 1

        # set hidden layer 
        for ii in self.hiddenLayers:
            vals = []
            for i in ii.neurons:
                val = i.calculateValue()
                vals.append(val)
        outVals = []

        # set output layer
        for i in self.outputLayer.neurons:
            val = i.calculateValue()
            outVals.append(val)
        self.updateNeurons()

        # return loss if targets were given
        if len(targets) == 0:
            return outVals
        else:
            return sum((np.array(targets, dtype=np.float64) - np.array(outVals, dtype=np.float64))**2)

    def databaseBackup(self):

        # backup structure
        time = datetime.datetime.now().strftime("%m-%d-%y_%H-%M")
        flName = path.join("databases", self.name + "_backup_" + time + ".db")
        backup = db(flName)
        backup.printCmds = False
        backup.newTable("Info", ("name", "hlc", "hls", "inps", "os", "br", "wr", "tt", "rs", "da"))
        backup.newTable("Neurons", ("LayerPosition", "Bias", "Weights"))
        backup.newTable("RecordedNeuralValues", ("position", "bias", "weights"))
        backup.table["Info"].insertRows(self.dataBase.table["Info"].getAll())
        backup.table["Neurons"].insertRows(self.dataBase.table["Neurons"].getAll())
        backup.table["RecordedNeuralValues"].insertRows(self.dataBase.table["RecordedNeuralValues"].getAll())

    def backPass(self, input, targets, learningRate=.001):

        # run values
        self.forwardPass(input)

        # fetch weights and biases as np array
        varInds = np.array(list(self.varIndex.keys()))

        # ajustment function to be applied to each var
        for ind in varInds:
            # (dependant neur layer, dependant neur id, inependant neur id if applicable else None)
        
            # check if bias
            isBias = ind[2] is None

            # define tools for tracking x mult
            derMults = {(i.layer, i.position):0 for i in self.allNeurons}

            # set initial derivative
            neur = self.layers[ind[0]].neurons[ind[1]]

            if isBias:
                # der is activation of bias contianing neuron
                derMults[(neur.layer, neur.position)] = neur.activation
            else:
                # der is value of neuron it's value is applied to times activation
                derMults[(neur.layer, neur.position)] = self.layers[ind[0] - 1].neurons[ind[2]].value*neur.activation

            # array of connection lists organized by layers
            eqNeurArr = self.allNeuronsData
            
            # function to calulate value for each neuron, to be vectorized one layer at a time
            for neurData in eqNeurArr:

                #calculate the mult for each neuron
                cons = neurData[2]
                lay = neurData[0]
                prevMults = np.array([derMults[(lay - 1, i[0])] for i in cons])
                if sum(prevMults) == 0:
                    continue
                weights = np.array([i[1] for i in cons])
                multAdds = prevMults*weights
                derMults[(neurData[0], neurData[1])] = sum(multAdds)*self.layers[neurData[0]].neurons[neurData[1]].activation

            # gather info for final calc
            outputDerMults = np.array([derMults[(len(self.layers) - 1, i.position)] for i in self.layers[-1].neurons])
            
            # get current value of variable
            if isBias:
                varValue = neur.bias
            else:
                varValue = neur.connections[ind[2]][1]

            # find derivative adds
            outputValues = np.array([i.value for i in self.layers[-1].neurons])
            outputDerAdds = outputValues - (outputDerMults*varValue)
            outputDerAdds = targets - outputDerAdds
            outputDerAdds *= outputDerMults *2
            #outputDerAdds = np.array(targets) - outputDerAdds

            # final derivative
            outputDerMults **= 2
            outputDerMults *= 2
            derivative = sum(outputDerMults*varValue - outputDerAdds)

            # new value
            newVarValue = round(varValue - derivative*learningRate, self.rounding)

            # set new value
            if isBias:
                neur.bias = newVarValue
            else:
                neur.connections[ind[2]][1] = newVarValue
            self.varIndex[(ind[0], ind[1], ind[2])] = newVarValue
            
            # records history of network (if choosen)
            if self.recordStructure:
                self.dataBase.table["RecordedNeuralValues"].insert((dumps((ind[0], ind[1])), neur.bias, dumps(neur.connections)))

        # count training examples
        self.timesTrained += 1

        # update database
        self.updateNeurons()

    def getDataStructure(self):
        # fetches history of network
        return self.dataBase.table["RecordedNeuralValues"].getAll()

    def downloadHistory(self, filename):

        # downloads neuron history as spreadsheet
        # only reccomended for very small networks

        ss = excel_tools.spreadSheet(f"{filename}.xlsx")
        ss.insertRows([["Layer", "Position", "Bias", "Weights"]])
        total = len(self.getDataStructure())
        print(total)
        count = 0
        for ind, bias, weights in self.getDataStructure():
            ind = loads(ind)
            if ind[0] == 0:
                continue
            weights = loads(weights)
            add = ind
            add.append(bias)
            add.extend([i[1] for i in weights])
            ss.insertRows([add])
            count += 1
            if not count % 500:
                print(count)
