## Differential Evolution Solver Class
## Based on algorithms developed by Dr. Rainer Storn & Kenneth Price
## Written By: Lester E. Godwin
##             PushCorp, Inc.
##             Dallas, Texas
##             972-840-0208 x102
##             godwin@pushcorp.com
## Created: 6/8/98
## Last Modified: 6/8/98
## Revision: 1.0
##
## Ported To Python From C++ July 2002
## Ported To Python By: James R. Phillips
##                      Birmingham, Alabama USA
##                      zunzun@zunzun.com

true  = 1
false = 0

                      
stBest1Exp			= 0
stRand1Exp			= 1
stRandToBest1Exp	= 2
stBest2Exp			= 3
stRand2Exp			= 4
stBest1Bin			= 5
stRand1Bin			= 6
stRandToBest1Bin	= 7
stBest2Bin			= 8
stRand2Bin			= 9

#/*------Constants for self.RandomUniform()---------------------------------------*/
SEED = 3
IM1  = 2147483563
IM2  = 2147483399
AM   = (1.0/IM1)
IMM1 = (IM1-1)
IA1  = 40014
IA2  = 40692
IQ1  = 53668
IQ2  = 52774
IR1  = 12211
IR2  = 3791
NTAB = 32
NDIV = (1+IMM1/NTAB)
EPS  = 1.2e-7
RNMX = (1.0-EPS)


class DESolver:

    def __init__(self, dim, popSize):
        self.nDim          = dim
        self.nPop          = popSize
        self.generations   = 0
        self.strategy      = stRand1Exp
        self.scale         = 0.7
        self.probability   = 0.5
        self.bestEnergy    = 0.0
        self.trialSolution = [0.0] * self.nDim
        self.bestSolution  = [0.0] * self.nDim
        self.popEnergy	   = [0.0] * self.nPop
        self.population	   = [0.0] * (self.nPop * self.nDim)
        self.iv            = [0] * NTAB
        self.idum          = 0
        self.idum2         = 123456789
        self.iy            = 0
        self.randCount     = 0



    def Dimension(self):
        return self.nDim

    def Population(self):
        return self.nPop

    def Energy(self):
        return self.bestEnergy
    
    def Solution(self):
        return self.bestSolution

    def Generations(self):
        return self.generations

    def Setup(self, min, max, deStrategy, diffScale, crossoverProb):
        self.strategy	 = deStrategy
        self.scale		 = diffScale
        self.probability = crossoverProb
        
        for i in range(self.nPop):
            for j in range(self.nDim):
                #print i,j, self.nDim, len(self.population), i*self.nDim+j, len(min), len(max)
                self.population[i*self.nDim+j] = self.RandomUniform(min[j],max[j])
            self.popEnergy[i] = 1.0E20

        for i in range(self.nDim):
            self.bestSolution[i] = 0.0

        if self.strategy == stBest1Exp:
            self.calcTrialSolution = self.Best1Exp
        elif self.strategy == stRand1Exp:
            self.calcTrialSolution = self.Rand1Exp
        elif self.strategy == stRandToBest1Exp:
            self.calcTrialSolution = self.RandToBest1Exp
        elif self.strategy == stBest2Exp:
            self.calcTrialSolution = self.Best2Exp
        elif self.strategy == stRand2Exp:
            self.calcTrialSolution = self.Rand2Exp
        elif self.strategy == stBest1Bin:
            self.calcTrialSolution = self.Best1Bin
        elif self.strategy == stRand1Bin:
            self.calcTrialSolution = self.Rand1Bin
        elif self.strategy == stRandToBest1Bin:
            self.calcTrialSolution = self.RandToBest1Bin
        elif self.strategy == stBest2Bin:
            self.calcTrialSolution = self.Best2Bin
        else: #self.strategy == stRand2Bin:
            self.calcTrialSolution = self.Rand2Bin

    def Solve(self, maxGenerations):
        self.bestEnergy = 1.0E20
        bAtSolution = false

        for generation in range(maxGenerations):
            if bAtSolution == true:
                break
            for candidate in range(self.nPop):
                self.calcTrialSolution(candidate)
                trialEnergy, bAtSolution = self.EnergyFunction(self.trialSolution, bAtSolution)

                if trialEnergy < self.popEnergy[candidate]:
                    # New low for this candidate
                    self.popEnergy[candidate] = trialEnergy
                    for copyCount in range(self.nDim):
                        self.population[candidate*self.nDim + copyCount] = self.trialSolution[copyCount]

                    # Check if all-time low
                    if trialEnergy < self.bestEnergy:
                        self.bestEnergy = trialEnergy
                        for z in range(self.nDim):
                            self.bestSolution[z] = self.trialSolution[z]

        self.generations = generation
        return bAtSolution

    def Best1Exp(self, candidate):
        ##print "Best1Exp"
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,0,0,0)
        n = int(self.RandomUniform(0.0,float(self.nDim)))

        for copyCount in range(self.nDim):
            self.trialSolution[copyCount] = self.population[candidate*self.nDim+copyCount]
        i = 0
        while(1):
            if self.RandomUniform(0.0, 1.0) >= self.probability or i == self.nDim:
                break
            self.trialSolution[n] = self.bestSolution[n] \
                                    + self.scale * (self.population[r1*self.nDim+n] - self.population[r2*self.nDim+n])
            n = (n + 1) % self.nDim
            i += 1


    def Rand1Exp(self, candidate):
        ##print "Rand1Exp"
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,0,0)
        n = int(self.RandomUniform(0.0,float(self.nDim)))

        for copyCount in range(self.nDim):
            self.trialSolution[copyCount] = self.population[candidate*self.nDim+copyCount]
        i = 0
        while(1):
            if self.RandomUniform(0.0, 1.0) >= self.probability or i == self.nDim:
                break
            self.trialSolution[n] = self.population[r1*self.nDim+n] + self.scale * (self.population[r2*self.nDim+n] - self.population[r3*self.nDim+n])
            n = (n + 1) % self.nDim
            i += 1


    def RandToBest1Exp(self, candidate):
        ##print "RandToBest1Exp"
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,0,0,0)
        n = int(self.RandomUniform(0.0,float(self.nDim)))

        for copyCount in range(self.nDim):
            self.trialSolution[copyCount] = self.population[candidate*self.nDim+copyCount]
        i = 0
        while(1):
            if self.RandomUniform(0.0, 1.0) >= self.probability or i == self.nDim:
                break
            self.trialSolution[n] += self.scale * (self.bestSolution[n] - self.trialSolution[n]) \
                                     + self.scale * (self.population[r1*self.nDim+n] - self.population[r2*self.nDim+n])
            n = (n + 1) % self.nDim
            i += 1


    def Best2Exp(self, candidate):
        ##print "Best2Exp"
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,1,0)
        n = int(self.RandomUniform(0.0,float(self.nDim)))

        for copyCount in range(self.nDim):
            self.trialSolution[copyCount] = self.population[candidate*self.nDim+copyCount]
        i = 0
        while(1):
            if self.RandomUniform(0.0, 1.0) >= self.probability or i == self.nDim:
                break
            self.trialSolution[n] = self.bestSolution[n] + \
                                    self.scale * (self.population[r1*self.nDim+n] + \
                                                  self.population[r2*self.nDim+n] - \
                                                  self.population[r3*self.nDim+n] - \
                                                  self.population[r4*self.nDim+n])
            n = (n + 1) % self.nDim
            i += 1


    def Rand2Exp(self, candidate):
        ##print "Rand2Exp"
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,1,1)
        n = int(self.RandomUniform(0.0,float(self.nDim)))

        for copyCount in range(self.nDim):
            self.trialSolution[copyCount] = self.population[candidate*self.nDim+copyCount]
        i = 0
        while(1):
            if self.RandomUniform(0.0, 1.0) >= self.probability or i == self.nDim:
                break
            self.trialSolution[n] = self.population[r1*self.nDim+n] + \
                                                    self.scale * (self.population[r2*self.nDim+n] + \
                                                                  self.population[r3*self.nDim+n] - \
                                                                  self.population[r4*self.nDim+n] - \
                                                                  self.population[r5*self.nDim+n])
            n = (n + 1) % self.nDim
            i += 1


    def Best1Bin(self, candidate):
        ##print "Best1Bin"
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,0,0,0)
        n = int(self.RandomUniform(0.0,float(self.nDim)))

        for copyCount in range(self.nDim):
            self.trialSolution[copyCount] = self.population[candidate*self.nDim+copyCount]
        i = 0
        while(1):
            if self.RandomUniform(0.0, 1.0) >= self.probability or i == self.nDim:
                break
            self.trialSolution[n] = self.bestSolution[n] + \
                                    self.scale * (self.population[r1*self.nDim+n] - \
                                                  self.population[r2*self.nDim+n])
            n = (n + 1) % self.nDim
            i += 1


    def Rand1Bin(self, candidate):
        ##print "Rand1Bin"
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,0,0)
        n = int(self.RandomUniform(0.0,float(self.nDim)))

        for copyCount in range(self.nDim):
            self.trialSolution[copyCount] = self.population[candidate*self.nDim+copyCount]
        i = 0
        while(1):
            if self.RandomUniform(0.0, 1.0) >= self.probability or i == self.nDim:
                break
            self.trialSolution[n] = self.population[r1*self.nDim+n] + \
                                    self.scale * (self.population[r2*self.nDim+n] -\
                                                  self.population[r3*self.nDim+n])
            n = (n + 1) % self.nDim
            i += 1


    def RandToBest1Bin(self, candidate):
        ##print "RandToBest1Bin"
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,0,0,0)
        n = int(self.RandomUniform(0.0,float(self.nDim)))

        for copyCount in range(self.nDim):
            self.trialSolution[copyCount] = self.population[candidate*self.nDim+copyCount]
        i = 0
        while(1):
            if self.RandomUniform(0.0, 1.0) >= self.probability or i == self.nDim:
                break
            self.trialSolution[n] += self.scale * (self.bestSolution[n] - self.trialSolution[n]) + \
                                     self.scale * (self.population[r1*self.nDim+n] - \
                                                   self.population[r2*self.nDim+n])
            n = (n + 1) % self.nDim
            i += 1


    def Best2Bin(self, candidate):
        ##print "Best2Bin"
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,1,0)
        n = int(self.RandomUniform(0.0,float(self.nDim)))

        for copyCount in range(self.nDim):
            self.trialSolution[copyCount] = self.population[candidate*self.nDim+copyCount]
        i = 0
        while(1):
            if self.RandomUniform(0.0, 1.0) >= self.probability or i == self.nDim:
                break
            self.trialSolution[n] = self.bestSolution[n] + \
                                    self.scale * (self.population[r1*self.nDim+n] + \
                                                  self.population[r2*self.nDim+n] - \
                                                  self.population[r3*self.nDim+n] - \
                                                  self.population[r4*self.nDim+n])
            n = (n + 1) % self.nDim
            i += 1


    def Rand2Bin(self, candidate):
        ##print "Rand2Bin"
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,1,1,1)
        n = int(self.RandomUniform(0.0,float(self.nDim)))

        for copyCount in range(self.nDim):
            self.trialSolution[copyCount] = self.population[candidate*self.nDim+copyCount]
        i = 0
        while(1):
            if self.RandomUniform(0.0, 1.0) >= self.probability or i == self.nDim:
                break
            self.trialSolution[n] = self.population[r1*self.nDim+n] + \
                                    self.scale * (self.population[r2*self.nDim+n] + \
                                                  self.population[r3*self.nDim+n] - \
                                                  self.population[r4*self.nDim+n] - \
                                                  self.population[r5*self.nDim+n])
            n = (n + 1) % self.nDim
            i += 1


    def SelectSamples(self, candidate, r1, r2, r3, r4, r5):
        if r1:
            while(1):
                r1 = int(self.RandomUniform(0.0, float(self.nPop)))
                if r1 != candidate:
                    break
        if r2:
            while(1):
                r2 = int(self.RandomUniform(0.0, float(self.nPop)))
                if r2 != candidate and r2 != r1:
                    break
        if r3:
            while(1):
                r3 = int(self.RandomUniform(0.0, float(self.nPop)))
                if r3 != candidate and r3 != r1 and r3 != r2:
                    break
        if r4:
            while(1):
                r4 = int(self.RandomUniform(0.0, float(self.nPop)))
                if r4 != candidate and r4 != r1 and r4 != r2 and r4 != r3:
                    break
        if r5:
            while(1):
                r5 = int(self.RandomUniform(0.0, float(self.nPop)))
                if r5 != candidate and r5 != r1 and r5 != r2 and r5 != r3 and r5 != r4:
                    break

        return r1, r2, r3, r4, r5



    def RandomUniform(self, minValue, maxValue):

        if self.iy == 0:
            self.idum = SEED

        if self.idum <= 0:
            if -self.idum < 1:
                self.idum = 1
            else:
                self.idum = -self.idum

            self.idum2 = self.idum

            for j in range(NTAB+7, -1, -1):
                k = self.idum / IQ1
                self.idum = IA1 * (self.idum - k*IQ1) - k*IR1
                if self.idum < 0:
                    self.idum += IM1
                if j < NTAB:
                    self.iv[j] = self.idum

            self.iy = self.iv[0]

        k = self.idum / IQ1
        self.idum = IA1 * (self.idum - k*IQ1) - k*IR1

        if self.idum < 0:
            self.idum += IM1

        k = self.idum2 / IQ2
        self.idum2 = IA2 * (self.idum2 - k*IQ2) - k*IR2

        if self.idum2 < 0:
            self.idum2 += IM2

        j = self.iy / NDIV
        self.iy = self.iv[j] - self.idum2
        self.iv[j] = self.idum

        if self.iy < 1:
            self.iy += IMM1

        result = AM * self.iy

        if result > RNMX:
            result = RNMX
        else:
            result = minValue + result * (maxValue - minValue)

        self.randCount += 1
        return result
    