import random, math

class Traces:
	"""
	Handles up to "maxe" traces to reduce
	computational complexity
	"""

	def __init__(self, e, maxe, minTraceValue):
		self.toTraces = [0]*len(e)
		self.e = e
		self.maxTraces = maxe
		self.toOrig = [0]*self.maxTraces
		self.minTraceValue = minTraceValue

	def init(self):
		"""
		Initializes traces for new episode
		"""
		for i in xrange(len(self.e)):
			self.e[i] = 0.0
		for i in xrange(len(self.e)):
			self.toTraces[i] = -1
		self.traces = 0

	def position(self, tracenumber):
		"""
		Return's original position.
		"""
		return self.toOrig[tracenumber]

	def count(self):
		return self.traces
	
	def multiply(self, factor):
		"""
		Multiplies traces with a given factor. If trace falls below
		minTraceValue treshold then remove it.
		"""

		for i in xrange(self.traces):
			self.e[self.toOrig[i]] *= factor

		i = 0
		while i < self.traces:
			if self.e[self.toOrig[i]] < self.minTraceValue:
				self.removeN(i)
				i -= 1
			i += 1
	
	def remove(self, origl):
		"""
		Remove a trace if trace is present. Original location on input
		"""
		i = self.toTraces[origl]
		if i != -1:
			self.removeN(i)

	def removeN(self, i):
		"""
		Removes a trace. Trace number on input.
		"""
		self.traces -= 1
	
		#delete the inverse
		self.toTraces[self.toOrig[i]] = -1

		self.e[self.toOrig[i]] = 0
		
		if i != self.traces:
			#put last on this spot
			self.toOrig[i] = self.toOrig[self.traces]
			#change inverse of the moved trace
			self.toTraces[self.toOrig[i]] = i
		

	def full(self, which):
		"""
		Set a trace to 1.0. If trace is not present, add it. 
		If maximum trace number is exceeded, replace a first trace.
		Parameter is in original order.
		TODO: change with random trace, when it works
		"""
		self.e[which] = 1.0

		if self.toTraces[which] == -1:

			if self.traces == self.maxTraces:
				self.removeN(0)

			self.toOrig[self.traces] = which
			self.toTraces[which] = self.traces

			self.traces += 1

class Tiles:
	"""
	Simple tiles
	"""
	def __init__(self, memorySize, numTilings, rand=random.Random(0)):
		self.t = TilesO(rand=rand)
		self.mem = memorySize
		self.num = numTilings

	def getTiles(self, variables):
		tiles = [0]*self.num
		self.t.getTiles(tiles, len(tiles), variables, len(variables), self.mem)
		return tiles

class TilesO:

	"""
	Tiles
	Author: ece.uhn.edu
	"""

	def __init__(self, rand=random.Random(0)):
		"""
		initialize hashing
		"""
		self.rndseq = [ rand.randint(0, 2000000000) for i in xrange(2048) ]

		# for testing
		# self.rndseq = [ i*999 for i in xrange(2048) ]
		
	#        int[]-OUT  int          double[]   int            int          int       int       int
	def getTiles(self, tiles, num_tilings, variables, num_variables, memory_size, hash1=-1, hash2=-1, hash3=-1):
		qstate = [0]*(num_variables)
		base = [0]*(num_variables)
		num_coordinates = 0
		coordinates = [0]*(num_variables+4)

                if hash1 == -1:
                        num_coordinates = num_variables + 1
                elif hash2 == -1:
                        num_coordinates = num_variables + 2
                        coordinates[num_variables + 1] = hash1
                elif hash3 == -1:
                        num_coordinates = num_variables + 3
                        coordinates[num_variables + 1] = hash1
                        coordinates[num_variables + 2] = hash2
                else:
                        num_coordinates = num_variables + 4
                        coordinates[num_variables + 1] = hash1
                        coordinates[num_variables + 2] = hash2
                        coordinates[num_variables + 3] = hash3
                
                # quantize state to integers (henceforth, tile widths == num_tilings) 
		for i in xrange(num_variables):
                        qstate[i] = (int)(math.floor(variables[i] * num_tilings))
                        base[i] = 0

		#compute the tile numbers
		for j in xrange(num_tilings):

                        # loop over each relevant dimension 
			for i in xrange(num_variables):

                                # find coordinates of activated tile in tiling space
                                if qstate[i] >= base[i]:
                                        coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % num_tilings)
                                else:
                                        coordinates[i] = qstate[i] + 1  + ((base[i] - qstate[i] - 1) % num_tilings) - num_tilings

                                # compute displacement of next tiling in quantized space 
                                base[i] += (1 + (2 * i))
                        
			
                        # add additional indices for tiling and hashing_set so they hash differently
                        coordinates[num_variables] = j

                        tiles[j] = self.hash_coordinates(coordinates, num_coordinates,  memory_size)
	

	def hash_coordinates(self, coordinates, num_indices, memory_size):
		"""	
		Takes an array of integer coordinates and returns the corresponding tile after hashing 
		"""
		
		sum = 0
		for i in xrange(num_indices):
			# add random table offset for this dimension and wrap around */
			index = coordinates[i]
			index += (449 * i)
			index %= 2048
			while index < 0:
				index += 2048
				
			# add selected random number to sum */
			sum += self.rndseq[index]
		
		index = sum % memory_size

		while index < 0:
			 index += memory_size
		
		return index


class RLSarsa:


	def __init__(self, actions, numTilings, memorySize=1000, maxtraces=100, mintracevalue=0.01, rand=random.Random(0)):
	
		self.N = memorySize
		self.M = actions
		self.TILINGS = numTilings
		
		self.theta = [0.0]*self.N  #learning function
		self.e = [0.0]*self.N  #traces
		
		self.Q = [0.0]*self.M  #action-state values
		self.tileLocations = [[0]*self.TILINGS for i in range(self.M)] # properties, one for each action
		
		self.tiles = TilesO()
		self.traces = Traces(self.e, maxtraces, mintracevalue)

		self.epsilon = 0.05
		self.alpha = 0.5
		self.lambda1 = 0.9
		self.gamma = 0.97

		self.rand = rand

	def loadQ(self):
		"""
		Loads action-state function values
		"""
		for a in xrange(self.M):
			self.loadQa(a)
	
	def loadQa(self, a):
		"""
		Loads action-state function values for action a.
		"""
		self.Q[a] = 0
		for j in xrange(self.TILINGS):
			self.Q[a] += self.theta[self.tileLocations[a][j]]
	
	def loadTileLocations(self, state_vars):
		
		zacasno = [0]*self.TILINGS
		
		for a in xrange(self.M):
	
			self.tiles.getTiles(zacasno, len(zacasno), state_vars, len(state_vars), self.N, a)
	
			for i in xrange(len(zacasno)):
				self.tileLocations[a][i] = zacasno[i]			


	def maxElementPosition(self, array): 
		"""
		Returns position of element with the highest value
		in an array. If there are more elements with the
		highest value, one is selected at random.
		"""
		
		maxElement = 0
		maxValue = array[0]

		num_ties = 1

		for a in xrange(len(array)):
			
			value = array[a]

			if value >= maxValue:
				if value > maxValue:
					maxValue = value
					maxElement = a
				else:
					num_ties += 1
					if 0 == self.rand.randint(0,num_ties-1):
						maxValue = value
						maxElement = a
					#maxElement = a # TODO - brisi - tu je zato, da je deterministicno
		return maxElement

	def tracesFallAndSet(self):
		
		#traces fall
		self.traces.multiply(self.gamma*self.lambda1)
		
		#remove traces for other actions
		for a in xrange(self.M):
			if a != self.choosenAction:
				for j in xrange(self.TILINGS):
					self.traces.remove(self.tileLocations[a][j])
				

		#add or reset traces
		for j in xrange(self.TILINGS):
			self.traces.full(self.tileLocations[self.choosenAction][j])
	
	def chooseActionP(self):

		self.choosenAction = self.maxElementPosition(self.Q)

		#with probability epsilon choose random action
		if (self.withProbability(self.epsilon)):
			self.choosenAction = self.rand.randint(0,self.M-1)
	
	def withProbability(self, p):
		return p > self.rand.random()
	
	
	def init(self, state_vars):
	
		sv = [float(a) for a in state_vars]
		
		self.traces.init()
		
		#choose first action
		self.loadTileLocations(sv);
		self.loadQ();
		self.chooseActionP();
		
		#set first traces;
		self.tracesFallAndSet();

		return self.choosenAction
	
	def decide(self, reward, state_vars):
		"""
		Function learns from previous actions and decides.
		""" 

		reward = float(reward)
			
		sv = [float(a) for a in state_vars]
	
		delta = reward - self.Q[self.choosenAction]

		#choose next action
		self.loadTileLocations(sv)
		self.loadQ()
		self.chooseActionP()
			
		#delta = reward - Q[previous] + gamma * Q[now]
		delta += self.gamma * self.Q[self.choosenAction]

		#learn
		temp = (self.alpha / self.TILINGS) * delta
		position = 0
		for i in xrange(self.traces.count()):
			position = self.traces.position(i)
			self.theta[position] += (temp * self.e[position])
		
		self.loadQa(self.choosenAction)
		
		self.tracesFallAndSet()
		
		return self.choosenAction
			

