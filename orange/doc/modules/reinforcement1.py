import orngReinforcement

r = orngReinforcement.RLSarsa(2,1)

ans = r.init([0,0]) #initialize episode

#if state is (0,0), act 0
for i in range(10):
  if ans == 0: reward = 1	
  else: reward = 0
  ans = r.decide(reward,[0,0])

#if state is (1,1), act 1
for i in range(10):
  if ans == 1: reward = 1 
  else: reward = 0
  ans = r.decide(reward,[1,1])

r.epsilon = 0.0 #no random action
r.alpha = 0.0 #stop learning

print "in (0,0) do", r.decide(0,[0,0]) #should output 0
print "in (1,1) do", r.decide(0,[1,1]) #should output 1
