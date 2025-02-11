import matplotlib.pyplot as plt
import time

total_reward_list = []
total_reward_test_list = []
total_reward_coop_list = []
timesteps = []

#env_text = 'MountainCarContinuous-v0'; max_reward = 100; min_reward = -50
#env_text = 'HalfCheetah-v4'; max_reward = 5000; min_reward = -500
#env_text = 'BipedalWalker-v3'; max_reward = 310; min_reward = -200
#env_text = 'Hopper-v4'; max_reward = 5000; min_reward = -500
#env_text = 'Swimmer-v4'; max_reward = 400; min_reward = 0
#env_text = 'Pendulum-v1'; max_reward = 50; min_reward = -1500
#env_text = 'InvertedPendulum-v4'; max_reward = 1500; min_reward = 0
#env_text = 'InvertedDoublePendulum-v4'; max_reward = 10000; min_reward = 0
env_text = 'Humanoid-v4'; max_reward = 5500; min_reward = 0
save_version = 'V1'

with open(env_text+'_Training_'+save_version+'.txt', 'r') as fp:
    for line in fp:
        line = line.strip()
        line = float(line)
        total_reward_list.append(line)

with open(env_text+'_ActorTest_'+save_version+'.txt', 'r') as fp:
    for line in fp:
        line = line.strip()
        line = float(line)
        total_reward_test_list.append(line)

with open(env_text+'_Cooperative_'+save_version+'.txt', 'r') as fp:
    for line in fp:
        line = line.strip()
        line = float(line)
        total_reward_coop_list.append(line)

with open(env_text+'_timesteps_'+save_version+'.txt', 'r') as fp:
    for line in fp:
        line = line.strip()
        line = float(line)
        timesteps.append(line)
plt.figure(dpi=300)
plt.subplot(3,1,1)
plt.title('Trained Network Averaged Return', fontsize=8)
plt.xlabel('Timesteps', fontsize=8)
plt.ylabel('Total Reward', fontsize=8)
#plt.plot(range(len(total_reward_list)), total_reward_list)
plt.scatter(timesteps, total_reward_list, s=0.2)
plt.grid()
plt.ylim(min_reward, max_reward)
plt.ticklabel_format(style='sci', axis='x', scilimits=(6,6))
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

plt.subplot(3,1,2)
plt.title('Actor Network Test Results',fontsize=8)
plt.xlabel('Timesteps', fontsize=8)
#plt.ylabel('Total Reward')
#plt.plot(range(len(total_reward_test_list)), total_reward_test_list)
plt.scatter(timesteps, total_reward_test_list, s=0.2)
plt.grid()
plt.ylim(min_reward,max_reward)
plt.ticklabel_format(style='sci', axis='x', scilimits=(6,6))
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

plt.subplot(3,1,3)
plt.title('Advisory Board Averaged Return',fontsize=8)
plt.xlabel('Timesteps', fontsize=8)
#plt.ylabel('Total Reward')
#plt.plot(range(len(total_reward_test_list)), total_reward_test_list)
plt.scatter(timesteps, total_reward_coop_list, s=0.2)
plt.grid()
plt.ylim(min_reward,max_reward)
plt.ticklabel_format(style='sci', axis='x', scilimits=(6,6))
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

plt.suptitle(' Advisory Board performance of '+env_text, fontsize=15)
plt.show()
