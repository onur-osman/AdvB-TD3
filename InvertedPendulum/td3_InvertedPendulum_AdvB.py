import numpy as np
import sys
import gym
import torch
import random
from td3 import TD3
from buffer import ExperienceReplay
import matplotlib.pyplot as plt
import copy


def main():
    env_text = 'InvertedPendulum-v4'
    save_version = 'V1'
    env = gym.make(env_text)
    # Uncomment to use GPU, but errors exist if GPU is not supported anymore.
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # set seed for reproducable results
    #seed = 1
    #env.seed(seed)
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    #random.seed(seed)
    loww = torch.FloatTensor(env.action_space.low).to(device)
    highh = torch.FloatTensor(env.action_space.high).to(device)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    buffer_size = 1000000
    batch_size = 100
    noise = 0.1

    # AdvB parameters
    DQN_models = []
    n_models = 10
    discard_period = 100  # episode for discarding random members
    save_models_period = 10
    test_period = 10
    n_best_model = 10
    n_random_model = n_models - n_best_model
    formation_period = 10000
    save_test_period = 10

    policy = TD3(state_dim, action_dim, max_action, env, device)

    # # AdvB member initialization
    for kkk in range(n_models):
        model_X = copy.deepcopy(policy.actor)
        model_X.load_state_dict(policy.actor.state_dict())
        DQN_models += [
            {'model': model_X, 'best_total_reward': -5000, 'last_total_reward': -5000, 'n_trials': 0, 'n_wins': 0}]

    try:
        print("Loading previous model")
        policy.load()
    except Exception as e:
        print('No previous model to load. Training from scratch.')

    buffer = ExperienceReplay(buffer_size, batch_size, device)

    save_score = 5000
    episodes = 1001
    timesteps = 2000

    best_reward = -1*sys.maxsize
    scores_over_episodes = []
    timesteps_over_episodes = []

    total_reward_adv_coop_list = []
    total_reward_list = []
    total_reward_test_list = []
    current_timestep = 0

    for episode in range(episodes):
        avg_reward = 0
        state = env.reset()[0]
        done = False
        for i in range(timesteps):
            current_timestep += 1
            if episode < formation_period:
                # Same as the TD3, select an action and add noise:
                action = policy.select_action(state) + np.random.normal(0, max_action * noise, size=action_dim)
                action = torch.FloatTensor(action).to(device)
                action = action.clip(loww, highh)
                if device.type == 'cuda':
                    next_state, reward, terminate, truncate, _ = env.step(action.cpu().numpy())
                else:
                    next_state, reward, terminate, truncate, _ = env.step(action.detach().numpy())

            else:
                possible_reward = [0]*n_models
                for k in range(n_models):
                    actor_member = DQN_models[k]['model']
                    state = torch.FloatTensor(state).to(device)
                    action = actor_member(state)
                    action = action.clip(loww, highh)
                    curr_q1, curr_q2 = policy.critic(state.view(1,-1), action.view(1,-1))
                    possible_reward[k] = float(torch.min(curr_q1, curr_q2))
                best_member_id = np.argmax(possible_reward)
                best_member = DQN_models[best_member_id]['model']
                action = best_member(state).detach().numpy() + np.random.normal(0, max_action * noise, size=action_dim)
                action = torch.FloatTensor(action).to(device)
                action = action.clip(loww, highh)
                next_state, reward, terminate, truncate, _ = env.step(action.detach().numpy())
            avg_reward += reward

            if terminate or truncate > 0:
                done = True

            if device.type == 'cuda':
                buffer.store_transition(state, action.cpu().numpy(), reward, next_state, done)
            else:
                buffer.store_transition(state, action.detach().numpy(), reward, next_state, done)

            state = next_state

           #env.render()
            if len(buffer) > batch_size:
                policy.train(buffer, i)

            if truncate:
                scores_over_episodes.append(avg_reward)
                #solved_text = 'SOLVED'
                #print('Episode: {}, finished with reward:{:.2f}, Finished at timestep:{}, solved:{}'.format(episode, avg_reward,i, solved_text))
                break
            elif terminate:
                #solved_text = 'not_solved'
                scores_over_episodes.append(avg_reward)
                #print('Episode: {}, finished with reward:{:.2f}, Finished at timestep:{}, solved:{}'.format(episode, avg_reward, i, solved_text))
                break
        timesteps_over_episodes.append(current_timestep)

        '''
        if(np.mean(scores_over_episodes[-50:]) > save_score):
            print('Saving agent- past 50 scores gave better avg than ', save_score)
            best_reward = np.mean(scores_over_episodes[-50:])
            save_score = best_reward
            policy.save()
            break # Saved agent. Break out of episodes and end, 400 is pretty good. 

        if(episode >= 0 and avg_reward > best_reward):
            print('Saving agent- score for this episode was better than best-known score..')
            best_reward = avg_reward
            policy.save() # Save current policy + optimizer
            torch.save(policy.actor.state_dict(), './walker2d_trained_actor.pth')
            torch.save(policy.critic.state_dict(), './walker2d_critic.pth')
            for k in range(n_models):
                if k < 10:
                    f_name = './walker2d_actor_0' + str(k) + '.pth'
                else:
                    f_name = './walker2d_actor_' + str(k) + '.pth'
                torch.save(DQN_models[k]['model'].state_dict(), f_name)
        '''

        # TEST
        # Test 1: Test the actor
        test_reward = 0
        state = env.reset()[0]
        for i in range(timesteps):
            action = policy.select_action(state)
            action = torch.FloatTensor(action).to(device)
            action = action.clip(loww, highh)
            if device.type == 'cuda':
                next_state, reward, terminate, truncate, _ = env.step(action.cpu().numpy())
            else:
                next_state, reward, terminate, truncate, _ = env.step(action.detach().numpy())
            state = next_state
            test_reward += reward
            # env.render()
            if truncate:
                solved_text = 'SOLVED'
                #print('TEST_1: finished with reward:{:.2f}, Finished at timestep:{}, Test:{}'.format(test_reward, i,
                #                                                                               solved_text))
                break
            elif terminate:
                solved_text = 'not_solved'
                #print('TEST_1: finished with reward:{:.2f}, Finished at timestep:{}, Test:{}'.format(test_reward, i,
                #                                                                               solved_text))
                break

        '''
        # Test 2: Test the members
        if n_random_model > 0:
            member_test_reward = [0.] * n_models
            for k in range(n_models):
                DQN_models[k]['n_trials'] += 1
                my_n_wins = DQN_models[k]['n_wins']
                state = env.reset()[0]
                for i in range(timesteps):
                    actor_member = DQN_models[k]['model']
                    state = torch.FloatTensor(state).to(device)
                    action = actor_member(state)
                    #action = torch.FloatTensor(action).to(device)
                    #action = action.clip(loww, highh)
                    if device.type == 'cuda':
                        next_state, reward, terminate, truncate, _ = env.step(action.detach().cpu().numpy())
                    else:
                        next_state, reward, terminate, truncate, _ = env.step(action.detach().numpy())
                    state = next_state
                    member_test_reward[k] += reward
                    #env.render()
                    if truncate:
                        #solved_text = 'SOLVED'
                        #print('TEST_2: finished with reward:{}, Finished at timestep:{}, Test:{}'.format(test_reward, i, solved_text))
                        break
                    elif terminate:
                        #solved_text = 'not_solved'
                        #print('TEST_2: finished with reward:{}, Finished at timestep:{}, Test:{}'.format(test_reward, i, solved_text))
                        break
                DQN_models[k]['last_total_reward'] = member_test_reward[k]
            #member_test_reward = float(member_test_reward)
            #print(('TEST_2: {:.2f}, '*len(member_test_reward)).format(*member_test_reward))
        '''

        # Test 3: Test Cooperative decision: for each action a member is chosen whose critic is the highest
        if episode % test_period == 0:
            advb_test_reward = 0
            state = env.reset()[0]
            #state = state.detach().cpu().numpy()
            for i in range(timesteps):
                member_critics = [0] * n_models
                state = torch.FloatTensor(state).to(device)
                for k in range(n_models):
                    actor_member = DQN_models[k]['model']
                    action = actor_member(state)
                    action = action.clip(loww, highh)
                    curr_q1, curr_q2 = policy.critic(state.view(1, -1), action.view(1, -1))
                    member_critics[k] = float(torch.max(curr_q1, curr_q2))
                best_member_id = np.argmax(member_critics)
                best_member = DQN_models[best_member_id]['model']
                action = best_member(state)

                action = action.clip(loww, highh)
                if device.type == 'cuda':
                    next_state, reward, terminate, truncate, _ = env.step(action.detach().cpu().numpy())
                else:
                    next_state, reward, terminate, truncate, _ = env.step(action.detach().numpy())
                state = next_state
                advb_test_reward += reward
                # env.render()
                if truncate:
                    # solved_text = 'SOLVED'
                    # print('TEST_2: finished with reward:{}, Finished at timestep:{}, Test:{}'.format(test_reward, i, solved_text))
                    break
                elif terminate:
                    # solved_text = 'not_solved'
                    # print('TEST_2: finished with reward:{}, Finished at timestep:{}, Test:{}'.format(test_reward, i, solved_text))
                    break
        # member_test_reward = float(member_test_reward)
        #print('TEST_3: {:.2f}, '.format(advb_test_reward))

        # EVALUATION
        evaluation_list = []
        for nn_models in range(n_models):
            evaluation_list.append(DQN_models[nn_models]['last_total_reward'])
        best_model_index = np.argmax(np.array(evaluation_list))
        if episode > formation_period:
            DQN_models[best_model_index]['n_wins'] += 1
        #print('Episode: {}, finished with reward:{:.2f}, Training Network Reward: {:.2f}, Advisory Board Reward: {:.2f}, Adv Cooperative Decision: {:.2f}'
        #      .format(episode, avg_reward, test_reward, DQN_models[best_model_index]['last_total_reward'], advb_test_reward))

        total_reward_adv_coop_list.append(advb_test_reward)
        total_reward_list.append(DQN_models[best_model_index]['last_total_reward'])
        total_reward_test_list.append(test_reward)
        coopNet_averaged = np.mean(np.array(total_reward_adv_coop_list)[-100:])

        # Discard in Best Models
        for nn_best_models in range(n_best_model):
            if test_reward > DQN_models[nn_best_models]['best_total_reward']:
                model_X = copy.deepcopy(policy.actor)
                model_X.load_state_dict(policy.actor.state_dict())
                New_model = {'model': model_X, 'best_total_reward': test_reward, 'last_total_reward': 0,
                             'n_trials': 1, 'n_wins': 1}
                #my_model = DQN_models[nn_best_models]['model']
                #my_model.load_state_dict(policy.actor.state_dict())
                #DQN_models[nn_best_models]['best_total_reward'] = test_reward
                #DQN_models[nn_best_models]['last_total_reward'] = test_reward
                #DQN_models[nn_best_models]['n_trials'] = 1
                #DQN_models[nn_best_models]['n_wins'] = 1
                #model_X = copy.deepcopy(policy.actor)
                #New_model = {'model': model_X, 'best_total_reward': test_reward,
                #             'last_total_reward': test_reward, 'n_trials': 1, 'n_wins': 1}
                DQN_models.insert(nn_best_models, New_model)
                DQN_models.pop(n_best_model)
                break
        # Discard worst model in randomly selected models
        if n_random_model > 0 and episode % discard_period == 0:
            dqn_performance = []
            for nn_random_models in range(n_best_model, n_models):
                dqn_performance.append(
                    DQN_models[nn_random_models]['n_wins'] / DQN_models[nn_random_models]['n_trials'])
            worst_model_index = np.argmin(np.array(dqn_performance))
            my_model = DQN_models[n_best_model + worst_model_index]['model']
            my_model.load_state_dict(policy.actor.state_dict())
            #DQN_models[n_best_model + worst_model_index]['model'] = copy.deepcopy(policy.actor)
            DQN_models[n_best_model + worst_model_index]['best_total_reward'] = 0
            DQN_models[n_best_model + worst_model_index]['last_total_reward'] = test_reward
            DQN_models[n_best_model + worst_model_index]['n_trials'] = 0
            DQN_models[n_best_model + worst_model_index]['n_wins'] = 0

        dqn_performance = []
        dqn_performance_wins = []
        dqn_performance_trials = []
        for nn_random_models in range(n_models):
            dqn_performance_wins.append(DQN_models[nn_random_models]['n_wins'])
            dqn_performance_trials.append(DQN_models[nn_random_models]['n_trials'])

        if episode % save_test_period == 0:
            print(
                'Episode: {}, finished with reward:{:.2f}, Training Network Reward: {:.2f}, Advisory Board Reward: {:.2f}, Adv Cooperative Decision: {:.2f}'
                .format(episode, avg_reward, test_reward, DQN_models[best_model_index]['last_total_reward'],
                        advb_test_reward))

            try:
                with open(env_text+'_Training_'+save_version+'.txt', 'w') as fp:
                    fp.write('\n'.join(str(item) for item in scores_over_episodes))
                with open(env_text+'_ActorTest_'+save_version+'.txt', 'w') as fp:
                    fp.write('\n'.join(str(item) for item in total_reward_test_list))
                with open(env_text+'_Cooperative_'+save_version+'.txt', 'w') as fp:
                    fp.write('\n'.join(str(item) for item in total_reward_adv_coop_list))
                with open(env_text+'_timesteps_'+save_version+'.txt', 'w') as fp:
                    fp.write('\n'.join(str(item) for item in timesteps_over_episodes))

                torch.save(policy.actor.state_dict(), './' + env_text + '_trained_actor.pth')
                torch.save(policy.critic.state_dict(), './' + env_text + '_critic.pth')
                for k in range(n_models):
                    if k < 10:
                        f_name = './' + env_text + '_actor_finalized_0' + str(k) + '.pth'
                    else:
                        f_name = './' + env_text + '_actor_finalized_' + str(k) + '.pth'
                    torch.save(DQN_models[k]['model'].state_dict(), f_name)
            except:
                print('something went wrong')

    fig = plt.figure()
    plt.plot(np.arange(1, len(scores_over_episodes) + 1), scores_over_episodes)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    torch.save(policy.actor.state_dict(), './'+env_text+'_trained_actor.pth')
    torch.save(policy.critic.state_dict(), './'+env_text+'_critic.pth')
    for k in range(n_models):
        if k < 10:
            f_name = './'+env_text+'_actor_finalized_0' + str(k) + '.pth'
        else:
            f_name = './'+env_text+'_actor_finalized_' + str(k) + '.pth'
        torch.save(DQN_models[k]['model'].state_dict(), f_name)


main()