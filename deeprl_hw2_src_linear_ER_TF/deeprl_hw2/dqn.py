import numpy as np
import matplotlib.pyplot as plt


"""Main DQN agent."""

class DQNAgent:
    """Class implementing DQN.
    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.
    Feel free to change the functions and funciton parameters that the
    class provides.
    We have provided docstrings to go along with our suggested API.
    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 target_q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 num_actions):
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.num_actions = num_actions


    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.
        This is inspired by the compile method on the
        keras.models.Model class.
        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.
        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        self.q_network.compile(loss=loss_func,optimizer=optimizer)
        self.target_q_network.compile(loss=loss_func,optimizer=optimizer)

    def calc_q_values(self, state):
        """
        Given a preprocessed state (or batch of states) calculate the Q-values.
        Basically run your network on these states.
        Return
        ------
        Q-values for the state(s)
        """

        #How many states???
        #q_values = np.zeros(...,self.policy.num_actions)

        #iterate through states
        #  q_values[sIdx] = self.q_network.predict(state[sIdx])

        #return q_values

        return self.q_network.predict(state)

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.
        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.
        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.
        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.
        This would also be a good place to call
        process_state_for_network in your preprocessor.
        Returns
        --------
        selected action
        """

        #get q-values
        #calc_q_values(state)
        q_vals = calc_q_values(state)
        return self.policy.select_action(q_vals)


    def update_policy(self):
        """Update your policy.
        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.
        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.
        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """

        #get minibatch
        minibatch = self.memory.sample(self.batch_size)
        minibatch = self.preprocessor.process_batch(minibatch)

        #init state inputs + state targets

        exampleState = minibatch[0].s_t
        inputStates = np.zeros((self.batch_size, exampleState.shape[1], exampleState.shape[2], exampleState.shape[3]))
        targets = np.zeros((self.batch_size,1,self.num_actions))

        #make compute state inputs and targets
        for sampleIdx in range(self.batch_size):
          s_t=minibatch[sampleIdx].s_t
          a_t=minibatch[sampleIdx].a_t   #This is action index
          r_t=minibatch[sampleIdx].r_t
          s_t1=minibatch[sampleIdx].s_t1
          is_terminal=minibatch[sampleIdx].is_terminal
          assert((s_t[0][0][0][0]).dtype=="float64")
          assert((s_t1[0][0][0][0]).dtype=="float64")
          
          inputStates[sampleIdx] = s_t

          #Note: q_t = 1x1xNUM_ACTIONS
          q_t = self.calc_q_values(s_t)
          targets[sampleIdx][:][:] = q_t
          if(is_terminal):
            targets[sampleIdx][0][a_t] = r_t
          else:
            q_t1 = self.target_q_network.predict(s_t1)
            targets[sampleIdx][0][a_t] = self.gamma*max(q_t1[0][0]) + r_t


        #update weights
        loss = self.q_network.train_on_batch(inputStates,targets)

        #update target if ready
        return loss

    def fit(self, env, num_iterations, max_episode_length=None,num_episodes=20):
        """Fit your model to the provided environment.
        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.
        You should probably also periodically save your network
        weights and any other useful info.
        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.
        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        reward_samp = 10000

        #get initial state
        self.preprocessor.process_state_for_network(env.reset())
        s_t = self.preprocessor.frames

        #init metric vectors (to eventually plot)
        allLoss=np.zeros(num_iterations)
        rewards=np.zeros(int (num_iterations/reward_samp))        

        #iterate through environment samples
        for iteration in range(num_iterations):
            #check if target  needs to be updated
            if(iteration % self.target_update_freq == 0):
                self.target_q_network.set_weights(self.q_network.get_weights())

            cum_reward = 0
            #select action
            q_t = self.calc_q_values(s_t)
            a_t = self.policy.select_action(q_t)
            #get next state, reward, is terminal
            (s_t1, r_t, is_terminal, info)= env.step(a_t)
            self.preprocessor.process_state_for_network(s_t1)
            s_t1 = self.preprocessor.frames

            #store sample in memory
            self.memory.append(self.preprocessor.process_state_for_memory(s_t), a_t,
            r_t, self.preprocessor.process_state_for_memory(s_t1),is_terminal)

            #update policy
            if(iteration>self.num_burn_in):
              loss =self.update_policy()
              allLoss[iteration] = loss

            if (iteration % 1000 == 0):
                print (q_t)

            if (iteration % reward_samp == 0):
                print("Start Evaluation")
                cum_reward = self.evaluate(env, num_episodes, max_episode_length)
                rewards[int(iteration / reward_samp)] = cum_reward
                print ("At iteration : ", iteration, " , Reward = ", cum_reward)
            

            #update new state
            if(is_terminal):
                self.preprocessor.reset()
                self.preprocessor.process_state_for_network(env.reset())
                s_t = self.preprocessor.frames
            else:
                s_t = s_t1

        print("DONE")
        np.save("loss_linear_MR_TF", allLoss)
        np.save("reward_linear_MR_TF", rewards)

        fig = plt.figure()
        plt.plot(allLoss)
        plt.ylabel('Loss function')
        fig.savefig('Loss.png')
        plt.clf()
        plt.plot(rewards)
        plt.ylabel('Average Reward')
        fig.savefig('reward.png')


    def evaluate(self, env, num_episodes, max_episode_length=1000):
        """Test your agent with a provided environment.
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.
        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.
        You can also call the render function here if you want to
        visually inspect your policy.
        """
        cumulative_reward = 0
        actions = np.zeros(env.action_space.n)
        for episodes in range(num_episodes):
            # get initial state
            self.preprocessor.reset()
            self.preprocessor.process_state_for_network(env.reset())
            state = self.preprocessor.frames
            steps = 0
            while steps < max_episode_length:
                q_vals = self.calc_q_values(state)
                action = np.argmax(q_vals)
                actions[action] += 1
                (next_state, reward, is_terminal, info) = env.step(action)
                #reward = self.preprocessor.process_reward(reward)
                cumulative_reward = cumulative_reward + reward
                self.preprocessor.process_state_for_network(next_state)
                next_state = self.preprocessor.frames
                state = next_state
                steps = steps + 1
                if is_terminal:
                    break
        print (actions)
        avg_reward = cumulative_reward / num_episodes

        return avg_reward
