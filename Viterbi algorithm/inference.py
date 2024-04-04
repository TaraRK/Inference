
# Modified by: Tara Rezaei


# use this to enable/disable graphics
enable_graphics = True

import sys
import numpy as np
import robot 
if enable_graphics:
    import graphics


#-----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps

    distr = robot.Distribution()
    
    distr.update(dict([(i, 1) for i in all_possible_hidden_states]))
    distr.renormalize()

    all_observations = []
    
    for i in observations:
        if i is None:
            all_observations.append(all_possible_observed_states)
        else:
            all_observations.append([i])


    # TODO: Compute the forward messages
    forward_messages[0] = robot.Distribution()

    for i in prior_distribution:

        for j in observation_model(i):
            if j in all_observations[0]:
                forward_messages[0][(i[0], i[1], 'stay')] = observation_model(i)[j]

    forward_messages[0].renormalize()

    for i in range(num_time_steps - 1):

        forward_messages[i + 1] = robot.Distribution()

        for j in forward_messages[i]: 

            for k in robot.transition_model(j):
                
                for l in observation_model(k) :

                    if l in all_observations[i + 1]:

                        if k not in forward_messages[i + 1]:
                            forward_messages[i + 1][k] = forward_messages[i][j] * robot.transition_model(j)[k] * observation_model(k)[l]
                        else:
                            forward_messages[i + 1][k] += forward_messages[i][j] * robot.transition_model(j)[k] * observation_model(k)[l]

        forward_messages[i + 1].renormalize()
    # TODO: Compute the backward messages

    backward_messages[-1] = distr

    for i in range(num_time_steps - 1, 0, -1):
        backward_messages[i - 1] = robot.Distribution()

        for j in all_possible_hidden_states:
            for k in robot.transition_model(j):
                if k in backward_messages[i]:
                    for l in observation_model(k):
                        if l in all_observations[i]:
                            if j not in backward_messages[i - 1]:
                                backward_messages[i - 1][j] = backward_messages[i][k] * robot.transition_model(j)[k] * observation_model(k)[l]
                            else:
                                backward_messages[i - 1][j] += backward_messages[i][k] * robot.transition_model(j)[k] * observation_model(k)[l]

        backward_messages[i - 1].renormalize()

    # TODO: Compute the marginals

    for i in range(num_time_steps):
        marginals[i] = robot.Distribution()
        marginals[i].update(dict(
            [(k, forward_messages[i][j] * backward_messages[i][k]) for j in forward_messages[i] for k in
             backward_messages[i] if j == k]))
        marginals[i].renormalize()
    return marginals


def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """
    num_time_steps = len(observations)
    maxLastState = []
    maxLast = -100000000
    all_observations = []

    for i in observations:
        if i is None:
            all_observations.append(all_possible_observed_states)
        else:
            all_observations.append([i])

    w = [robot.Distribution() if i == 0 else None for i in range (num_time_steps)] 
    estimated_hidden_states = [robot.Distribution() for i in range (num_time_steps)] 

    for i in prior_distribution:
        for j in observation_model(i):
            if j in all_observations[0]:
                w[0][i]=["no previous", np.log(prior_distribution[i]) * np.log(observation_model(i)[j])]


    for i in range(num_time_steps- 1):
        w[i + 1], transition = robot.Distribution(), robot.Distribution()
        
        for j in w[i]:
            for k in transition_model(j):
                transition[k] = [j, np.log(transition_model(j)[k]) + w[i][j][1]]

        for j in transition:      
            for k in observation_model(j):
                if k in all_observations[i + 1]:
                    w[i + 1][j] = [transition[j][0], np.log(observation_model(j)[k]) + transition[j][1]]
        
    for i in w[-1]:   
        if w[-1][i][1] > maxLast:
            maxLast = w[-1][i][1]
            maxLastState = i

    estimated_hidden_states[-1] = maxLastState

    for i in range(num_time_steps - 2, -1, -1):
        estimated_hidden_states[i] = w[i + 1][estimated_hidden_states[i + 1]][0]
    
    return estimated_hidden_states

def second_best(all_possible_hidden_states,
                all_possible_observed_states,
                prior_distribution,
                transition_model,
                observation_model,
                observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: This is for you to implement

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states


#-----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(initial_distribution, transition_model, observation_model,
                  num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from a hidden Markov model given an initial
    # distribution, transition model, observation model, and number of time
    # steps, generate samples from the corresponding hidden Markov model
    hidden_states = []
    observations  = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state       = initial_distribution().sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state   = hidden_states[-1]
        new_state    = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1: # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


#-----------------------------------------------------------------------------
# Main
#

if __name__ == '__main__':
    # flags
    make_some_observations_missing = False
    use_graphics                   = enable_graphics
    need_to_generate_data          = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(robot.initial_distribution,
                          robot.transition_model,
                          robot.observation_model,
                          num_time_steps,
                          make_some_observations_missing)

    all_possible_hidden_states   = robot.get_all_hidden_states()
    all_possible_observed_states = robot.get_all_observed_states()
    prior_distribution           = robot.initial_distribution()

    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 robot.transition_model,
                                 robot.observation_model,
                                 observations)
    print('\n')

    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        top_10_states = sorted(marginals[timestep].items(), key=lambda x: x[-1], reverse=True)[:10]
        print([s for s in top_10_states if s[-1]>0])
    else:
        print('*No marginal computed*')
    print('\n')

    timestep = 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        top_10_states = sorted(marginals[timestep].items(), key=lambda x: x[-1], reverse=True)[:10]
        print([s for s in top_10_states if s[-1]>0])
    else:
        print('*No marginal computed*')
    print('\n')


    timestep = 0
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        top_10_states = sorted(marginals[timestep].items(), key=lambda x: x[-1], reverse=True)[:10]
        print([s for s in top_10_states if s[-1]>0])
    else:
        print('*No marginal computed*')
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               robot.transition_model,
                               robot.observation_model,
                               observations)
    print('\n')

    if num_time_steps > 10:
        print("Last 10 hidden states in the MAP estimate:")
        for time_step in range(num_time_steps - 10, num_time_steps):
            if estimated_states[time_step] is None:
                print('Missing')
            else:
                print(estimated_states[time_step])
        print('\n')

        print('Finding second-best MAP estimate...')
        estimated_states2 = second_best(all_possible_hidden_states,
                                        all_possible_observed_states,
                                        prior_distribution,
                                        robot.transition_model,
                                        robot.observation_model,
                                        observations)
        print('\n')

        print("Last 10 hidden states in the second-best MAP estimate:")
        for time_step in range(num_time_steps - 10 - 1, num_time_steps):
            if estimated_states2[time_step] is None:
                print('Missing')
            else:
                print(estimated_states2[time_step])
        print('\n')

        difference = 0
        for time_step in range(num_time_steps):
            if estimated_states[time_step] != hidden_states[time_step]:
                difference += 1
        print("Number of differences between MAP estimate and true hidden " + \
              "states:", difference)
        true_prob = robot.sequence_prob(hidden_states,
                                       robot.transition_model, robot.observation_model,
                                       prior_distribution, observations)
        print("True Sequence Prob:", true_prob)
        map_prob = robot.sequence_prob(estimated_states,
                                       robot.transition_model, robot.observation_model,
                                       prior_distribution, observations)
        print("MAP Estimate Prob:", map_prob)


        difference = 0
        for time_step in range(num_time_steps):
            if estimated_states2[time_step] != hidden_states[time_step]:
                difference += 1
        print("Number of differences between second-best MAP estimate and " + \
              "true hidden states:", difference)
        map_prob2 = robot.sequence_prob(estimated_states2,
                                       robot.transition_model, robot.observation_model,
                                       prior_distribution, observations)
        print("Second-best MAP Estimate Prob:", map_prob2)

        difference = 0
        for time_step in range(num_time_steps):
            if estimated_states[time_step] != estimated_states2[time_step]:
                difference += 1
        print("Number of differences between MAP and second-best MAP " + \
              "estimates:", difference)

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
