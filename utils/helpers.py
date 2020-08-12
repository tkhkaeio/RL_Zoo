import numpy as np

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num+1)[1:-1]

# discretize the state and return the corresponding index
# (the upper and lower limits of bin are specific to the cartpole environment)
def discretize_state(observation, num_discretize):
    c_pos, c_v, p_angle, p_v = observation
    discretized = [
        np.digitize(c_pos, bins=bins(-2.4, 2.4, num_discretize)), 
        np.digitize(c_v, bins=bins(-3.0, 3.0, num_discretize)),
        np.digitize(p_angle, bins=bins(-0.5, 0.5, num_discretize)),
        np.digitize(p_v, bins=bins(-2.0, 2.0, num_discretize))
    ]
    return sum([x * (num_discretize ** i) for i, x in enumerate(discretized)])