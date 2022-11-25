from Common.Utils import *
import torch_ard as nn_ard

def compressing(network):
    print("--------compressing---------------")

    compressed_model = {}

    network.network[0].save_net()
    network.network[2].save_net()
    network.network[4].save_net()

    compressed_model["network.0.weight"] = network.network[0].W.detach()
    compressed_model["network.2.weight"] = network.network[0].W.detach()
    compressed_model["network.4.weight"] = network.network[0].W.detach()

    compressed_model["network.0.bias"] = network.network[0].bias.detach()
    compressed_model["network.2.bias"] = network.network[0].bias.detach()
    compressed_model["network.4.bias"] = network.network[0].bias.detach()

    return compressed_model

def save_model(network, loss_best, loss_now, path, ard=False):

    print("--------save model ---------------")
    save_state = {}

    if network.net_type == "bnn":
        save_state["network"] = compressing(network)
        sparsity_ratio = round(100. * nn_ard.get_dropped_params_ratio(network), 3)
        save_state["sparsity_ratio"] = sparsity_ratio
        print('Sparsification ratio: %.3f%%' % sparsity_ratio)
    else:
        save_state["network"] = network.state_dict()

    if loss_best > loss_now:
        if ard:
            torch.save(save_state, path + "/best_" + path[-3:])
        else:
            torch.save(save_state, path + "/better_" + path[-3:])
        return loss_now
    else:
        if not ard:
            torch.save(save_state, path + "/current_" + path[-3:])

def validate_measure(error_list):
    error_max = np.max(error_list, axis=0)
    mean = np.mean(error_list, axis=0)
    std = np.std(error_list, axis=0)
    loss = np.sqrt(mean**2 + std**2)

    return [loss, mean, std, error_max]

def get_random_action_batch(observation, env_action, test_env, model_buffer, max_action, min_action):

    env_action_noise, _ = add_noise(env_action, scale=0.1)
    action_noise = normalize(env_action_noise, max_action, min_action)
    next_observation, reward, done, info = test_env.step(env_action_noise)
    model_buffer.add(observation, action_noise, reward, next_observation, float(done))

def set_sync_env(env, test_env):

    position = env.sim.data.qpos.flat.copy()
    velocity = env.sim.data.qvel.flat.copy()

    test_env.set_state(position, velocity)