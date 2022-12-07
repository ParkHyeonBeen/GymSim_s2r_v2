import argparse, sys, os
from copy import deepcopy
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Algorithm.SAC_v3 import *
from Trainer import *
from Common.Utils import *
from Common.Buffer import *
from Common.logger import *

# fork, forkserver for Linux
# spawn for Windows 10
torch.multiprocessing.set_start_method('spawn', force=True) # critical for make multiprocessing work
import torch.multiprocessing as mp
from multiprocessing import Process, Manager, freeze_support
from multiprocessing.managers import BaseManager

torch.autograd.set_detect_anomaly(True)

def hyperparameters(env_name="Hopper-v4"):
    parser = argparse.ArgumentParser(description='Soft Actor Critic (SAC) v3 example')
    parser.add_argument('--base_path', default="/home/phb/ETRI/GymSim_s2r_2/", help='base path of the current project')
    parser.add_argument("--train", default="True", type=str2bool, help="If True, run_train")

    # note in txt
    parser.add_argument('--note',
                        default="change bnn std 0.1 --> 0.01",
                        type=str, help='note about what to change')

    # For test
    parser.add_argument("--test_eps", '-te', default=100, type=int, help="The number of test episode using trained policy.")
    parser.add_argument("--render", default="False", type=str2bool)
    parser.add_argument("--result_name", "-rn", default="1010-1815Hopper-v4", type=str,
                        help="Checkpoint path to a pre-trained model.")
    parser.add_argument("--model_on", default="True", type=str2bool, help="if True, activate model network")
    parser.add_argument("--result_ver", default="best", type=str, help="choose the version of the results")

    ## To make a uncertain system
    parser.add_argument('--num_case', '-nc', default=10, type=int, help='the number of cases in certain range')
    parser.add_argument('--which_kind', '-wk', default='disturb', type=str, help='disturb, freq, uncertain, noise')

    parser.add_argument('--max_disturb', '-xd', default=0.2, type=float, help='max mag of disturbance for action')
    parser.add_argument('--min_disturb', '-nd', default=0.0, type=float, help='min mag of disturbance for action')

    parser.add_argument('--max_freq', '-xf', default=10, type=int, help='max frequency of disturbance for action')
    parser.add_argument('--min_freq', '-nf', default=0, type=int, help='min frequency of disturbance for action')

    parser.add_argument('--max_uncertain', '-xu', default=0.5, type=float,
                        help='max mag of uncertainty for model param')
    parser.add_argument('--min_uncertain', '-nu', default=0.0, type=float,
                        help='min mag of uncertainty for model param')

    parser.add_argument('--max_noise', '-xn', default=0.1, type=float, help='max std of gaussian noise for state')
    parser.add_argument('--min_noise', '-nn', default=0.0, type=float, help='min std of gaussian noise for state')

    #multi processing
    parser.add_argument("--num_worker", '-nw', default=3, type=int, help="The number of agents for collect data.")
    parser.add_argument("--num_update_worker", '-nuw', default=3, type=int, help="The number of agents for update networks.")
    parser.add_argument('--eval_step', '-es', default=5000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--max_step', '-ms', default=6e6, type=int, help='Maximum training step')
    parser.add_argument('--model_train_start_step', '-mtss', default=3e6, type=int)
    parser.add_argument('--reg_weight', '-rw', default=5.0e-14, type=float, help='hopper   : 5.0e-13'
                                                                                 'walker2d : 5.0e-12'
                                                                                 'ant      : 5.0e-14'
                                                                                 'humanoid : 1.0e-13'
                                                                                 'cheetah  : 5.0e-14')

    # estimate a model dynamics
    parser.add_argument('--develop-mode', '-dm', default='imn', help="none, mn, imn")
    parser.add_argument('--use_prev_policy', '-upp', default='False', type=str2bool, help="if True, use prev best policy")
    parser.add_argument('--net-type', default="bnn", help='dnn, bnn')
    parser.add_argument('--model-hidden-dim', default=256, type=int)
    parser.add_argument('--model-lr', default=3e-4, type=float)
    parser.add_argument('--inv-model-lr-dnn', default=3e-4, type=float)
    parser.add_argument('--kl-weight', default=0.05, type=float)
    parser.add_argument("--eps-p", default=1e-1, type=float, help="Standard deviation for spatial smoothness")
    parser.add_argument("--lambda-t", default=1e-1, type=float, help="Temporal smoothness for policy loss")
    parser.add_argument("--lambda-s", default=5e-1, type=float, help="Spatial smoothness for policy loss")
    parser.add_argument("--n-history", default=3, type=int, help="state history stack")
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--eval-episode', default=10, type=int, help='Number of episodes to perform evaluation')

    #environment
    parser.add_argument('--domain-type', default='gym', type=str, help='gym')
    parser.add_argument('--env-name', '-en', default=env_name, help='Pendulum-v0, MountainCarContinuous-v0, Door')
    parser.add_argument('--robots', default='Panda', help='if domain type is suite, choose the robots')
    parser.add_argument('--discrete', default=False, type=bool, help='Always Continuous')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')

    #sac
    parser.add_argument('--batch-size', default=256, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--update_iter', default=1, type=int)
    parser.add_argument('--train-alpha', default=False, type=bool)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--actor-lr', default=0.001, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)
    parser.add_argument('--alpha-lr', default=0.0001, type=float)
    parser.add_argument('--encoder-lr', default=0.001, type=float)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--critic-update', default=1, type=int)
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')
    parser.add_argument('--log_std_min', default=-10, type=int, help='For squashed gaussian actor')
    parser.add_argument('--log_std_max', default=2, type=int, help='For squashed gaussian actor')

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')

    args = parser.parse_known_args()

    if len(args) != 1:
        args = args[0]

    return args

def main(args):

    print(torch.cuda.is_available())
    if args.cpu_only == True:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # random seed setting
    random_seed = set_seed(args.random_seed)
    print("Develop mode:", args.develop_mode)

    if args.train:
        env_name = args.env_name
    else:
        env_name = args.result_name[9:]
        print(env_name)

    #env setting
    env, test_env = gym_env(env_name, random_seed)
    max_action = env.action_space.high
    min_action = env.action_space.low
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    BaseManager.register('Buffer', Buffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.Buffer(state_dim=state_dim*args.n_history,
                                   action_dim=action_dim*args.n_history,
                                   next_state_dim=state_dim,
                                   args=args,
                                   max_size=args.buffer_size,
                                   on_policy=False,
                                   device=device)

    algorithm = SAC_v3(state_dim, action_dim, replay_buffer, args, device)

    if args.train:
        log_dir = create_log_directories("./results", args.env_name)
        update_log_file = log_dir['train'] + "/step.log"
        config_log_file = log_dir['base'] + "/config.log"
        evaluation_log_file = log_dir['train'] + "/evaluation.log"
        config_logger = create_config_logger(args, file=config_log_file)
        startTime = time.time()

        log_queue = setup_primary_logging(update_log_file, evaluation_log_file)

        # share the global parameters in multiprocessing
        algorithm.critic1.share_memory()
        algorithm.critic2.share_memory()
        algorithm.target_critic1.share_memory()
        algorithm.target_critic2.share_memory()
        algorithm.actor.share_memory()
        algorithm.log_alpha.share_memory_()  # variable
        algorithm.worker_step.share_memory_()
        algorithm.update_step.share_memory_()
        algorithm.eps.share_memory_()
        share_parameters(algorithm.actor_optimizer)
        share_parameters(algorithm.critic1_optimizer)
        share_parameters(algorithm.critic2_optimizer)
        share_parameters(algorithm.alpha_optimizer)

        algorithm.imn.share_memory()
        share_parameters(algorithm.imn_optimizer)

        rewards_queue = mp.Queue()

        processes = []
        rewards = []
        eval_step = []

        for i in range(args.num_worker):
            process = Process(target=model_trainer, args=(
                i, algorithm, rewards_queue, replay_buffer, log_dir, args, log_queue,
                startTime))  # the args contain shared and not shared
            process.daemon = True  # all processes closed when the main stops
            processes.append(process)

        for p in processes:
            p.start()
            time.sleep(5)

        while True:  # keep geting the episode reward from the queue
            r = rewards_queue.get()
            if r is not None:
                rewards.append(r[0])
                eval_step.append(r[1])
                del r
            else:
                break
            if len(rewards) > 0:
                plot_fig(rewards, log_dir['train'] + "/reward.png")

        [p.join() for p in processes]  # finished at the same time

    # Test trained policy
    else:
        np.random.seed(77)

        if env_name == "HalfCheetah-v4":
            fail_score = 4 * 1183.44
        elif env_name == "Ant-v4":
            fail_score = 2 * 647.3822
        else:
            fail_score = 0.

        eval_reward = DataManager()
        eval_success = DataManager()
        if args.model_on:
            eval_model_error = DataManager()

        log_dir = load_log_directories(args.result_name)
        load_model(algorithm.actor, log_dir["policy"], "policy_best")
        result_txt = open(log_dir["test"] + "/test_result_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind + ".txt", 'w')

        if args.model_on:
            if args.net_type == "dnn":
                load_model(algorithm.imn, log_dir[args.net_type], "better_"+args.net_type)
            else:
                load_model(algorithm.imn, log_dir[args.net_type], args.result_ver + "_" + args.net_type)
            algorithm.imn.evaluates()

        if args.which_kind == "disturb":
            min_case = args.min_disturb
            max_case = args.max_disturb
        elif args.which_kind == "freq":
            min_case = args.min_freq
            max_case = args.max_freq
        elif args.which_kind == "uncertain":
            min_case = args.min_uncertain
            max_case = args.max_uncertain
            init_geom_size = deepcopy(env.unwrapped.model.geom_size)
            init_body_mass = deepcopy(env.unwrapped.model.body_mass)
            init_body_inertia = deepcopy(env.unwrapped.model.body_inertia)
        else:
            min_case = args.min_noise
            max_case = args.max_noise

        normal_mean = 0
        normal_std = 0

        for case in np.round(np.linspace(min_case, max_case, args.num_case + 1), 3):
            if args.which_kind == "disturb":
                print("disturbance scale: ", case * 100, " percent of max thrust", file=result_txt)
                print("disturbance scale: ", case * 100, " percent of max thrust")

                eval_reward.get_xticks(np.round(case * 100, 3))
                eval_success.get_xticks(np.round(case * 100, 3))
                if args.model_on:
                    eval_model_error.get_xticks(np.round(case * 100, 3))

            elif args.which_kind == "freq":
                print("The number of cycles: ", int(case), " during a episode", file=result_txt)
                print("The number of cycles: ", int(case), " during a episode")

                eval_reward.get_xticks(int(case))
                eval_success.get_xticks(int(case))
                if args.model_on:
                    eval_model_error.get_xticks(int(case))

            elif args.which_kind == "uncertain":
                random_ratio = (case - max_case/2)*2
                env.unwrapped.model.geom_size = init_geom_size*(1. + random_ratio)
                env.unwrapped.model.body_mass = init_body_mass*(1. + random_ratio)
                env.unwrapped.model.body_inertia = init_body_inertia*(1. + random_ratio)

                print("uncertainty scale: ", random_ratio * 100, " percent of init property", file=result_txt)
                print("uncertainty scale: ", random_ratio * 100, " percent of init property")
                print("geom size : ", env.unwrapped.model.geom_size)

                eval_reward.get_xticks(np.round(random_ratio * 100, 3))
                eval_success.get_xticks(np.round(random_ratio * 100, 3))
                if args.model_on:
                    eval_model_error.get_xticks(np.round(random_ratio * 100, 3))

            else:
                print("standard deviation of state noise: ", case, file=result_txt)
                print("standard deviation of state noise: ", case)

                eval_reward.get_xticks(np.round(case, 3))
                eval_success.get_xticks(np.round(case, 3))
                if args.model_on:
                    eval_model_error.get_xticks(np.round(case, 3))

            reward_list = []
            model_error_list = []
            success_rate = 0
            suc_reward = 0

            for eps in range(args.test_eps):
                observation = env.reset()
                episode_reward = 0

                episode_model_error = []
                dist = np.zeros(action_dim)
                step = 0

                for step in range(env.spec.max_episode_steps):
                    action = algorithm.eval_action(observation)

                    if args.model_on:
                        action_dob = (action - dist) if args.which_kind != "noise" else action

                        env_action = denormalize(action_dob, max_action, min_action)
                        env_action = np.clip(env_action, min_action, max_action)

                        if args.which_kind == "disturb":
                            env_action = add_disturbance(env_action, step, env.spec.max_episode_steps,
                                                         scale=case,
                                                         frequency=2)

                        if args.which_kind == "freq":
                            env_action = add_disturbance(env_action, step, env.spec.max_episode_steps,
                                                         scale=0.1,
                                                         frequency=case)

                        next_observation, reward, done, _ = env.step(env_action)

                        next_observation_imn = deepcopy(next_observation)
                        if args.which_kind == "noise":
                            next_observation_imn = np.random.normal(next_observation_imn, case)

                        action_hat = algorithm.imn(observation,
                                                   next_observation_imn).detach().cpu().numpy()[0]

                        dist = action_hat - action
                        dist = np.clip(dist, -1.0, 1.0)
                        episode_model_error.append(np.sqrt(np.mean(dist ** 2)))
                    else:
                        env_action = denormalize(action, max_action, min_action)
                        if args.which_kind == "disturb":
                            env_action = add_disturbance(env_action, step, env.spec.max_episode_steps,
                                                         scale=case,
                                                         frequency=2)
                        if args.which_kind == "freq":
                            env_action = add_disturbance(env_action, step, env.spec.max_episode_steps,
                                                         scale=0.1,
                                                         frequency=case)

                        next_observation, reward, done, _ = env.step(env_action)

                    episode_reward += reward
                    observation = next_observation
                    if args.render:
                        env.render()

                    if done:
                        break

                if episode_reward < 0:
                    episode_reward = 0.

                if step + 1 == env.spec.max_episode_steps and episode_reward > fail_score:
                    suc_reward += episode_reward
                    success_rate += 1.

                print('Episode: ', eps, '| Episode Reward: ', episode_reward,
                      '| Mean and Std of Episode Model error: ',
                      np.mean(episode_model_error), np.std(episode_model_error))

                reward_list.append(episode_reward)
                model_error_list.append(np.mean(episode_model_error))

            if success_rate != 0:
                suc_reward /= success_rate
            else:
                suc_reward = 0.

            success_rate /= args.test_eps
            avg_reward = sum(reward_list) / args.test_eps
            eval_reward.put_data((np.mean(reward_list), np.std(reward_list)))

            if args.model_on:
                if case == 0:
                    normal_mean = np.mean(model_error_list)
                    normal_std = np.std(model_error_list)

                eval_model_error.put_data((np.mean(model_error_list) - normal_mean, np.std(model_error_list) - normal_std))

            eval_success.put_data(success_rate*100)

            print('Success rate: ', success_rate*100, '| Average Reward: ', avg_reward, '| Success Reward: ', suc_reward, file=result_txt)
            print('Success rate: ', success_rate*100, '| Average Reward: ', avg_reward, '| Success Reward: ', suc_reward)

        eval_reward.plot_variance_fig(log_dir["test"] + "/reward_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind, need_xticks=True)
        eval_success.bar_fig(log_dir["test"] + "/success_rate_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind)
        if args.model_on:
            eval_model_error.plot_variance_fig(log_dir["test"] + "/model_error_%s" % time.strftime(
                "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind, need_xticks=True)

        eval_reward.save_data(log_dir["test"], "/reward_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind, numpy=True)
        eval_reward.save_data(log_dir["test"], "/reward_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind)
        if args.model_on:
            eval_model_error.save_data(log_dir["test"], "/model_error_%s" % time.strftime(
                "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind, numpy=True)
            eval_model_error.save_data(log_dir["test"], "/model_error_%s" % time.strftime(
                "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind)
        eval_success.save_data(log_dir["test"], "/success_rate_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind, numpy=True)
        eval_success.save_data(log_dir["test"], "/success_rate_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind)

        result_txt.close()

if __name__ == '__main__':
    # freeze_support()
    #
    # # env_list = ["Walker2d-v4", "Ant-v4", "Humanoid-v4", "Hopper-v4", "InvertedDoublePendulum-v4", "HumanoidStandup-v4"]
    # env_list = ["HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "Walker2d-v4", "Ant-v4", "HumanoidStandup-v4"]
    #
    # for env in env_list:
    #     args = hyperparameters(env)
    #     main(args)
    args = hyperparameters()
    main(args)

