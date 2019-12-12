from main_TD3 import *

seeds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
seed = seed[0]
env = RandomMiniEnv
env_param = EnvParams(iteration_timeout=250,
                        goal_ang_dist=np.pi/8,
                        goal_spat_dist=1,
                        robot_name=StandardRobotExamples.INDUSTRIAL_TRICYCLE_V1)

env = EgocentricCostmap(env(params=env_param, 
                        turn_off_obstacles=False,
                        draw_new_turn_on_reset=False,
                        seed=seed))	
env = bc_gym_wrapper(env, normalize=True)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])
PATH = "./models/TD3_RandomMiniEnv_0_actor"
evaluation_model = Actor(state_dim, action_dim, max_action)
evaluation_model.load_state_dict(torch.load(PATH, map_location='cpu'))
for seed in seeds:
    env = RandomMiniEnv
    env_param = EnvParams(iteration_timeout=250,
                            goal_ang_dist=np.pi/8,
                            goal_spat_dist=1,
                            robot_name=StandardRobotExamples.INDUSTRIAL_TRICYCLE_V1)

    env = EgocentricCostmap(env(params=env_param, 
                            turn_off_obstacles=False,
                            draw_new_turn_on_reset=False,
                            seed=seed))	
    env = bc_gym_wrapper(env, normalize=True)
    prev_state = None
    state = env.reset()
    env.render()
    done = False
    while not done:
        if done: 
            state = env.reset()
            done = False 
        action = evaluation_model(torch.FloatTensor(state)).detach().numpy()  
        next_state, r, done, _ = env.step(action)
        prev_state = state
        state = next_state
        env.render()


