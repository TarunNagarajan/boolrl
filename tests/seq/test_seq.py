import torch
from boolrl.environments.boolean_env_seq import BooleanSimplificationEnvSeq
from boolrl.agents.agent_seq import DQNAgentSeq
from boolrl import config

def test_DQN_SEQ(num_test_episodes=10):
    env = BooleanSimplificationEnvSeq(max_expression_depth=config.MAX_EXPRESSION_DEPTH,
                                   max_literals=config.MAX_LITERALS,
                                   max_steps=config.MAX_STEPS_PER_EPISODE)

    agent = DQNAgentSeq(
        vocab_size=env.vocab_size,
        action_size=env.get_action_size(),
        seed=config.AGENT_SEED,
        hidden_size=config.HIDDEN_SIZE,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        tau=config.TAU,
        buffer_size=config.BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
        update_every=config.UPDATE_EVERY
    )

    try:
        agent.qnet_policy.load_state_dict(torch.load('checkpoint_seq.pth', weights_only=True))
        agent.qnet_policy.eval()
        agent.epsilon = 0.0
        print(f"{COLOR_GREEN}Successfully loaded checkpoint_seq.pth{COLOR_RESET}")
    except FileNotFoundError:
        print(f"{COLOR_RED}Error: checkpoint_seq.pth not found. Please ensure the training script has been run and the file exists.{COLOR_RESET}")
        return
    except Exception as e:
        print(f"{COLOR_RED}Error loading model: {e}{COLOR_RESET}")
        return

    print(f"\n{COLOR_LAVENDER}--- Starting SEQ Model Testing ---{COLOR_RESET}")

    optimal_count = 0
    for episode in range(1, num_test_episodes + 1):
        state = env.reset()
        initial_expr = env.current_expression
        initial_complexity = env._get_complexity(initial_expr)
        known_best_complexity = env.known_best_complexity
        
        episode_reward = 0.0
        done = False
        steps_taken = 0

        print(f"\n{COLOR_LAVENDER}--- Test Episode {episode}/{num_test_episodes} ---{COLOR_RESET}")
        print(f"Initial Expression: {COLOR_BLUE}{initial_expr}{COLOR_RESET}")
        print(f"Initial Complexity: {initial_complexity}")
        print(f"Known Best Complexity: {known_best_complexity}")

        while not done and steps_taken < config.MAX_STEPS_PER_EPISODE:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            steps_taken += 1

        final_expr = env.current_expression
        final_complexity = env._get_complexity(final_expr)

        print(f"Final Expression:   {COLOR_GREEN}{final_expr}{COLOR_RESET}")
        print(f"Final Complexity:   {final_complexity}")
        print(f"Episode Reward:     {episode_reward:.2f}")
        print(f"Steps Taken:        {steps_taken}")
        if final_complexity <= known_best_complexity:
            print(f"Result:             {COLOR_GREEN}Simplified to optimal or better!{COLOR_RESET}")
            optimal_count += 1
        else:
            print(f"Result:             {COLOR_YELLOW}Did not reach optimal complexity.{COLOR_RESET}")

    accuracy = (optimal_count / num_test_episodes) * 100 if num_test_episodes > 0 else 0
    print(f"\n{COLOR_LAVENDER}--- Test Summary ---{COLOR_RESET}")
    print(f"Total Episodes:     {num_test_episodes}")
    print(f"Successful Simplifications: {optimal_count}")
    print(f"Accuracy:           {accuracy:.2f}%{COLOR_RESET}")

COLOR_LAVENDER = "\u001b[95m"
COLOR_BLUE = "\u001b[94m"
COLOR_GREEN = "\u001b[92m"
COLOR_YELLOW = "\u001b[93m"
COLOR_RED = "\u001b[91m"
COLOR_RESET = "\u001b[0m"

if __name__ == "__main__":
    test_DQN_SEQ(num_test_episodes=100)
