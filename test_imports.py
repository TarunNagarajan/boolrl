#!/usr/bin/env python3
"""
Test script to verify all modules can be imported without errors
and basic functionality works as expected.
"""

def test_imports():
    print("Testing imports...")
    
    # Test main modules
    try:
        import config
        print("- Main config module imported")
    except ImportError as e:
        print(f"- Failed to import main config module: {e}")
        return False
    
    # Test environment imports
    try:
        from environments.base_env import BaseBooleanEnv
        from environments.boolean_env_mlp import BooleanSimplificationEnv
        from environments.boolean_env_gnn import BooleanSimplificationEnvGNN
        from environments.boolean_env_seq import BooleanSimplificationEnvSeq
        print("- Environment modules imported")
    except ImportError as e:
        print(f"- Failed to import environment modules: {e}")
        return False
    
    # Test agent imports
    try:
        from agents.agent_mlp import DQNAgent
        from agents.agent_gnn import DQNAgentGNN
        from agents.agent_seq import DQNAgentSeq
        print("- Agent modules imported")
    except ImportError as e:
        print(f"- Failed to import agent modules: {e}")
        return False
    
    # Test model imports
    try:
        from gnn_models import GNNQNetwork
        print("- GNN model imported")
    except ImportError as e:
        print(f"- Failed to import GNN model: {e}")
        return False
    
    # Test replay buffer
    try:
        from replay_buffer import ReplayBuffer
        print("- Replay buffer imported")
    except ImportError as e:
        print(f"- Failed to import replay buffer: {e}")
        return False
    
    print("\nAll imports successful!")
    return True


def test_basic_functionality():
    print("\nTesting basic functionality...")
    
    try:
        from environments.boolean_env_mlp import BooleanSimplificationEnv
        import config
        
        # Create a simple environment
        env = BooleanSimplificationEnv(
            max_expression_depth=3,
            max_literals=3,
            max_steps=10
        )
        
        # Test environment reset
        state = env.reset()
        print(f"- Environment reset successful, state shape: {state.shape}")
        
        # Test action size
        action_size = env.get_action_size()
        print(f"- Action space size: {action_size}")
        
        # Test state size
        state_size = env.get_state_size()
        print(f"- State space size: {state_size}")
        
        # Test a simple step
        action = 0  # Use first action
        new_state, reward, done, info = env.step(action)
        print(f"- Step executed, new state shape: {new_state.shape}, reward: {reward}, done: {done}")
        
    except Exception as e:
        print(f"- Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nBasic functionality test passed!")
    return True


if __name__ == "__main__":
    print("Running import and functionality tests for boolrl...\n")
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    if success:
        print("\n- All tests passed! The boolrl project is working correctly.")
    else:
        print("\n- Some tests failed. Please check the error messages above.")