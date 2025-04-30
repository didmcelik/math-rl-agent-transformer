from train.train_agent_addition import train_agent

if __name__ == "__main__":
    # ADDITION addition_trained_agent = train_agent(episodes=5000)
    mul_trained_agent = train_agent(episodes=15000, training_type='mul')
