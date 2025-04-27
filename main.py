
from train.train_agent import train_agent
import pickle
from utils.solve_examples import solve_examples

if __name__ == "__main__":
    trained_agent = train_agent(num_episodes=10000)

    with open("trained_agent.pkl", "wb") as f:
        pickle.dump(trained_agent, f)

    print("\n✅ Trained agent has been saved successfully!")

    print("\n🔎 Solving example problems with the trained agent...\n")
    solve_examples(trained_agent, num_examples=5)

    print("\n✅ Example problems solved. Training and testing complete.")
