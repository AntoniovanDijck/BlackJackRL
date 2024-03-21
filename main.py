# Main 
# Antonio van Dyck
# 21-03-2024
# 
from helpers.test import test_agent
from helpers.train import train_agent

def main():

    rounds = 100

    train_agent(rounds)
    print(f"\nTraining the agent for {rounds} rounds:")

    print("\nTesting the trained agent with detailed metrics and plotting performance:")
    test_agent(rounds=1000, report_interval=1)

if __name__ == "__main__":
    main()
