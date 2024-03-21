# Main 
# Antonio van Dyck
# 21-03-2024
# 
from helpers.test import test_agent
from helpers.train import train_agent

def main():
 
    # FOR TRAINING UNCOMMENT THE FOLLOWING LINES:

    # print(f"\nTraining the agent for {training_rounds} rounds:")
    # training_rounds = 10000
    # train_agent(training_rounds)

    # TESTING THE LATEST WEIGHTS
    print("\nTesting the trained agent with detailed metrics and plotting performance:")
    testing_rounds = 1000
    test_agent(rounds=testing_rounds, report_interval=1)

if __name__ == "__main__":
    main()
