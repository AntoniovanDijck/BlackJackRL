from tqdm import tqdm
import os
from classes.game.blackjack import BlackjackGame

def train_agent(rounds=100000, save_filename='weights/dqn.weights.pth'):
    game = BlackjackGame()

        # Check if the weights file exists before attempting to load
    if os.path.exists(save_filename):
        game.player.load(save_filename)
        print(f"Loaded weights from {save_filename}")
    else:
        print(f"No pretrained weights file found at {save_filename}. Proceeding without loading.")

    # Set hyperparameters...
    for _ in tqdm(range(rounds), desc="Training Progress"):
        game.play_round(learn=True)
        
        # Check if we can replay
        if len(game.player.memory) >= game.player.batch_size:  
            game.player.replay()

        # Save the model every X rounds
        if _ % 10000 == 0:
            game.player.save(save_filename)

                    # Final save
    game.player.save(save_filename)
    print("Training completed.")

