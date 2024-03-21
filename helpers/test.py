import matplotlib.pyplot as plt
from tqdm import tqdm
from classes.game.blackjack import BlackjackGame

def test_agent(rounds=1000, load_filename='weights/dqn.weights.pth', report_interval=100):
    game = BlackjackGame()
    game.player.load(load_filename)

    # Initialize metrics
    results = {'Player wins': 0, 'Dealer wins': 0, 'Player busts': 0, 'Dealer busts': 0, 'Push': 0}
    win_rates = []
    loss_rates = []
    push_rates = []
    intervals = []

    for i in tqdm(range(1, rounds + 1)):
        result = game.play_round(learn=False)
        results[result] += 1

        # Report and collect metrics at each interval
        if i % report_interval == 0 or i == rounds:
            total_games = sum(results.values())
            win_rate = results['Player wins'] / total_games
            loss_rate = (results['Dealer wins'] + results['Player busts']) / total_games
            push_rate = results['Push'] / total_games
            win_rates.append(win_rate)
            loss_rates.append(loss_rate)
            push_rates.append(push_rate)
            intervals.append(i)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(intervals, win_rates, label='Win Rate')
    plt.plot(intervals, loss_rates, label='Loss Rate')
    plt.plot(intervals, push_rates, label='Push Rate')
    plt.xlabel('Rounds')
    plt.ylabel('Rate')
    plt.title('Agent Performance Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print final metrics
    for outcome, count in results.items():
        print(f"{outcome}: {count}")



