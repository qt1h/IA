import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    max_q = np.max(Q[sprime])
    Q[s, a] = Q[s, a] + alpha * (r + gamma * max_q - Q[s, a])
    return Q

def epsilon_greedy(Q, s, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[s])

if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # hyperparameters
    alpha = 0.5
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    alpha_decay = 0.995

    n_epochs = 5000
    max_itr_per_epoch = 1000 #avoid infinite loop without finding a solution
    rewards = []
    window_size = 100
    recent_rewards = []
    stability_threshold = 100
    stable_count = 0
    target_stable_avg = 9
    stable_limit = 50

    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()
        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)
            Sprime, R, done, _, info = env.step(A)
            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma)
            S = Sprime
            r += R

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        alpha = max(0.01, alpha * alpha_decay)
        rewards.append(r)
        recent_rewards.append(r)

        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)

        avg_recent_reward = np.mean(recent_rewards)

        if avg_recent_reward >= target_stable_avg:
            stable_count += 1
            if stable_count >= stable_limit:
                print(f"Politique stabilisée après {e+1} épisodes. Arrêt de l'entraînement.")
                break
        else:
            stable_count = 0

    print(f"Récompense moyenne finale sur les {n_epochs} épisodes:", np.mean(rewards))

    plt.plot(rewards)
    plt.title("Récompenses par épisode")
    plt.xlabel("Épisodes")
    plt.ylabel("Récompense")
    plt.show()

    env = gym.make("Taxi-v3", render_mode="human")
    S, _ = env.reset()
    total_reward = 0
    print("Exécution d'un épisode final avec la table Q apprise...")

    for _ in range(max_itr_per_epoch):
        env.render()
        A = np.argmax(Q[S])
        S, R, done, _, info = env.step(A)
        total_reward += R

        if done:
            print(f"Épisode terminé avec une récompense totale de: {total_reward}")
            break

    env.close()
