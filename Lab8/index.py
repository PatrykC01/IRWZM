# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10.8",
#     "numpy>=2.4.4",
#     "pandas>=3.0.2",
#     "tensorflow>=2.21.0",
# ]
# ///
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
from dataclasses import dataclass

# ==========================================
# KONFIGURACJA I DEFINICJA ŚRODOWISKA MDP
# ==========================================

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Wariant 3: Leczenie infekcji
STATES =["Healthy", "MildInfection", "SevereInfection"]
ACTIONS = ["Observe", "Antibiotic", "Hospitalize"]

S = {name: i for i, name in enumerate(STATES)}
A = {name: i for i, name in enumerate(ACTIONS)}

@dataclass
class StepResult:
    next_state: int
    reward: float
    done: bool

class InfectionEnv:
    def __init__(self, max_steps=20):
        self.max_steps = max_steps
        # Koszty: Observe 0, Antibiotic -2, Hospitalize -4
        self.costs = {
            A["Observe"]: 0.0,
            A["Antibiotic"]: 2.0,   
            A["Hospitalize"]: 4.0,  
        }
        self.reset()

    def reset(self, start_state="SevereInfection"):
        self.state = S[start_state]
        self.t = 0
        return self.state

    def step(self, action):
        self.t += 1
        s = self.state

        # Definiowanie rozkładu przejść P(s' | s, a)
        if s == S["MildInfection"]:
            if action == A["Observe"]:
                probs = [(S["MildInfection"], 0.60), (S["SevereInfection"], 0.30), (S["Healthy"], 0.10)]
            elif action == A["Antibiotic"]:
                probs = [(S["Healthy"], 0.55), (S["MildInfection"], 0.40), (S["SevereInfection"], 0.05)]
            else: # Hospitalize
                probs = [(S["Healthy"], 0.70), (S["MildInfection"], 0.25), (S["SevereInfection"], 0.05)]

        elif s == S["SevereInfection"]:
            if action == A["Observe"]:
                probs = [(S["SevereInfection"], 0.75), (S["MildInfection"], 0.20), (S["Healthy"], 0.05)]
            elif action == A["Antibiotic"]:
                probs = [(S["MildInfection"], 0.55), (S["Healthy"], 0.25), (S["SevereInfection"], 0.20)]
            else: # Hospitalize
                probs = [(S["MildInfection"], 0.55), (S["Healthy"], 0.35), (S["SevereInfection"], 0.10)]

        else: # Healthy
            if action == A["Observe"]:
                probs = [(S["Healthy"], 0.90), (S["MildInfection"], 0.10)]
            elif action == A["Antibiotic"]:
                probs = [(S["Healthy"], 0.92), (S["MildInfection"], 0.08)]
            else: # Hospitalize
                probs = [(S["Healthy"], 0.93), (S["MildInfection"], 0.07)]

        r = np.random.rand()
        cum = 0
        next_state = s
        for ns, p in probs:
            cum += p
            if r <= cum:
                next_state = ns
                break

        # Nagrody: +9 za przejście do H, -7 za przejście do S
        r_trans = 0.0
        if next_state == S["Healthy"]:
            r_trans = 9.0
        elif next_state == S["SevereInfection"]:
            r_trans = -7.0

        # Całkowita nagroda: nagroda za przejście odjąć koszt akcji
        reward = r_trans - self.costs[action]

        self.state = next_state
        done = self.t >= self.max_steps
        return StepResult(next_state, reward, done)

# ==========================================
# 1. TABELARYCZNY Q-LEARNING
# ==========================================
print("=== 1. TABELARYCZNY Q-LEARNING ===")

def epsilon_greedy(Q, s, eps):
    if np.random.rand() < eps:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[s])

def q_learning(env, episodes=2000, alpha=0.1, gamma=0.95, epsilon=0.3):
    Q = np.zeros((len(STATES), len(ACTIONS)))
    rewards =[]

    for ep in range(episodes):
        s = env.reset(start_state="SevereInfection")
        total = 0
        while True:
            a = epsilon_greedy(Q, s, epsilon)
            res = env.step(a)
            Q[s, a] += alpha * (res.reward + gamma * np.max(Q[res.next_state]) - Q[s, a])
            total += res.reward
            s = res.next_state
            if res.done:
                break
        rewards.append(total)

    return Q, rewards

env_tab = InfectionEnv()
Q_table, q_rewards = q_learning(env_tab)
policy_q = {STATES[s]: ACTIONS[np.argmax(Q_table[s])] for s in range(len(STATES))}

print("Wyuczona polityka Q-learning:", policy_q)
print("\nTabela Q:")
print(pd.DataFrame(Q_table, index=STATES, columns=ACTIONS))

plt.figure(figsize=(8,4))
plt.plot(pd.Series(q_rewards).rolling(50).mean())
plt.xlabel("Epizod")
plt.ylabel("Średnia nagroda")
plt.title("Q-learning - Krzywa uczenia (Wariant 3)")
plt.savefig("qlearning_curve.png")
plt.close()
print("\n[INFO] Zapisano wykres do pliku 'qlearning_curve.png'.\n")

# ==========================================
# 2. DEEP REINFORCEMENT LEARNING (DQN)
# ==========================================
print("=== 2. DEEP REINFORCEMENT LEARNING (DQN) ===")

def encode_state(state_idx, n_states):
    v = np.zeros(n_states, dtype=np.float32)
    v[state_idx] = 1.0
    return v

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s.astype(np.float32), a.astype(np.int32), r.astype(np.float32), s2.astype(np.float32), d.astype(np.float32)

    def __len__(self):
        return len(self.buffer)

def build_q_network(state_dim, action_dim):
    model = keras.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(action_dim)
    ])
    return model

def epsilon_greedy_action(q_model, state_vec, epsilon, action_dim):
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    q = q_model(np.expand_dims(state_vec, axis=0), training=False).numpy()[0]
    return int(np.argmax(q))

def update_target_network(q_model, target_model):
    target_model.set_weights(q_model.get_weights())

def dqn_policy_tf(q_model):
    pol = {}
    for i, s_name in enumerate(STATES):
        v = encode_state(i, len(STATES))
        q = q_model(np.expand_dims(v, axis=0), training=False).numpy()[0]
        pol[s_name] = ACTIONS[int(np.argmax(q))]
    return pol

n_states = len(STATES)
n_actions = len(ACTIONS)

env_dqn = InfectionEnv(max_steps=20)

q_model = build_q_network(state_dim=n_states, action_dim=n_actions)
target_model = build_q_network(state_dim=n_states, action_dim=n_actions)
update_target_network(q_model, target_model)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.Huber()
buffer = ReplayBuffer(capacity=10000)

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
batch_size = 64
episodes = 1200
warmup_steps = 200
target_update_freq = 50

episode_rewards =[]

@tf.function
def train_step(states_b, actions_b, rewards_b, next_states_b, dones_b):
    next_q = target_model(next_states_b, training=False)
    max_next_q = tf.reduce_max(next_q, axis=1)
    targets = rewards_b + gamma * max_next_q * (1.0 - dones_b)

    with tf.GradientTape() as tape:
        q_values = q_model(states_b, training=True)
        idx = tf.stack([tf.range(tf.shape(actions_b)[0]), actions_b], axis=1)
        q_sa = tf.gather_nd(q_values, idx)
        loss = loss_fn(targets, q_sa)

    grads = tape.gradient(loss, q_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_model.trainable_variables))
    return loss

print("Rozpoczęto trening DQN (to może chwilę potrwać)...")
step_count = 0
for ep in range(episodes):
    s = env_dqn.reset(start_state="SevereInfection")
    s_vec = encode_state(s, n_states)
    total_reward = 0.0

    while True:
        a = epsilon_greedy_action(q_model, s_vec, epsilon, n_actions)
        res = env_dqn.step(a)

        s2_vec = encode_state(res.next_state, n_states)
        buffer.push(s_vec, a, res.reward, s2_vec, float(res.done))
        step_count += 1
        s_vec = s2_vec
        total_reward += res.reward

        if step_count > warmup_steps and len(buffer) >= batch_size:
            sb, ab, rb, s2b, db = buffer.sample(batch_size)
            train_step(
                tf.convert_to_tensor(sb),
                tf.convert_to_tensor(ab),
                tf.convert_to_tensor(rb),
                tf.convert_to_tensor(s2b),
                tf.convert_to_tensor(db),
            )

        if res.done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (ep + 1) % target_update_freq == 0:
        update_target_network(q_model, target_model)

    episode_rewards.append(total_reward)

    if (ep + 1) % 200 == 0:
        print(f"Epizod {ep+1}/{episodes} | reward={total_reward:.2f} | epsilon={epsilon:.3f}")

print("\nWyuczona polityka DQN:", dqn_policy_tf(q_model))

plt.figure(figsize=(8,4))
plt.plot(pd.Series(episode_rewards).rolling(50).mean())
plt.xlabel("Epizod")
plt.ylabel("Średnia suma nagród (okno=50)")
plt.title("DQN - Krzywa uczenia (Wariant 3)")
plt.savefig("dqn_curve.png")
plt.close()
print("[INFO] Zapisano wykres do pliku 'dqn_curve.png'.\n")

# ==========================================
# 3. EXPLAINABLE RL (XRL)
# ==========================================
print("=== 3. EXPLAINABLE RL (XRL) ===")

def explain_action(q_model, state_idx):
    state_vec = encode_state(state_idx, n_states)
    q_values = q_model(np.expand_dims(state_vec, axis=0), training=False).numpy()[0]
    explanation = pd.DataFrame({
        "Akcja": ACTIONS,
        "Wartość Q(s,a)": q_values
    }).sort_values("Wartość Q(s,a)", ascending=False)
    return explanation

def counterfactual_explanation(q_model, state_idx):
    df = explain_action(q_model, state_idx)
    best = df.iloc[0]
    second = df.iloc[1]
    diff = best["Wartość Q(s,a)"] - second["Wartość Q(s,a)"]
    return {
        "Najlepsza akcja": best["Akcja"],
        "Alternatywa": second["Akcja"],
        "Różnica Q": float(diff)
    }

test_state = "SevereInfection"
test_state_idx = STATES.index(test_state)

print(f"--> Wyjaśnienie akcji dla stanu: '{test_state}':")
df_explain = explain_action(q_model, test_state_idx)
print(df_explain.to_string(index=False))

print("\n--> Analiza kontrfaktyczna ('Co by było, gdyby...?'):")
cf_explain = counterfactual_explanation(q_model, test_state_idx)
for key, val in cf_explain.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.4f}")
    else:
        print(f"  {key}: {val}")

print("\n--> Analiza wrażliwości decyzji na zmianę nagrody (Global Explainability):")
print("Sprawdzamy jak zmieni się polityka, gdy koszt akcji 'Hospitalize' wzrośnie z 4.0 do drastycznego 25.0.")

def evaluate_policy_under_cost(hospitalize_cost):
    env_tmp = InfectionEnv(max_steps=20)
    env_tmp.costs[A["Hospitalize"]] = hospitalize_cost

    q_tmp = build_q_network(n_states, n_actions)
    target_tmp = build_q_network(n_states, n_actions)
    update_target_network(q_tmp, target_tmp)

    buffer_tmp = ReplayBuffer()
    optimizer_tmp = keras.optimizers.Adam(1e-3)

    
    for _ in range(400):
        s = env_tmp.reset()
        s_vec = encode_state(s, n_states)
        while True:
            a = epsilon_greedy_action(q_tmp, s_vec, 0.2, n_actions)
            res = env_tmp.step(a)
            s2_vec = encode_state(res.next_state, n_states)
            buffer_tmp.push(s_vec, a, res.reward, s2_vec, float(res.done))

            if len(buffer_tmp) >= batch_size:
                sb, ab, rb, s2b, db = buffer_tmp.sample(batch_size)
                next_q = target_tmp(s2b, training=False)
                max_next_q = tf.reduce_max(next_q, axis=1)
                targets = rb + gamma * max_next_q * (1.0 - db)

                with tf.GradientTape() as tape:
                    q_values = q_tmp(sb, training=True)
                    idx = tf.stack([tf.range(tf.shape(ab)[0]), ab], axis=1)
                    q_sa = tf.gather_nd(q_values, idx)
                    loss = loss_fn(targets, q_sa)

                grads = tape.gradient(loss, q_tmp.trainable_variables)
                optimizer_tmp.apply_gradients(zip(grads, q_tmp.trainable_variables))

            s_vec = s2_vec
            if res.done:
                break
                
    return dqn_policy_tf(q_tmp)

new_policy = evaluate_policy_under_cost(25.0)
print("Zmieniona polityka (po radykalnym wzroście kosztu hospitalizacji):")
print(new_policy)
print("\nWniosek XRL: Agent unika najdroższej metody leczenia pomimo wysokiego prawdopodobieństwa wyleczenia, ponieważ kara (koszt) znacząco przewyższa nagrodę.")
