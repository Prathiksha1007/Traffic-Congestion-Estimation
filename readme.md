
---

# 🚦 Traffic Congestion Reduction using SARSA Reinforcement Learning

<h3 align="center">
<img alt="Alt text" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
<img alt="Alt text" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>
<img alt="Alt text" src="https://img.shields.io/badge/matplotlib-%23ffffff.svg?style=for-the-badge&logo=matplotlib&logoColor=black"/>
<img alt="Alt text" src="https://img.shields.io/badge/pygame-0078D6?style=for-the-badge&logo=pygame&logoColor=white"/>
<img alt="Alt text" src="https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white"/>
</h3>

!![Demo Video](demo.mp4)

---

## 🧠 Project Overview

This project simulates a **smart traffic management system** using the **SARSA (State-Action-Reward-State-Action)** algorithm to dynamically control traffic lights and reduce congestion.
Unlike traditional systems with fixed signal timings, this model **learns from real-time conditions** to optimize signal switching, minimize waiting times, and improve overall flow — even during peak hours.

A major enhancement in this version is the **Emergency Vehicle Priority System**, which detects emergency vehicles (like ambulances or fire trucks) and instantly adjusts traffic lights to ensure their safe and rapid passage.

---

## ⚙️ How It Works

The simulation is built using **Python (Pygame)** and models a four-way intersection.
Key features include:

* **Dynamic traffic light control** based on congestion.
* **Real-time vehicle generation** with direction and color-coded intentions:

  * 🟧 **Orange:** Straight
  * 🔵 **Blue:** Left turn
  * 💗 **Pink:** Right turn
* **Adaptive SARSA learning** that improves decisions over time.
* **Emergency vehicle detection & override**, where normal SARSA control is temporarily paused to give priority to emergency vehicles.

---

## 🚨 Emergency Vehicle Handling

When an emergency vehicle is detected:

1. SARSA temporarily **pauses its learning update** to avoid corrupting the Q-table.
2. The system **forces a green signal** in the emergency vehicle’s direction.
3. Once the emergency vehicle crosses, SARSA **resumes normal learning and control**.

This ensures smooth integration of emergency response handling without disrupting ongoing reinforcement learning.

---

## 🧩 Code Structure

| File                  | Description                                                      |
| --------------------- | ---------------------------------------------------------------- |
| `main.py`             | Runs the traffic simulation and connects all modules             |
| `intersection.py`     | Defines the intersection layout and signal behavior              |
| `crossing.py`         | Handles vehicle movement and road configuration                  |
| `traffic_lights.py`   | Manages light cycles and signal state transitions                |
| `vehicle.py`          | Creates and updates vehicle behavior (including emergency logic) |
| `sarsa.py`            | Implements the SARSA learning algorithm                          |
| `model.py`            | Applies the trained SARSA model to control signals               |
| `train.py`            | Trains the model through generations and saves the Q-table       |
| `dashboard.py`        | (Optional) Displays performance metrics or visualization         |
| `simulation_logs.csv` | Stores logged data for analysis                                  |
| `requirements.txt`    | Dependencies list                                                |

---

## 🧮 SARSA Learning Details

The SARSA agent learns the optimal action (signal change) for each intersection state using:

* **Learning rate (α):** Controls how much new information overrides old knowledge
* **Discount factor (γ):** Balances immediate vs. future rewards
* **Exploration rate (ε):** Governs random exploration vs. using learned policy

The reward function uses two congestion indicators:

* **Delay Time Indicator (DTI):** Average waiting time of vehicles
* **Vehicle Count:** Number of vehicles per lane

Rewards are given for congestion reduction and penalties for increases.
SARSA updates Q-values continuously to favor actions that minimize delay and maximize flow efficiency.

---

## 📊 Training and Results

Training is done across multiple generations (e.g., 50), where each generation consists of 10,000 reward evaluations.
After each generation, the **Q-table** (`sarsa_q_table.npy`) is saved, representing learned signal policies.

Plots below show the convergence behavior across different hyperparameter settings:

|               α = 0.05, γ = 0.9              |               α = 0.05, γ = 0.95              |
| :------------------------------------------: | :-------------------------------------------: |
| <img src="./plots/alpha_0_05_gamma_0_9.png"> | <img src="./plots/alpha_0_05_gamma_0_95.png"> |

|               α = 0.05, γ = 0.99              |               α = 0.1, γ = 0.9              |
| :-------------------------------------------: | :-----------------------------------------: |
| <img src="./plots/alpha_0_05_gamma_0_99.png"> | <img src="./plots/alpha_0_1_gamma_0_9.png"> |

|               α = 0.1, γ = 0.95              |               α = 0.1, γ = 0.99              |
| :------------------------------------------: | :------------------------------------------: |
| <img src="./plots/alpha_0_1_gamma_0_95.png"> | <img src="./plots/alpha_0_1_gamma_0_99.png"> |

---

## 🧩 Installation and Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/Traffic-Congestion-Control.git
   cd Traffic-Congestion-Control
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the simulation**

   ```bash
   python main.py
   ```

4. **To train the model**

   ```bash
   python train.py
   ```

---

## 💡 Future Improvements

* Integrating **real-time traffic data** from sensors or cameras.
* Using **deep reinforcement learning (DQN)** for more complex intersections.
* Implementing **multi-agent learning** for multiple connected junctions.

---
