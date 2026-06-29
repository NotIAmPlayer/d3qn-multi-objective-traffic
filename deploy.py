import sys

# add your device's sumo/tools path
SUMO_PATH = 'D:/Eclipse/Sumo'
sys.path.append(f'{SUMO_PATH}/tools')

import torch
import torch.nn as nn
import numpy as np
import traci
import os

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = "d3qn_v2_2way_model.pth"

SUMO_BINARY = os.path.join(f"{SUMO_PATH}/bin", "sumo-gui")

SUMO_CFG = os.path.join(
    os.getcwd(),
    "2way-single-intersection",
    "2way-gen.sumocfg"
)

TL_ID = "t"

DELTA_TIME = 5
YELLOW_TIME = 3

ACTION_TO_PHASE = {
    0:0,
    1:2,
    2:4,
    3:6
}


# ==========================================================
# NETWORK
# ==========================================================

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128,action_size)
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)

        return v + (a - a.mean(dim=1,keepdim=True))


# ==========================================================
# DEPLOY AGENT
# ==========================================================

class DeployAgent:
    def __init__(self,model_path):
        ckpt=torch.load(
            model_path,
            map_location="cpu",
            weights_only=False
        )

        self.state_size=ckpt["state_size"]
        self.action_size=ckpt["action_size"]
        self.hidden=ckpt["hidden"]

        self.model=DuelingDQN(
            self.state_size,
            self.action_size,
            self.hidden
        )

        self.model.load_state_dict(
            ckpt["policy_net_state_dict"]
        )

        self.model.eval()

    def predict(self, state):
        state = state_rms.normalize(state)
        s = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            q = self.model(s)
            print("Q Values:", q.numpy())
            return q.argmax().item()


# ==========================================================
# ENVIRONMENT
# ==========================================================

# ─── Running Mean/Std untuk online state normalization ────────────────────────
class RunningMeanStd:
    """Online mean/std estimator (Welford's algorithm). Tidak butuh data awal."""
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape,  dtype=np.float64)
        self.count = 1e-4  # avoid div-by-zero

    def normalize(self, x):
        x = np.array(x, dtype=np.float64)
        std = np.sqrt(self.var / self.count) + 1e-8
        return ((x - self.mean) / std).astype(np.float32)

temp_ckpt=torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

state_rms = RunningMeanStd(shape=(8,))

state_rms.mean = temp_ckpt["state_rms_mean"]
state_rms.var = temp_ckpt["state_rms_var"]
state_rms.count = temp_ckpt["state_rms_count"]

# ─── Environment ──────────────────────────────────────────────────────────────
class DeployEnv:
    """
    Environment SUMO untuk dataset 2way-single-intersection.

    State  : antrian 8 lane masuk (normalized dengan RunningMeanStd)
    Action : 4 fasa hijau
    Reward : multi-objective
             -0.4*queue  -0.3*waiting_time  -0.2*delay  +1.2*throughput_step
    """
    FREE_FLOW_TIME = 10   # detik, estimasi travel time tanpa antrian

    def __init__(self, yellow_duration=YELLOW_TIME):
        self.yellow_duration      = yellow_duration
        self.current_phase        = 0
        self.prev_queue           = 0
        self._vehicle_enter_time  = {}

    # ── Start ──────────────────────────────────────────────────────────────────
    def start(self):
        traci.start([SUMO_BINARY, "-c", SUMO_CFG])
        print("Loaded :", traci.simulation.getLoadedNumber())
        print("Departed:", traci.simulation.getDepartedNumber())
        print("Vehicle :", traci.vehicle.getIDCount())
        print("Expected:", traci.simulation.getMinExpectedNumber())

        self.CONTROLLED_LANES = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(TL_ID)
        ))

        for _ in range(100):          # warmup
            traci.simulationStep()
        
        print("Vehicle Count:", traci.vehicle.getIDCount())

        print("Lane Vehicles:")
        for lane in self.CONTROLLED_LANES:
            print(
                lane,
                traci.lane.getLastStepVehicleNumber(lane),
                traci.lane.getLastStepHaltingNumber(lane)
            )

        raw_state = self._get_state()
        return raw_state

    # ── Step ───────────────────────────────────────────────────────────────────
    def step(self, action):
        self._apply_action(action)

        # tracking kendaraan masuk sebelum simulasi maju
        for v in traci.vehicle.getIDList():
            if v not in self._vehicle_enter_time:
                self._vehicle_enter_time[v] = traci.simulation.getTime()

        for _ in range(DELTA_TIME):
            traci.simulationStep()

        # kendaraan yang baru selesai dalam window ini
        current_time  = traci.simulation.getTime()
        step_throughput = 0
        step_delay      = 0.0
        for v in traci.simulation.getArrivedIDList():
            if v in self._vehicle_enter_time:
                tt = current_time - self._vehicle_enter_time.pop(v)
                step_throughput += 1
                step_delay      += max(tt - self.FREE_FLOW_TIME, 0.0)

        raw_state = self._get_state()
        # state = state_rms.normalize(raw_state)

        reward, reward_components = self._get_reward(step_throughput, step_delay)
        queue_len = self._get_total_queue()

        info = {
            'queue':         queue_len,
            'waiting_time':  self._get_waiting_time(),
            'throughput':    step_throughput,
            'delay':         step_delay,
            'reward':        reward_components,
            'current_phase': self.current_phase
        }
        done = traci.simulation.getMinExpectedNumber() == 0
        
        return state, reward, done, info

    # ── Internal helpers ───────────────────────────────────────────────────────
    def _get_state(self):
        return np.array([
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in self.CONTROLLED_LANES
        ], dtype=np.float32)

    def _get_total_queue(self):
        return sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in self.CONTROLLED_LANES
        )

    def _get_waiting_time(self):
        return sum(
            traci.lane.getWaitingTime(lane)
            for lane in self.CONTROLLED_LANES
        )

    def _get_reward(self, step_throughput, step_delay):
        """
        Multi-objective reward (FIXED):
          - queue_length   : mendorong antrian pendek
          - waiting_time   : mendorong kendaraan tidak lama diam
          - delay          : mendorong waktu tempuh mendekati free-flow
          - throughput     : mendorong kendaraan cepat melintas

        Bug fix:
          - prev_queue di-update di sini (sebelumnya tidak pernah di-update)
          - Pakai CONTROLLED_LANES yang sama dengan state (bukan getControlledLanes)
        """
        current_queue = self._get_total_queue()
        waiting_time  = self._get_waiting_time()

        # Normalisasi ringan agar skala reward seimbang
        norm_wait = waiting_time / (len(self.CONTROLLED_LANES) * 100.0 + 1e-8)
        norm_delay = step_delay / (DELTA_TIME + 1e-8)

        reward = (
            -0.4 * current_queue
            - 0.3 * norm_wait
            - 0.2 * norm_delay
            + 1.2 * step_throughput
        )

        # Update prev_queue setelah dipakai (bug fix!)
        self.prev_queue = current_queue

        components = {
            'queue':      -0.4 * current_queue,
            'wait':       -0.3 * norm_wait,
            'delay':      -0.2 * norm_delay,
            'throughput': +1.2 * step_throughput,
        }
        return reward, components

    def _apply_action(self, action):
        print("="*30)
        print("Action :", action)
        print("Current:", self.current_phase)
        print("Target :", ACTION_TO_PHASE[action])
        print("SUMO phase:", traci.trafficlight.getPhase(TL_ID))

        target_phase = ACTION_TO_PHASE[action]
        
        if target_phase != self.current_phase:
            yellow_phase = self.current_phase + 1
            traci.trafficlight.setPhase(TL_ID, yellow_phase)
            for _ in range(self.yellow_duration):
                traci.simulationStep()
        
        self.current_phase = target_phase
        traci.trafficlight.setPhase(TL_ID, target_phase)

        print("After apply:", traci.trafficlight.getPhase(TL_ID))

    def close(self):
        traci.close()


# ==========================================================
# MAIN
# ==========================================================

agent=DeployAgent(MODEL_PATH)

env = DeployEnv()
state = env.start()

episode_reward      = 0
total_queue         = 0
total_wait          = 0
total_throughput    = 0
decision_count      = 0

while traci.simulation.getMinExpectedNumber() > 0:
    print("Raw State:", state)
    print("Vehicle Count:", traci.vehicle.getIDCount())
    action = agent.predict(state)

    state, reward, done, info = env.step(action)

    episode_reward += reward

    total_queue += info['queue']
    total_wait += info['waiting_time']
    total_throughput += info['throughput']

    decision_count += 1

traci.close()

print("="*50)
print("DEPLOYMENT FINISHED")
print("="*50)

print(f"Average Queue       : {total_queue/decision_count:.2f}")
print(f"Average Waiting     : {total_wait/decision_count:.2f}")
print(f"Total Throughput    : {total_throughput}")
print(f"Decisions           : {decision_count}")