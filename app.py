import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
import time
import io
import base64
from datetime import datetime, timedelta
from PIL import Image

st.set_page_config(
    page_title="AquaFlow Vision",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Global simulation constants
# ----------------------------------------------------------------------------
H, W = 240, 420
FPS = 6
PX_TO_M = 0.012
MAX_FRAMES = 320
JPEG_QUALITY = 78

SCENARIOS = [
    "Furrow + blockage & leak",
    "Furrow uniform (baseline)",
    "Sprinkler + low pressure & runoff",
    "Flooded basin + levee breach",
]


# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------
def smooth_noise(h, w, scale=32, seed=0):
    rng = np.random.default_rng(seed)
    small_h = max(2, int(h // scale))
    small_w = max(2, int(w // scale))
    small = rng.random((small_h, small_w)).astype(np.float32)
    noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if noise.max() > noise.min():
        noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


def make_soil_background(seed=7):
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0] = 116
    img[..., 1] = 86
    img[..., 2] = 52

    n1 = smooth_noise(H, W, 28, seed)
    n2 = smooth_noise(H, W, 9, seed + 1)
    texture = (n1 * 34 - 17) + (n2 * 18 - 9)

    img = img.astype(np.float32)
    img += texture[..., None] * np.array([1.0, 0.92, 0.84], dtype=np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    veg_noise = smooth_noise(H, W, 24, seed + 3)
    veg_mask = veg_noise > 0.67
    veg_mask = cv2.dilate(veg_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    if np.any(veg_mask):
        img[veg_mask] = (
            0.55 * img[veg_mask].astype(np.float32)
            + 0.45 * np.array([66, 122, 52], dtype=np.float32)
        ).astype(np.uint8)

    return img


def segment_water(frame_rgb):
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    lower_blue = np.array([88, 45, 45], dtype=np.uint8)
    upper_blue = np.array([128, 255, 240], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    return mask > 0


def compute_optical_flow(prev_gray, gray):
    if prev_gray is None:
        return np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)

    scale = 0.5
    prev_small = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    curr_small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    flow_small = cv2.calcOpticalFlowFarneback(
        prev_small,
        curr_small,
        None,
        pyr_scale=0.5,
        levels=2,
        winsize=15,
        iterations=2,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    flow = cv2.resize(flow_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    flow = flow / scale
    return flow.astype(np.float32)


def iou_score(truth_mask, pred_mask):
    truth = truth_mask.astype(bool)
    pred = pred_mask.astype(bool)

    intersection = np.logical_and(truth, pred).sum()
    union = np.logical_or(truth, pred).sum()

    if union == 0:
        return 1.0
    return float(intersection / union)


def boundary_f1_score(truth_mask, pred_mask):
    truth = truth_mask.astype(bool)
    pred = pred_mask.astype(bool)

    if not np.any(truth) and not np.any(pred):
        return 1.0
    if not np.any(truth) or not np.any(pred):
        return 0.0

    truth_u8 = truth.astype(np.uint8) * 255
    pred_u8 = pred.astype(np.uint8) * 255

    kernel3 = np.ones((3, 3), np.uint8)
    truth_boundary = cv2.subtract(truth_u8, cv2.erode(truth_u8, kernel3))
    pred_boundary = cv2.subtract(pred_u8, cv2.erode(pred_u8, kernel3))

    kernel5 = np.ones((5, 5), np.uint8)
    truth_boundary_dilated = cv2.dilate(truth_boundary, kernel5)
    pred_boundary_dilated = cv2.dilate(pred_boundary, kernel5)

    tp = np.count_nonzero(np.logical_and(truth_boundary_dilated > 0, pred_boundary > 0))
    fp = np.count_nonzero(np.logical_and(truth_boundary_dilated == 0, pred_boundary > 0))
    fn = np.count_nonzero(np.logical_and(pred_boundary_dilated == 0, truth_boundary > 0))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return float(f1)


def encode_rgb(img_rgb, quality=JPEG_QUALITY):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return buf.tobytes()


def make_gif(frame_bytes_list, max_frames=30, duration=170):
    imgs = []
    for b in frame_bytes_list[-max_frames:]:
        try:
            img = Image.open(io.BytesIO(b))
            imgs.append(img.convert("RGB"))
        except Exception:
            continue

    if not imgs:
        return None

    buf = io.BytesIO()
    imgs[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=duration,
        loop=0,
    )
    return buf.getvalue()


def display_gif(gif_bytes, width=340):
    b64 = base64.b64encode(gif_bytes).decode()
    st.markdown(
        f'<img src="data:image/gif;base64,{b64}" width="{width}" style="border-radius:8px;">',
        unsafe_allow_html=True,
    )


def metrics_df(engine):
    if not engine.metrics.get("time"):
        return pd.DataFrame()

    df = pd.DataFrame(engine.metrics)
    if len(df) > 240:
        step = max(1, len(df) // 240)
        df = df.iloc[::step]
    return df


def generate_historical_events():
    rng = np.random.default_rng(12)
    rows = []
    now = datetime.now()

    for i in range(6):
        start = now - timedelta(days=6 - i, hours=int(rng.integers(1, 8)))
        duration_s = int(rng.integers(1800, 4200))
        end = start + timedelta(seconds=duration_s)

        scenario = rng.choice(
            [
                "Furrow + blockage & leak",
                "Sprinkler + low pressure & runoff",
                "Flooded basin + levee breach",
            ]
        )

        rows.append(
            {
                "Start": start.strftime("%Y-%m-%d %H:%M"),
                "End": end.strftime("%Y-%m-%d %H:%M"),
                "Scenario": scenario,
                "Duration s": duration_s,
                "Frames": int(duration_s * FPS / 8),
                "Coverage %": round(float(rng.uniform(68, 97)), 1),
                "Avg speed m/s": round(float(rng.uniform(0.05, 0.16)), 3),
                "Alerts": int(rng.integers(0, 4)),
                "Weed mow events": int(rng.integers(0, 2)),
                "Status": "Completed",
            }
        )

    return pd.DataFrame(rows)


def store_current_event(engine, manual=False):
    if engine.frame_idx < 5:
        return

    status = "Saved manually" if manual else ("Completed" if engine.completed else "In progress")
    row = engine.current_summary(status=status)
    df = pd.DataFrame([row])

    if "history_events" not in st.session_state:
        st.session_state.history_events = generate_historical_events()

    st.session_state.history_events = pd.concat(
        [st.session_state.history_events, df],
        ignore_index=True,
    )
    engine.event_stored = True


# ----------------------------------------------------------------------------
# Weed lifecycle and regrowth model
# ----------------------------------------------------------------------------
class WeedModel:
    def __init__(self, seed=17):
        self.cell = 12
        self.gh = H // self.cell
        self.gw = W // self.cell
        self.rng = np.random.default_rng(seed)

        self.stage = np.zeros((self.gh, self.gw), dtype=np.int8)
        self.age = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.biomass = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.root_reserve = self.rng.uniform(0.2, 0.8, size=(self.gh, self.gw)).astype(np.float32)
        self.moisture = np.full((self.gh, self.gw), 0.12, dtype=np.float32)

        seedbank_noise = smooth_noise(self.gh, self.gw, 3, seed + 5)
        self.seedbank = np.clip(seedbank_noise, 0, 1).astype(np.float32)

        initial_weeds = self.seedbank > 0.68
        self.stage[initial_weeds] = 2
        n_initial = int(np.count_nonzero(initial_weeds))
        if n_initial > 0:
            self.biomass[initial_weeds] = self.rng.uniform(0.35, 0.75, size=n_initial)
            self.age[initial_weeds] = self.rng.uniform(15, 40, size=n_initial)

        self.hours = 0.0
        self.mow_count = 0
        self.last_mow_hours = None
        self.step_count = 0

        self.history = {
            "time": [],
            "biomass": [],
            "regrowth": [],
            "moisture": [],
        }
        for i in range(6):
            self.history[f"stage{i}"] = []

    def render(self):
        color_map = np.array(
            [
                [0, 0, 0],          # none
                [130, 190, 95],     # germinating
                [75, 160, 75],      # juvenile
                [38, 115, 48],      # mature
                [158, 138, 62],     # senescent
                [95, 215, 125],     # regrowth
            ],
            dtype=np.uint8,
        )

        low_color = color_map[self.stage.astype(int)]
        low_alpha = ((self.biomass > 0.04) & (self.stage > 0)).astype(np.uint8)

        img = cv2.resize(low_color, (W, H), interpolation=cv2.INTER_NEAREST)
        alpha = cv2.resize(low_alpha, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        return img, alpha

    def mow(self):
        cut = (self.stage >= 2) & (self.stage <= 4)
        if not np.any(cut):
            return

        self.root_reserve[cut] = np.clip(0.30 + self.biomass[cut] * 0.55, 0, 1)

        regrow = cut & (self.root_reserve > 0.46)
        non_regrow = cut & (~regrow)

        self.stage[regrow] = 5
        self.biomass[regrow] *= 0.18
        self.age[regrow] = 0

        self.stage[non_regrow] = 0
        self.biomass[non_regrow] = 0.02
        self.age[non_regrow] = 0

        self.mow_count += 1
        self.last_mow_hours = self.hours

    def step(self, dt, water_mask):
        self.step_count += 1
        self.hours += 1.4

        water_u8 = water_mask.astype(np.uint8) * 255
        coarse_water = cv2.resize(water_u8, (self.gw, self.gh), interpolation=cv2.INTER_AREA) > 80

        self.moisture = np.clip(
            0.90 * self.moisture + 0.22 * coarse_water.astype(np.float32),
            0,
            1,
        )

        rand = self.rng.random(self.stage.shape)
        germinate = (
            (self.stage == 0)
            & (self.seedbank > 0.38)
            & (self.moisture > 0.34)
            & (rand < 0.12)
        )

        self.stage[germinate] = 1
        self.age[germinate] = 0
        self.biomass[germinate] = 0.05

        growth_factor = np.ones_like(self.biomass, dtype=np.float32)
        growth_factor[self.stage == 1] = 0.55
        growth_factor[self.stage == 2] = 1.00
        growth_factor[self.stage == 3] = 0.65
        growth_factor[self.stage == 4] = 0.12
        growth_factor[self.stage == 5] = 1.50

        growth = 0.0065 * (0.25 + self.moisture) * growth_factor
        self.biomass = np.clip(self.biomass + growth, 0, 1.6)
        self.age += 1.4

        to_juvenile = (self.stage == 1) & ((self.age > 7) | (self.biomass > 0.16))
        self.stage[to_juvenile] = 2

        to_mature = (self.stage == 2) & (self.age > 36) & (self.biomass > 0.62)
        self.stage[to_mature] = 3

        to_senescent = (self.stage == 3) & ((self.age > 150) | (self.moisture < 0.13))
        self.stage[to_senescent] = 4

        senescent = self.stage == 4
        self.biomass[senescent] *= 0.985
        self.seedbank[senescent] = np.clip(self.seedbank[senescent] + 0.002, 0, 1)

        dead = senescent & (self.biomass < 0.06)
        self.stage[dead] = 0
        self.biomass[dead] = 0
        self.age[dead] = 0

        regrowth_established = (self.stage == 5) & (self.biomass > 0.34)
        self.stage[regrowth_established] = 2

        self.root_reserve = np.clip(
            self.root_reserve + 0.0015 * (self.moisture > 0.3),
            0,
            1,
        )

        if self.step_count % 3 == 0:
            counts = np.bincount(self.stage.ravel().astype(int), minlength=6).astype(float)
            pct = 100 * counts / max(1.0, counts.sum())

            active = self.stage > 0
            regrowth = self.stage == 5

            self.history["time"].append(round(self.hours, 1))
            for i in range(6):
                self.history[f"stage{i}"].append(round(float(pct[i]), 2))

            self.history["biomass"].append(
                round(float(self.biomass[active].mean()) if np.any(active) else 0.0, 3)
            )
            self.history["regrowth"].append(
                round(float(self.biomass[regrowth].mean()) if np.any(regrowth) else 0.0, 3)
            )
            self.history["moisture"].append(round(float(self.moisture.mean()), 3))


# ----------------------------------------------------------------------------
# Irrigation simulation + computer vision engine
# ----------------------------------------------------------------------------
class IrrigationSim:
    def __init__(self, scenario, seed=11):
        self.scenario = scenario
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.dt = 1.0 / FPS
        self.t = 0.0
        self.water_time = 0.0
        self.frame_idx = 0
        self.completed = False
        self.event_stored = False
        self.irrigation_on = True

        self.X, self.Y = np.meshgrid(np.arange(W), np.arange(H))

        self.weeds = WeedModel(seed + 4)
        self.alerts = []
        self.history_frames = []

        self.metrics = {
            "time": [],
            "coverage": [],
            "obs_front": [],
            "exp_front": [],
            "speed_m_s": [],
            "iou": [],
            "boundary_f1": [],
            "front_mae": [],
            "leak_area": [],
        }

        self.anom = {
            "TP_blocked": 0,
            "FP_blocked": 0,
            "FN_blocked": 0,
            "TP_leak": 0,
            "FP_leak": 0,
            "FN_leak": 0,
        }

        self.truth_blocked_frames = 0
        self.truth_leak_frames = 0

        self.speed_ema = 0.0
        self.current_coverage = 0.0
        self.current_speed_m_s = 0.0
        self.current_iou = 1.0
        self.current_boundary_f1 = 1.0
        self.current_front_mae = 0.0
        self.current_leak_area = 0

        self.start_time = datetime.now()

        self._build_scenario()
        self._initialize_first_frame()

    def _build_scenario(self):
        self.background = make_soil_background(self.seed)
        self.expected_mask = np.zeros((H, W), dtype=np.uint8)
        self.coverage_mask = np.zeros((H, W), dtype=np.uint8)

        self.water_mask = np.zeros((H, W), dtype=bool)
        self.prev_water_mask = None
        self.cumulative_obs = np.zeros((H, W), dtype=np.float32)

        self.mode = "furrow"
        self.radial_type = ""
        self.furrows = []
        self.sources = []

        self.basin_mask = None
        self.obstacle_mask = None
        self.breach_point = None
        self.breach_distance = 0.0
        self.obstruction_alerted = False
        self.leak_alerted = False

        if "Furrow" in self.scenario:
            self._build_furrow()
        elif "Sprinkler" in self.scenario:
            self._build_sprinkler()
        else:
            self._build_flood()

        safe = cv2.dilate(self.expected_mask, np.ones((7, 7), np.uint8), iterations=1)
        self.expected_safe = safe > 0

        if np.count_nonzero(self.coverage_mask) == 0:
            self.coverage_mask = self.expected_mask.copy()

        self.coverage_mask = cv2.dilate(
            self.coverage_mask,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )

    def _build_furrow(self):
        self.mode = "furrow"
        self.main_x = 46

        rows = list(range(48, H - 24, 32))
        scenario_lower = self.scenario.lower()
        blocked_idx = 2 if "blockage" in scenario_lower else None
        leak_idx = 4 if "leak" in scenario_lower else None

        cv2.line(self.background, (self.main_x, 18), (self.main_x, H - 18), (78, 60, 40), 12)
        cv2.line(self.expected_mask, (self.main_x, 18), (self.main_x, H - 18), 255, 14)

        for i, y in enumerate(rows):
            speed = float(self.rng.uniform(8.5, 12.0))
            x0 = self.main_x + 7
            x1 = W - 26

            blocked_x = int(W * 0.52) if i == blocked_idx else None
            leak_x = int(W * 0.36) if i == leak_idx else None

            cv2.line(self.background, (self.main_x + 5, y), (W - 24, y), (88, 66, 42), 7)
            cv2.line(self.expected_mask, (self.main_x + 5, y), (W - 24, y), 255, 9)

            self.furrows.append(
                {
                    "y": int(y),
                    "x0": float(x0),
                    "x1": float(x1),
                    "speed": speed,
                    "front": float(x0),
                    "expected_front": float(x0),
                    "front_obs": float(x0),
                    "blocked_x": blocked_x,
                    "leak_x": leak_x,
                    "blocked_active": False,
                    "leak_active": False,
                    "leak_radius": 0.0,
                    "stopped_frames": 0,
                    "alerted_blocked": False,
                }
            )

        self.coverage_mask = self.expected_mask.copy()

    def _build_sprinkler(self):
        self.mode = "radial"
        self.radial_type = "sprinkler"

        centers = [(115, 82), (305, 86), (210, 170)]
        planned_speeds = [7.2, 7.5, 7.2]
        low_idx = 1
        leak_idx = 2
        max_radius = 74

        for i, (x, y) in enumerate(centers):
            zone = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(zone, (x, y), max_radius + 16, 255, -1)

            cv2.circle(self.expected_mask, (x, y), max_radius, 255, -1)
            cv2.circle(self.background, (x, y), max_radius, (92, 72, 48), 2)

            actual_speed = planned_speeds[i] * (0.42 if i == low_idx else 1.0)

            self.sources.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "planned_speed": float(planned_speeds[i]),
                    "actual_speed": float(actual_speed),
                    "max_radius": max_radius,
                    "radius": 0.0,
                    "expected_radius": 0.0,
                    "radius_obs": 0.0,
                    "low_pressure": i == low_idx,
                    "leak": i == leak_idx,
                    "leak_active": False,
                    "leak_radius": 0.0,
                    "leak_point": (int(x + max_radius + 10), int(y + 14)),
                    "alerted_low": False,
                    "zone_mask": zone > 0,
                }
            )

        self.coverage_mask = self.expected_mask.copy()

    def _build_flood(self):
        self.mode = "radial"
        self.radial_type = "flood"

        x1, y1, x2, y2 = 64, 44, W - 46, H - 42

        self.basin_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(self.basin_mask, (x1, y1), (x2, y2), 255, -1)

        self.obstacle_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(self.obstacle_mask, (235, 135), (70, 34), 0, 0, 360, 255, -1)

        cv2.rectangle(self.background, (x1, y1), (x2, y2), (74, 58, 40), -1)
        cv2.ellipse(self.background, (235, 135), (70, 34), 0, 0, 360, (126, 96, 58), -1)
        cv2.rectangle(self.background, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), (132, 102, 64), 8)

        self.expected_mask = self.basin_mask.copy()
        self.coverage_mask = self.basin_mask.copy()

        inlet = (x1 + 18, y1 + 18)
        max_radius = int(np.hypot(x2 - inlet[0], y2 - inlet[1])) + 25
        breach_point = (x2, 140)
        breach_distance = float(np.hypot(breach_point[0] - inlet[0], breach_point[1] - inlet[1]))

        self.breach_point = breach_point
        self.breach_distance = breach_distance

        self.sources.append(
            {
                "x": int(inlet[0]),
                "y": int(inlet[1]),
                "planned_speed": 22.0,
                "actual_speed": 22.0,
                "max_radius": max_radius,
                "radius": 0.0,
                "expected_radius": 0.0,
                "radius_obs": 0.0,
                "low_pressure": False,
                "leak": False,
                "leak_active": False,
                "leak_radius": 0.0,
                "leak_point": (int(breach_point[0] + 10), int(breach_point[1])),
                "alerted_low": False,
                "zone_mask": self.basin_mask > 0,
            }
        )

    def _initialize_first_frame(self):
        self.water_mask = self.update_water_current()
        self.prev_water_mask = self.water_mask

        raw = self.render_raw(self.water_mask)
        gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        segmented = segment_water(raw)
        flow = np.zeros((H, W, 2), dtype=np.float32)

        self.process_measurements(segmented, flow)

        annotated = self.annotate(raw, segmented, flow)
        jpg = encode_rgb(annotated)
        if jpg is not None:
            self.history_frames.append(jpg)

        self.prev_gray = gray

    def add_alert(self, alert_type, severity, location, description):
        if location is None:
            location = (0, 0)

        self.alerts.append(
            {
                "Frame": self.frame_idx,
                "Time s": round(self.t, 1),
                "Type": alert_type,
                "Severity": severity,
                "Location": f"x={int(location[0])}, y={int(location[1])}",
                "Description": description,
            }
        )

        if len(self.alerts) > 250:
            self.alerts.pop(0)

    def mow_event(self, reason="Manual mowing"):
        self.weeds.mow()
        self.add_alert(
            "Weed disturbance / mowing",
            "Info",
            (W // 2, H // 2),
            f"{reason}. Watch lifecycle regrowth in the Weed tab.",
        )

    def update_water_current(self):
        if not self.irrigation_on and self.prev_water_mask is not None:
            u8 = self.prev_water_mask.astype(np.uint8) * 255
            u8 = cv2.erode(u8, np.ones((3, 3), np.uint8), iterations=1)
            return u8 > 0

        return self.update_water_active()

    def update_water_active(self):
        if self.mode == "furrow":
            return self._update_furrow()
        if self.radial_type == "sprinkler":
            return self._update_sprinkler()
        return self._update_flood()

    def _update_furrow(self):
        mask = np.zeros((H, W), dtype=np.uint8)

        cv2.line(mask, (self.main_x, 18), (self.main_x, H - 18), 255, 10)

        for f in self.furrows:
            elapsed = self.water_time
            base_target = min(f["x1"], f["x0"] + f["speed"] * elapsed)
            f["expected_front"] = float(base_target)

            target = base_target
            if f["blocked_x"] is not None:
                target = min(target, f["blocked_x"] - 4)

            old = f["front"]
            if target > old:
                max_step = f["speed"] * self.dt * 1.2 + 0.35
                new_front = old + min(target - old, max_step) + float(self.rng.normal(0, 0.12))
                f["front"] = float(np.clip(new_front, f["x0"], target))

            if f["blocked_x"] is not None and base_target > f["blocked_x"] + 8:
                f["blocked_active"] = True

            if f["leak_x"] is not None and f["front"] >= f["leak_x"]:
                f["leak_active"] = True

            if f["front"] > f["x0"] + 1:
                cv2.line(
                    mask,
                    (int(f["x0"]), f["y"]),
                    (int(f["front"]), f["y"]),
                    255,
                    5,
                )

            if f["leak_active"]:
                f["leak_radius"] = min(24.0, f["leak_radius"] + 0.16)
                cy = f["y"] + 22
                cv2.line(mask, (int(f["leak_x"]), f["y"]), (int(f["leak_x"]), int(cy)), 255, 4)
                cv2.circle(mask, (int(f["leak_x"]), int(cy)), int(f["leak_radius"]), 255, -1)

        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        return mask > 0

    def _update_sprinkler(self):
        mask = np.zeros((H, W), dtype=np.uint8)

        for s in self.sources:
            planned = min(s["max_radius"], s["planned_speed"] * self.water_time)
            actual_target = min(s["max_radius"], s["actual_speed"] * self.water_time)

            s["expected_radius"] = float(planned)

            old = s["radius"]
            if actual_target > old:
                step = s["actual_speed"] * self.dt * 1.25 + 0.25
                s["radius"] = old + min(actual_target - old, step)

            cv2.circle(mask, (s["x"], s["y"]), int(s["radius"]), 255, -1)

            if s["leak"] and s["radius"] > 45:
                s["leak_active"] = True

            if s["leak_active"]:
                s["leak_radius"] = min(26.0, s["leak_radius"] + 0.18)
                lx, ly = s["leak_point"]
                cv2.ellipse(
                    mask,
                    (lx, ly),
                    (int(s["leak_radius"] * 1.5), int(s["leak_radius"] * 0.9)),
                    0,
                    0,
                    360,
                    255,
                    -1,
                )

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask > 0

    def _update_flood(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        s = self.sources[0]

        planned = min(s["max_radius"], s["planned_speed"] * self.water_time)
        actual_target = min(s["max_radius"], s["actual_speed"] * self.water_time)

        s["expected_radius"] = float(planned)

        old = s["radius"]
        if actual_target > old:
            step = s["actual_speed"] * self.dt * 1.2 + 0.4
            s["radius"] = old + min(actual_target - old, step)

        cv2.circle(mask, (s["x"], s["y"]), int(s["radius"]), 255, -1)

        mask = cv2.bitwise_and(mask, self.basin_mask)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(self.obstacle_mask))

        if s["radius"] > self.breach_distance - 18:
            s["leak_active"] = True

        if s["leak_active"]:
            s["leak_radius"] = min(28.0, s["leak_radius"] + 0.22)
            lx, ly = self.breach_point
            cv2.circle(mask, (lx + 10, ly), int(s["leak_radius"]), 255, -1)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask > 0

    def render_raw(self, water_mask):
        frame = self.background.copy()

        weed_img, weed_alpha = self.weeds.render()
        if np.any(weed_alpha):
            frame[weed_alpha] = (
                0.62 * frame[weed_alpha].astype(np.float32)
                + 0.38 * weed_img[weed_alpha].astype(np.float32)
            ).astype(np.uint8)

        if np.any(water_mask):
            ripple = (
                np.sin(self.X / 9.0 - self.t * 2.6)
                + np.cos(self.Y / 7.0 - self.t * 1.9)
            ) * 0.25

            alpha = 0.60 + 0.16 * np.clip(ripple[water_mask], -0.5, 0.5)
            base = frame[water_mask].astype(np.float32)
            water_color = np.array([70, 130, 205], dtype=np.float32)

            frame[water_mask] = np.clip(
                base * (1.0 - alpha)[:, None] + water_color[None, :] * alpha[:, None],
                0,
                255,
            ).astype(np.uint8)

            specular = water_mask & (ripple > 0.38)
            if np.any(specular):
                frame[specular] = np.clip(
                    frame[specular].astype(np.float32) + 30,
                    0,
                    255,
                ).astype(np.uint8)

        return frame

    def render_weed_map(self):
        frame = self.background.copy()
        weed_img, weed_alpha = self.weeds.render()

        if np.any(weed_alpha):
            frame[weed_alpha] = (
                0.55 * frame[weed_alpha].astype(np.float32)
                + 0.45 * weed_img[weed_alpha].astype(np.float32)
            ).astype(np.uint8)

        return frame

    def process_measurements(self, segmented, flow):
        coverage_mask = self.coverage_mask > 0
        denom = np.count_nonzero(coverage_mask)

        if denom > 0:
            wet_in_coverage = np.logical_and(segmented, coverage_mask)
            self.current_coverage = float(np.count_nonzero(wet_in_coverage) / denom)
        else:
            self.current_coverage = 0.0

        if self.mode == "furrow":
            obs_front, exp_front, blocked_det = self.detect_furrow_fronts(segmented)
        else:
            obs_front, exp_front, blocked_det = self.detect_radial_fronts(segmented)

        if len(self.metrics["obs_front"]) > 0:
            prev_front = self.metrics["obs_front"][-1]
            raw_speed = (obs_front - prev_front) / self.dt
            raw_speed = max(0.0, raw_speed)
            self.speed_ema = 0.75 * self.speed_ema + 0.25 * raw_speed
        else:
            self.speed_ema = 0.0

        self.current_speed_m_s = self.speed_ema * PX_TO_M

        leak_det, leak_centroid, leak_area, truth_leak = self.detect_leak(segmented)
        self.current_leak_area = leak_area

        obstruction_det = False
        if self.mode == "radial" and self.radial_type == "flood":
            obstruction_det = self.detect_obstruction(segmented)

        blocked_det = blocked_det or obstruction_det
        truth_blocked = self.truth_blocked()

        if leak_det and not self.leak_alerted:
            self.add_alert(
                "Leakage zone / off-target water",
                "High",
                leak_centroid,
                f"Water outside planned area detected ({leak_area}px). Possible leakage or runoff.",
            )
            self.leak_alerted = True

        self.update_anomaly_metrics(truth_blocked, blocked_det, truth_leak, leak_det)

        iou = iou_score(self.water_mask, segmented)
        bf1 = boundary_f1_score(self.water_mask, segmented)
        front_mae = self.front_mae()

        self.current_iou = iou
        self.current_boundary_f1 = bf1
        self.current_front_mae = front_mae

        self.metrics["time"].append(round(self.t, 2))
        self.metrics["coverage"].append(round(self.current_coverage * 100, 2))
        self.metrics["obs_front"].append(round(obs_front, 1))
        self.metrics["exp_front"].append(round(exp_front, 1))
        self.metrics["speed_m_s"].append(round(self.current_speed_m_s, 4))
        self.metrics["iou"].append(round(iou, 3))
        self.metrics["boundary_f1"].append(round(bf1, 3))
        self.metrics["front_mae"].append(round(front_mae, 2))
        self.metrics["leak_area"].append(int(leak_area))

        self.cumulative_obs += segmented.astype(np.float32)
        self.segmented_current = segmented
        self.flow_current = flow

    def detect_furrow_fronts(self, segmented):
        obs_fronts = []
        exp_fronts = []
        blocked_det = False

        for f in self.furrows:
            y = f["y"]
            y1 = max(0, y - 3)
            y2 = min(H, y + 4)

            x0 = int(f["x0"])
            x1 = int(f["x1"]) + 1

            band = segmented[y1:y2, x0:x1]

            if band.size == 0:
                front_obs = f["x0"]
            else:
                cols = np.any(band, axis=0)
                if np.any(cols):
                    front_obs = f["x0"] + float(np.max(np.where(cols)[0]))
                else:
                    front_obs = f["x0"]

            f["front_obs"] = 0.65 * float(f.get("front_obs", f["x0"])) + 0.35 * float(front_obs)

            obs_fronts.append(f["front_obs"])
            exp_fronts.append(f["expected_front"])

            gap = f["expected_front"] - f["front_obs"]

            if self.irrigation_on and gap > 28 and f["expected_front"] > f["x0"] + 45:
                f["stopped_frames"] += 1
            else:
                f["stopped_frames"] = max(0, f["stopped_frames"] - 1)

            if f["stopped_frames"] > 8:
                blocked_det = True
                if not f["alerted_blocked"]:
                    self.add_alert(
                        "Blocked channel / unexpected stop",
                        "High",
                        (int(f["front_obs"]), y),
                        f"Furrow near y={y} stopped around x={int(f['front_obs'])} px while expected to advance.",
                    )
                    f["alerted_blocked"] = True

        obs_mean = float(np.mean(obs_fronts)) if obs_fronts else 0.0
        exp_mean = float(np.mean(exp_fronts)) if exp_fronts else 0.0

        return obs_mean, exp_mean, blocked_det

    def detect_radial_fronts(self, segmented):
        obs_radii = []
        exp_radii = []
        blocked_det = False

        for s in self.sources:
            local = np.logical_and(segmented, s["zone_mask"])
            points = np.argwhere(local)

            if len(points) > 40:
                dists = np.sqrt(
                    (points[:, 1] - s["x"]) ** 2 + (points[:, 0] - s["y"]) ** 2
                )
                radius_obs = float(np.percentile(dists, 88))
            else:
                radius_obs = 0.0

            s["radius_obs"] = 0.70 * float(s.get("radius_obs", 0.0)) + 0.30 * radius_obs

            obs_radii.append(s["radius_obs"])
            exp_radii.append(s["expected_radius"])

            if (
                self.radial_type == "sprinkler"
                and s["low_pressure"]
                and self.irrigation_on
                and self.water_time > 6
                and s["expected_radius"] - s["radius_obs"] > 16
            ):
                blocked_det = True
                if not s["alerted_low"]:
                    self.add_alert(
                        "Low pressure / stagnation",
                        "Medium",
                        (s["x"], s["y"]),
                        f"Sprinkler at ({s['x']}, {s['y']}) wetting radius is lagging expected hydraulic movement.",
                    )
                    s["alerted_low"] = True

        obs_mean = float(np.mean(obs_radii)) if obs_radii else 0.0
        exp_mean = float(np.mean(exp_radii)) if exp_radii else 0.0

        return obs_mean, exp_mean, blocked_det

    def detect_leak(self, segmented):
        outside = np.logical_and(segmented, ~self.expected_safe)
        outside_u8 = outside.astype(np.uint8) * 255
        outside_u8 = cv2.morphologyEx(outside_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        outside = outside_u8 > 0

        area = int(np.count_nonzero(outside))
        truth_leak = self.truth_leak()

        if area > 140:
            coords = np.argwhere(outside)
            centroid = (int(coords[:, 1].mean()), int(coords[:, 0].mean()))
            return True, centroid, area, truth_leak

        return False, None, area, truth_leak

    def detect_obstruction(self, segmented):
        if self.radial_type != "flood":
            return False

        expected_area = np.count_nonzero(self.expected_mask > 0)
        if expected_area == 0:
            return False

        obs_area = np.count_nonzero(np.logical_and(segmented, self.expected_mask > 0))
        obs_fraction = obs_area / expected_area

        s = self.sources[0]
        expected_fraction = min(
            1.0,
            (s["expected_radius"] / max(1.0, float(s["max_radius"]))) ** 1.5,
        )

        if expected_fraction > 0.32 and obs_fraction < expected_fraction * 0.80:
            if not self.obstruction_alerted:
                self.add_alert(
                    "Obstruction / dry pocket",
                    "Medium",
                    (235, 135),
                    "Raised obstruction is preventing water from entering expected basin area.",
                )
                self.obstruction_alerted = True
            return True

        return False

    def truth_blocked(self):
        if self.mode == "furrow":
            return any(f["blocked_active"] for f in self.furrows)

        if self.radial_type == "sprinkler":
            return any(
                s["low_pressure"] and s["expected_radius"] > 22 for s in self.sources
            )

        if self.radial_type == "flood":
            s = self.sources[0]
            return s["expected_radius"] > 0.32 * s["max_radius"]

        return False

    def truth_leak(self):
        if self.mode == "furrow":
            return any(f["leak_active"] for f in self.furrows)

        return any(s.get("leak_active", False) for s in self.sources)

    def update_anomaly_metrics(self, truth_blocked, blocked_det, truth_leak, leak_det):
        if truth_blocked:
            self.truth_blocked_frames += 1
        else:
            self.truth_blocked_frames = 0

        if truth_leak:
            self.truth_leak_frames += 1
        else:
            self.truth_leak_frames = 0

        if self.frame_idx < 8:
            return

        if truth_blocked:
            if blocked_det:
                self.anom["TP_blocked"] += 1
            elif self.truth_blocked_frames > 10:
                self.anom["FN_blocked"] += 1
        else:
            if blocked_det:
                self.anom["FP_blocked"] += 1

        if truth_leak:
            if leak_det:
                self.anom["TP_leak"] += 1
            elif self.truth_leak_frames > 10:
                self.anom["FN_leak"] += 1
        else:
            if leak_det:
                self.anom["FP_leak"] += 1

    def front_mae(self):
        errors = []

        if self.mode == "furrow":
            for f in self.furrows:
                errors.append(abs(float(f.get("front_obs", f["x0"])) - float(f["front"])))
        else:
            for s in self.sources:
                errors.append(abs(float(s.get("radius_obs", 0.0)) - float(s["radius"])))

        if not errors:
            return 0.0

        return float(np.mean(errors))

    def annotate(self, raw_frame, segmented, flow):
        ann = raw_frame.copy()

        contours, _ = cv2.findContours(
            segmented.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(ann, contours, -1, (0, 255, 255), 1)

        if flow is not None:
            step = 28
            for y in range(step // 2, H, step):
                for x in range(step // 2, W, step):
                    if segmented[y, x]:
                        dx = float(flow[y, x, 0])
                        dy = float(flow[y, x, 1])
                        mag = float(np.hypot(dx, dy))

                        if 0.35 < mag < 30:
                            end_x = int(x + np.clip(dx * 3, -18, 18))
                            end_y = int(y + np.clip(dy * 3, -18, 18))
                            cv2.arrowedLine(
                                ann,
                                (x, y),
                                (end_x, end_y),
                                (255, 255, 255),
                                1,
                                tipLength=0.3,
                            )

        if self.mode == "furrow":
            for f in self.furrows:
                obs_x = int(f.get("front_obs", f["x0"]))
                exp_x = int(f["expected_front"])

                color = (255, 80, 0) if f["alerted_blocked"] else (0, 200, 0)

                cv2.line(ann, (obs_x, f["y"] - 6), (obs_x, f["y"] + 6), color, 2)
                cv2.line(ann, (exp_x, f["y"] - 3), (exp_x, f["y"] + 3), (255, 255, 0), 1)

                if f["blocked_active"]:
                    cv2.putText(
                        ann,
                        "BLOCKED",
                        (max(0, obs_x - 22), f["y"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 80, 0),
                        1,
                        cv2.LINE_AA,
                    )

                if f["leak_active"] and f["leak_x"] is not None:
                    leak_y = f["y"] + 22
                    cv2.circle(ann, (int(f["leak_x"]), leak_y), int(f["leak_radius"] + 3), (255, 0, 0), 2)
                    cv2.putText(
                        ann,
                        "LEAK",
                        (int(f["leak_x"]) - 12, leak_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

        else:
            for s in self.sources:
                obs_r = max(1, int(s.get("radius_obs", 0)))
                exp_r = max(1, int(s["expected_radius"]))

                cv2.circle(ann, (s["x"], s["y"]), obs_r, (0, 255, 0), 1)
                cv2.circle(ann, (s["x"], s["y"]), exp_r, (255, 255, 0), 1)

                if s.get("leak_active", False):
                    lx, ly = s["leak_point"]
                    cv2.circle(ann, (lx, ly), int(s.get("leak_radius", 0)) + 4, (255, 0, 0), 2)
                    cv2.putText(
                        ann,
                        "LEAK",
                        (lx - 12, ly - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

            if self.radial_type == "flood" and self.obstruction_alerted:
                cv2.ellipse(ann, (235, 135), (70, 34), 0, 0, 360, (0, 140, 255), 2)
                cv2.putText(
                    ann,
                    "DRY POCKET",
                    (185, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 140, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.putText(
            ann,
            f"t={self.t:5.1f}s frame={self.frame_idx:03d} coverage={self.current_coverage * 100:4.1f}%",
            (8, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return ann

    def step(self):
        if self.completed:
            return

        self.t += self.dt

        if self.irrigation_on:
            self.water_time += self.dt

        if self.frame_idx == 115 and self.weeds.mow_count == 0:
            self.mow_event("Scheduled mowing to demonstrate weed regrowth")

        self.water_mask = self.update_water_current()
        self.prev_water_mask = self.water_mask

        raw = self.render_raw(self.water_mask)
        gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)

        flow = compute_optical_flow(self.prev_gray, gray)
        segmented = segment_water(raw)

        self.process_measurements(segmented, flow)
        self.weeds.step(self.dt, self.water_mask)

        annotated = self.annotate(raw, segmented, flow)
        jpg = encode_rgb(annotated)

        if jpg is not None:
            self.history_frames.append(jpg)

        self.prev_gray = gray
        self.frame_idx += 1

        if self.frame_idx >= MAX_FRAMES:
            self.completed = True

    def current_summary(self, status=None):
        speeds = self.metrics["speed_m_s"]
        avg_speed = float(np.mean(speeds[5:])) if len(speeds) > 5 else 0.0

        return {
            "Start": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "End": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Scenario": self.scenario,
            "Duration s": round(self.t, 1),
            "Frames": self.frame_idx,
            "Coverage %": round(self.current_coverage * 100, 1),
            "Avg speed m/s": round(avg_speed, 4),
            "Alerts": len(self.alerts),
            "Weed mow events": self.weeds.mow_count,
            "Status": status or ("Completed" if self.completed else "In progress"),
        }


# ----------------------------------------------------------------------------
# Streamlit state and controls
# ----------------------------------------------------------------------------
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

if "history_events" not in st.session_state:
    st.session_state.history_events = generate_historical_events()

st.sidebar.title("AquaFlow controls")

scenario = st.sidebar.selectbox(
    "Demo scenario",
    SCENARIOS,
    index=0,
    help="All scenarios are synthetic but physically motivated. They auto-start with live CV processing.",
)

if st.sidebar.button("Reset / reload demo"):
    st.session_state.reset_counter += 1

engine_key_changed = (
    "engine" not in st.session_state
    or st.session_state.get("engine_scenario") != scenario
    or st.session_state.get("engine_reset") != st.session_state.reset_counter
)

if engine_key_changed:
    seed = 11 + st.session_state.reset_counter
    st.session_state.engine = IrrigationSim(scenario, seed=seed)
    st.session_state.engine_scenario = scenario
    st.session_state.engine_reset = st.session_state.reset_counter
    st.session_state.playing = True
    st.session_state.frame_view = None
    st.session_state.demo_gif = None

engine = st.session_state.engine

if "playing" not in st.session_state:
    st.session_state.playing = True

if "playback_speed" not in st.session_state:
    st.session_state.playback_speed = 1.0

engine.irrigation_on = st.sidebar.toggle(
    "Irrigation on",
    value=engine.irrigation_on,
    help="Turn irrigation off to observe recession, stagnation, and paused front movement.",
)

st.session_state.playback_speed = st.sidebar.slider(
    "Playback speed",
    min_value=0.5,
    max_value=4.0,
    value=float(st.session_state.playback_speed),
    step=0.5,
)

if st.sidebar.button("Mow / disturb vegetation"):
    engine.mow_event("Manual mowing")
    if not st.session_state.playing and not engine.completed:
        engine.step()
    st.session_state.frame_view = len(engine.history_frames) - 1

st.sidebar.caption("No API key required. Demo data is generated locally and starts automatically.")

# ----------------------------------------------------------------------------
# Header metrics
# ----------------------------------------------------------------------------
st.title("💧 AquaFlow Vision")
st.caption(
    "AI-powered irrigation water-flow tracking using water segmentation, optical flow, "
    "front tracking, anomaly detection, and weed lifecycle/regrowth modeling."
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Sim time", f"{engine.t:.1f} s")
c2.metric("Frame", f"{engine.frame_idx}/{MAX_FRAMES}")
c3.metric("Coverage", f"{engine.current_coverage * 100:.1f}%")
c4.metric("Front speed", f"{engine.current_speed_m_s:.3f} m/s")
c5.metric("Active alerts", len(engine.alerts))
c6.metric("Weed mow events", engine.weeds.mow_count)

tab_live, tab_flow, tab_spatial, tab_weed, tab_alerts, tab_history, tab_eval, tab_about = st.tabs(
    [
        "Live Demo",
        "Flow Analytics",
        "Spatial Map",
        "Weed Lifecycle",
        "Alerts",
        "History",
        "Evaluation",
        "About",
    ]
)

# ----------------------------------------------------------------------------
# Live demo tab
# ----------------------------------------------------------------------------
with tab_live:
    left, right = st.columns([2, 1])

    max_idx = max(0, len(engine.history_frames) - 1)

    if st.session_state.playing:
        view_idx = max_idx
    else:
        view_idx = st.session_state.get("frame_view", max_idx)
        if view_idx is None or view_idx < 0 or view_idx > max_idx:
            view_idx = max_idx

    with left:
        st.image(
            engine.history_frames[view_idx],
            caption="Live computer-vision water tracking: boundary, optical flow, fronts, anomalies",
            use_column_width=True,
        )

        b1, b2, b3 = st.columns([1, 1, 2])

        with b1:
            if st.button("Pause live demo" if st.session_state.playing else "Resume live demo"):
                st.session_state.playing = not st.session_state.playing
                if not st.session_state.playing:
                    st.session_state.frame_view = max_idx
                else:
                    st.session_state.frame_view = None

        with b2:
            if st.button("Step +1"):
                st.session_state.playing = False
                if not engine.completed:
                    engine.step()
                st.session_state.frame_view = len(engine.history_frames) - 1

        with b3:
            if not st.session_state.playing and max_idx > 0:
                idx = st.slider("Frame review", 0, max_idx, view_idx, key="frame_slider")
                st.session_state.frame_view = idx
            else:
                st.caption("Pause to scrub frame by frame.")

        if engine.completed:
            st.success("Simulation completed. Event stored in History. Reset to run another demo.")

    with right:
        if len(engine.history_frames) >= 30 and st.session_state.get("demo_gif") is None:
            with st.spinner("Preparing live demo clip..."):
                gif = make_gif(engine.history_frames, max_frames=30, duration=170)
                st.session_state.demo_gif = gif if gif is not None else b""

        if st.session_state.get("demo_gif"):
            st.caption("Auto-generated demo clip")
            display_gif(st.session_state.demo_gif, width=320)

        st.metric("Segmentation IoU", f"{engine.current_iou:.3f}")
        st.metric("Boundary F1", f"{engine.current_boundary_f1:.3f}")
        st.metric("Front MAE", f"{engine.current_front_mae:.1f} px")
        st.metric("Leak area", f"{engine.current_leak_area}px")

        st.subheader("Recent alerts")
        if engine.alerts:
            recent_alerts = pd.DataFrame(engine.alerts[-5:][::-1])
            st.dataframe(recent_alerts, use_container_width=True)
        else:
            st.info("No anomalies detected yet.")

# ----------------------------------------------------------------------------
# Flow analytics tab
# ----------------------------------------------------------------------------
with tab_flow:
    df = metrics_df(engine)

    if df.empty:
        st.info("Waiting for simulation metrics...")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            fig_front = go.Figure()
            fig_front.add_scatter(
                x=df["time"],
                y=df["exp_front"],
                mode="lines",
                name="Expected front",
                line=dict(dash="dash", color="gold"),
            )
            fig_front.add_scatter(
                x=df["time"],
                y=df["obs_front"],
                mode="lines",
                name="Observed front",
                line=dict(color="cyan"),
            )
            fig_front.update_layout(
                title="Expected vs observed water movement",
                xaxis_title="Time s",
                yaxis_title="Front position px",
                height=320,
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_front, use_container_width=True)

        with col_b:
            fig_cov = go.Figure()
            fig_cov.add_scatter(
                x=df["time"],
                y=df["coverage"],
                mode="lines",
                name="Coverage %",
                line=dict(color="mediumseagreen"),
            )
            fig_cov.update_layout(
                title="Irrigation coverage over time",
                xaxis_title="Time s",
                yaxis_title="Coverage %",
                height=320,
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_cov, use_container_width=True)

        fig_speed = go.Figure()
        fig_speed.add_scatter(
            x=df["time"],
            y=df["speed_m_s"],
            mode="lines",
            name="Propagation speed",
            line=dict(color="royalblue"),
        )
        fig_speed.update_layout(
            title="Approximate water propagation speed",
            xaxis_title="Time s",
            yaxis_title="Speed m/s",
            height=280,
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_speed, use_container_width=True)

# ----------------------------------------------------------------------------
# Spatial water distribution map tab
# ----------------------------------------------------------------------------
with tab_spatial:
    frames = max(1, engine.frame_idx + 1)
    cum_norm = engine.cumulative_obs / frames
    cum_smooth = cv2.GaussianBlur(cum_norm.astype(np.float32), (15, 15), 0)

    heat_u8 = np.clip(cum_smooth * 255, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(engine.background, 0.45, heat_rgb, 0.55, 0)

    left, right = st.columns([2, 1])

    with left:
        st.image(blended, caption="Spatial water-distribution map (cumulative wetting frequency)", use_column_width=True)

    with right:
        vals = cum_smooth[engine.coverage_mask > 0]

        if len(vals) > 0 and vals.mean() > 0:
            cv_val = float(vals.std() / vals.mean())
            uniformity = max(0.0, 1.0 - cv_val)
        else:
            uniformity = 0.0

        st.metric("Final coverage", f"{engine.current_coverage * 100:.1f}%")
        st.metric("Distribution uniformity", f"{uniformity:.3f}")
        st.metric("Cumulative frames integrated", frames)

        st.caption(
            "The map shows where water has been observed by the segmentation model. "
            "Blue/red hotspots indicate repeated wetting; cooler regions indicate poor distribution."
        )

# ----------------------------------------------------------------------------
# Weed lifecycle tab
# ----------------------------------------------------------------------------
with tab_weed:
    weed_df = pd.DataFrame(engine.weeds.history)

    left, right = st.columns([2, 1])

    with left:
        if weed_df.empty:
            st.info("Waiting for weed lifecycle history...")
        else:
            stage_labels = {
                "stage1": "Germinating",
                "stage2": "Juvenile",
                "stage3": "Mature",
                "stage4": "Senescent",
                "stage5": "Regrowth",
            }

            fig_stage = go.Figure()

            for key, label in stage_labels.items():
                fig_stage.add_scatter(
                    x=weed_df["time"],
                    y=weed_df[key],
                    name=label,
                    stackgroup="one",
                )

            if engine.weeds.last_mow_hours is not None:
                fig_stage.add_vline(
                    x=engine.weeds.last_mow_hours,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mowing",
                )

            fig_stage.update_layout(
                title="Weed lifecycle stages over simulated time",
                xaxis_title="Simulated hours",
                yaxis_title="Grid cell %",
                height=320,
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_stage, use_container_width=True)

            fig_bio = go.Figure()
            fig_bio.add_scatter(
                x=weed_df["time"],
                y=weed_df["biomass"],
                name="Mean active biomass",
                line=dict(color="darkgreen"),
            )
            fig_bio.add_scatter(
                x=weed_df["time"],
                y=weed_df["regrowth"],
                name="Mean regrowth biomass",
                line=dict(color="limegreen", dash="dot"),
            )
            fig_bio.update_layout(
                title="Biomass and post-mowing regrowth",
                xaxis_title="Simulated hours",
                yaxis_title="Biomass index",
                height=280,
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_bio, use_container_width=True)

    with right:
        st.image(
            engine.render_weed_map(),
            caption="Weed lifecycle/regrowth map",
            use_column_width=True,
        )

        st.metric("Simulated weed hours", f"{engine.weeds.hours:.1f}")
        st.metric("Mowing events", engine.weeds.mow_count)

        if st.button("Mow now", key="mow_tab"):
            engine.mow_event("Manual mowing from Weed tab")
            if not st.session_state.playing and not engine.completed:
                engine.step()
            st.session_state.frame_view = len(engine.history_frames) - 1

        st.caption(
            "This module tracks lifecycle and regrowth, not simple weed detection. "
            "Mowing reduces biomass, but cells with root reserves re-enter regrowth, especially where irrigation increases moisture."
        )

# ----------------------------------------------------------------------------
# Alerts tab
# ----------------------------------------------------------------------------
with tab_alerts:
    if engine.alerts:
        alerts_df = pd.DataFrame(engine.alerts[::-1])
        st.dataframe(alerts_df, use_container_width=True)
    else:
        st.info("No alerts yet.")

# ----------------------------------------------------------------------------
# History tab
# ----------------------------------------------------------------------------
with tab_history:
    if st.button("Store current irrigation event", key="store_history"):
        store_current_event(engine, manual=True)

    st.dataframe(st.session_state.history_events, use_container_width=True)

    csv = st.session_state.history_events.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download history CSV",
        data=csv,
        file_name="irrigation_history.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------------
# Evaluation tab
# ----------------------------------------------------------------------------
with tab_eval:
    a = engine.anom

    def safe_precision(tp, fp):
        return tp / (tp + fp) if (tp + fp) > 0 else 1.0

    def safe_recall(tp, fn):
        return tp / (tp + fn) if (tp + fn) > 0 else 1.0

    blocked_precision = safe_precision(a["TP_blocked"], a["FP_blocked"])
    blocked_recall = safe_recall(a["TP_blocked"], a["FN_blocked"])
    leak_precision = safe_precision(a["TP_leak"], a["FP_leak"])
    leak_recall = safe_recall(a["TP_leak"], a["FN_leak"])

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Segmentation IoU", f"{engine.current_iou:.3f}")
    m2.metric("Boundary F1", f"{engine.current_boundary_f1:.3f}")
    m3.metric("Blocked-flow precision", f"{blocked_precision:.3f}")
    m4.metric("Blocked-flow recall", f"{blocked_recall:.3f}")

    m5, m6, m7, m8 = st.columns(4)

    m5.metric("Leak precision", f"{leak_precision:.3f}")
    m6.metric("Leak recall", f"{leak_recall:.3f}")
    m7.metric("Front MAE", f"{engine.current_front_mae:.1f} px")
    m8.metric("Front MAE", f"{engine.current_front_mae * PX_TO_M:.3f} m")

    df = metrics_df(engine)

    if not df.empty:
        col1, col2 = st.columns(2)

        with col1:
            fig_iou = go.Figure()
            fig_iou.add_scatter(
                x=df["time"],
                y=df["iou"],
                name="Water mask IoU",
                line=dict(color="royalblue"),
            )
            fig_iou.add_scatter(
                x=df["time"],
                y=df["boundary_f1"],
                name="Boundary F1",
                line=dict(color="darkorange"),
            )
            fig_iou.update_layout(
                title="Water-boundary tracking accuracy",
                xaxis_title="Time s",
                yaxis_title="Score",
                height=300,
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_iou, use_container_width=True)

        with col2:
            fig_mae = go.Figure()
            fig_mae.add_scatter(
                x=df["time"],
                y=df["front_mae"],
                name="Front MAE px",
                line=dict(color="crimson"),
            )
            fig_mae.update_layout(
                title="Irrigation-front tracking error",
                xaxis_title="Time s",
                yaxis_title="MAE px",
                height=300,
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_mae, use_container_width=True)

    counts_df = pd.DataFrame(
        [
            {
                "Anomaly": "Blocked / stagnation",
                "True positives": a["TP_blocked"],
                "False positives": a["FP_blocked"],
                "False negatives": a["FN_blocked"],
                "Precision": round(blocked_precision, 3),
                "Recall": round(blocked_recall, 3),
            },
            {
                "Anomaly": "Leakage / off-target water",
                "True positives": a["TP_leak"],
                "False positives": a["FP_leak"],
                "False negatives": a["FN_leak"],
                "Precision": round(leak_precision, 3),
                "Recall": round(leak_recall, 3),
            },
        ]
    )

    st.dataframe(counts_df, use_container_width=True)

# ----------------------------------------------------------------------------
# About tab
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### AquaFlow Vision

        This demo focuses on **physical water movement**, not crop tracking.

        The system simulates and analyzes:

        - Water segmentation from soil, vegetation, and weeds
        - Optical-flow estimation of water surface movement
        - Irrigation-front tracking
        - Approximate propagation speed
        - Unexpected stops and blocked channels
        - Leakage and off-target runoff zones
        - Coverage over time
        - Spatial water-distribution mapping
        - Expected vs observed hydraulic movement
        - Abnormal-flow alert generation
        - Historical irrigation event storage
        - Frame-by-frame replay
        - Water-boundary and anomaly-detection evaluation

        It also includes a distinct weed module focused on **weed lifecycle and regrowth**, not simple weed detection.
        Mowing disturbance, root reserves, moisture-dependent germination, senescence, and regrowth are modeled over time.

        The demo is fully synthetic and local. No API key is required.
        """
    )

# ----------------------------------------------------------------------------
# Automatic playback loop
# ----------------------------------------------------------------------------
if st.session_state.playing and not engine.completed:
    engine.step()

    if engine.completed:
        st.session_state.playing = False
        if not engine.event_stored:
            store_current_event(engine, manual=False)
        st.rerun()

    sleep_time = max(0.03, 1.0 / (FPS * float(st.session_state.get("playback_speed", 1.0))))
    time.sleep(sleep_time)
    st.rerun()