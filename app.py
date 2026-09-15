import os
import time
import requests
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import numpy as np


# ============================================================
# SMART PLANT GUARDIAN
# AI + IoT SMART FARMING ASSISTANT
# ============================================================

st.set_page_config(
    page_title="Smart Plant Guardian",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:#f5f8f6;
    color:#17221b;
}

.block-container {
    padding-top:1.15rem;
    padding-bottom:2rem;
    max-width:1400px;
}

.hero {
    background:linear-gradient(135deg,#163d2a,#286244);
    border-radius:22px;
    padding:28px 34px;
    margin-bottom:18px;
}

.hero h1,
.hero p {
    color:#fff !important;
}

.hero h1 {
    margin:0;
    font-size:2.25rem;
}

.hero p {
    margin:7px 0 0;
    font-size:1rem;
}

.section-title {
    color:#173d29 !important;
    font-size:1.45rem;
    font-weight:800;
    margin:22px 0 12px;
}

.card {
    background:#fff !important;
    border:1px solid #d9e4dc;
    border-radius:16px;
    padding:16px;
    box-shadow:0 3px 12px rgba(25,55,38,.07);
    color:#17221b !important;
    min-height:108px;
}

.card * {
    color:#17221b !important;
}

.metric {
    font-size:1.5rem;
    font-weight:800;
    color:#173d29 !important;
}

.label {
    font-size:.82rem;
    color:#53645a !important;
    font-weight:700;
}

.status {
    font-size:.86rem;
    font-weight:800;
    margin-top:5px;
}

.explain {
    font-size:.80rem;
    color:#53645a !important;
    margin-top:4px;
}

.flow {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:8px;
    flex-wrap:wrap;
    margin:12px 0 20px;
}

.flow-item {
    background:#e8f2ec !important;
    color:#173d29 !important;
    border:1px solid #bfd4c6;
    border-radius:999px;
    padding:10px 15px;
    font-weight:800;
}

.arrow {
    color:#286244 !important;
    font-size:1.25rem;
    font-weight:900;
}

.info-box {
    background:#edf6f0 !important;
    border-left:5px solid #286244;
    border-radius:12px;
    padding:14px 17px;
    color:#173d29 !important;
}

.info-box * {
    color:#173d29 !important;
}

.warning-box {
    background:#fff7df !important;
    border-left:5px solid #d49a16;
    border-radius:12px;
    padding:14px 17px;
    color:#5b4305 !important;
}

.warning-box * {
    color:#5b4305 !important;
}

.danger-box {
    background:#fff0ee !important;
    border-left:5px solid #c84b3c;
    border-radius:12px;
    padding:14px 17px;
    color:#64251f !important;
}

.danger-box * {
    color:#64251f !important;
}

.feature {
    background:#fff !important;
    border:1px solid #d9e4dc;
    border-radius:14px;
    padding:14px 16px;
    color:#17221b !important;
    min-height:112px;
    margin-bottom:12px;
}

.feature * {
    color:#17221b !important;
}

.assistant-card {
    background:#fff !important;
    border:1px solid #d9e4dc;
    border-radius:16px;
    padding:18px;
    color:#17221b !important;
}

.assistant-card * {
    color:#17221b !important;
}

.footer {
    text-align:center;
    color:#637269 !important;
    padding:20px 0 5px;
    font-size:.82rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# BLYNK
# ============================================================

BLYNK_URL = "https://blynk.cloud/external/api/getAll"


DEFAULT_PIN_MAP = {
    "temperature": "v0",
    "humidity": "v1",
    "soil": "v2",
    "tank": "v3",
    "pump": "v4",
    "alert": "v5",
    "light": "v6",
    "waterlog": "v7",
    "flow": "v8",
    "priority": "v9",
    "rain": "v10",
    "safety_pump": "v11",
}


def get_blynk_token():
    token = ""

    for key in ("BLYNK_AUTH_TOKEN", "BLYNK_TOKEN"):
        try:
            token = st.secrets.get(key, "")
        except Exception:
            token = ""

        if token:
            return token

    return (
        os.getenv("BLYNK_AUTH_TOKEN", "")
        or os.getenv("BLYNK_TOKEN", "")
    )


def get_pin_map():
    pin_map = DEFAULT_PIN_MAP.copy()

    try:
        custom = st.secrets.get("PIN_MAP", {})

        if isinstance(custom, dict):
            for key, value in custom.items():
                if key in pin_map and value:
                    value = str(value).lower()

                    if not value.startswith("v"):
                        value = "v" + value

                    pin_map[key] = value
    except Exception:
        pass

    return pin_map


def get_blynk_data():
    token = get_blynk_token()

    if not token:
        return {}

    try:
        response = requests.get(
            BLYNK_URL,
            params={"token": token},
            headers={"Cache-Control": "no-cache"},
            timeout=3,
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):
            return data

        return {}

    except Exception:
        return {}


def get_value(data, pin):
    raw = data.get(pin)

    if raw is None:
        return None

    try:
        return float(raw)
    except Exception:
        return None


# ============================================================
# SENSOR INTERPRETATION
# ============================================================

def sensor_status_temperature(v):

    if v is None:
        return "Unavailable", "Waiting for sensor data."

    if v >= 40:
        return "Abnormal — High", "Heat conditions are severe."

    if v >= 35:
        return "Warning — High", \
            "Temperature is elevated; monitor crop heat stress."

    if v < 15:
        return "Warning — Low", \
            "Temperature is low; monitor crop conditions."

    return "Normal", \
        "Temperature is within the working monitoring range."


def sensor_status_humidity(v):

    if v is None:
        return "Unavailable", "Waiting for sensor data."

    if v >= 90:
        return "Warning — Very humid", \
            "High humidity may increase disease risk."

    if v >= 80:
        return "Warning — High", \
            "Humidity is high; monitor leaf-disease conditions."

    if v < 35:
        return "Warning — Low", \
            "Dry air may increase water stress."

    return "Normal", \
        "Humidity is within the working monitoring range."


def soil_info(v):

    if v is None:
        return "Unavailable", "Waiting for soil sensor data."

    if v >= 3000:
        return "Dry", \
            "Soil is dry; irrigation may be required."

    if v >= 1800:
        return "Moderately moist", \
            "Soil has some moisture; continue monitoring."

    return "Moist", \
        "Soil moisture is currently adequate."


def tank_info(v):

    if v is None:
        return "Unavailable", "Waiting for tank sensor data."

    if v >= 2500:
        return "High", \
            "Tank level is high; monitor for excessive filling."

    if v >= 1000:
        return "Available", \
            "Water is available for irrigation."

    return "Low / unavailable", \
        "Water availability is insufficient for normal irrigation."


def light_info(v):

    if v is None:
        return "Unavailable", "Waiting for light sensor data."

    if v < 500:
        return "Low light", \
            "Light level is low; monitor crop exposure."

    return "Normal", \
        "Light level is above the low-light warning threshold."


def rain_info(v):

    if v is None:
        return "Unavailable", \
            "Rain sensor data unavailable."

    if v < 2500:
        return "Rain detected", \
            "Avoid unnecessary irrigation while rain is detected."

    return "No rain detected", \
        "No rain signal is currently detected."


def waterlog_info(v):

    if v is None:
        return "Unavailable", \
            "Waiting for waterlogging sensor data."

    if v >= 1000:
        return "Waterlogging detected", \
            "Stop irrigation and allow excess water to drain."

    return "Normal", \
        "No significant standing-water signal detected."


def pump_info(v):

    if v is None:
        return "Unavailable", \
            "Pump status unavailable."

    if v > 0:
        return "ON", \
            "Irrigation pump is currently active."

    return "OFF", \
        "Irrigation pump is currently inactive."


def safety_pump_info(v):

    if v is None:
        return "Unavailable", \
            "Safety pump status unavailable."

    if v > 0:
        return "ON", \
            "Safety pump is currently active."

    return "OFF", \
        "Safety pump is currently inactive."


def flow_info(v):

    if v is None:
        return "Unavailable", \
            "Waiting for flow data."

    lpm = v / 10.0

    if lpm <= 0.05:
        return f"{lpm:.2f} L/min", \
            "No meaningful water flow is currently detected."

    return f"{lpm:.2f} L/min", \
        "Water flow is being detected."


# ============================================================
# RISK CALCULATIONS
# ============================================================

def clamp(x):
    return max(0, min(100, float(x)))


def drought_score(
    soil,
    temp,
    humidity,
    tank,
    rain,
    waterlog
):

    score = 0

    if soil is not None:

        if soil >= 3000:
            score += 55

        elif soil >= 2400:
            score += 40

        elif soil >= 1800:
            score += 22

    if temp is not None:

        if temp >= 35:
            score += 18

        elif temp >= 32:
            score += 10

    if humidity is not None:

        if humidity < 40:
            score += 15

        elif humidity < 55:
            score += 8

    if tank is not None and tank < 1000:
        score += 12

    if rain is not None and rain < 2500:
        score -= 15

    if waterlog is not None and waterlog >= 1000:
        score -= 20

    return int(clamp(score))


def heat_score(temp, humidity):

    if temp is None:
        return 0

    score = 0

    if temp >= 40:
        score += 75

    elif temp >= 38:
        score += 60

    elif temp >= 35:
        score += 42

    elif temp >= 32:
        score += 22

    if humidity is not None:

        if humidity >= 80:
            score += 20

        elif humidity >= 65:
            score += 10

    return int(clamp(score))


def waterlog_score(v):

    if v is None:
        return 0

    if v >= 1800:
        return 100

    if v >= 1400:
        return 80

    if v >= 1000:
        return 60

    if v > 500:
        return 25

    return 0


def agricultural_risk(
    drought,
    heat,
    waterlog,
    tank,
    disease=0,
    pest=0,
    nutrient=0
):

    tank_risk = 100 if (
        tank is not None and tank < 1000
    ) else 0

    return int(
        clamp(
            drought * .25
            + heat * .18
            + waterlog * .18
            + tank_risk * .10
            + disease * .14
            + pest * .05
            + nutrient * .10
        )
    )


def severity(score):

    if score >= 75:
        return "High"

    if score >= 45:
        return "Moderate"

    if score >= 20:
        return "Mild"

    return "Low"


# ============================================================
# DISEASE MODEL
# ============================================================

CLASS_NAMES = [

    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",

    "Blueberry___healthy",

    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",

    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",

    "Orange___Haunglongbing_(Citrus_greening)",

    "Peach___Bacterial_spot",
    "Peach___healthy",

    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",

    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",

    "Raspberry___healthy",

    "Soybean___healthy",

    "Squash___Powdery_mildew",

    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",

    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


@st.cache_resource
def load_disease_model():

    model_path = "model/mobilenetv2_plant.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(
            model.classifier[1].in_features,
            len(CLASS_NAMES)
        )
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    # --------------------------------------------------------
    # Support both:
    # 1. Plain state_dict
    # 2. Checkpoints containing state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove possible "module." prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]

        cleaned_state_dict[new_key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.eval()

    return model


@st.cache_resource
def load_pest_model():

    model_path = "model/best.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pest model file not found: {model_path}"
        )

    return YOLO(model_path)


transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def disease_predict(image):

    """
    Returns:

        (label, confidence)

    if MobileNetV2 successfully produces a prediction.

    Returns:

        (None, 0.0)

    ONLY when the model cannot produce a prediction.
    """

    try:

        model = load_disease_model()

        x = transform(
            image.convert("RGB")
        ).unsqueeze(0)

        with torch.no_grad():

            logits = model(x)

            probs = torch.softmax(
                logits,
                dim=1
            )[0]

        best_index = int(
            torch.argmax(probs).item()
        )

        best_confidence = float(
            probs[best_index].item()
        ) * 100.0

        best_label = CLASS_NAMES[best_index]

        return best_label, best_confidence

    except Exception as e:

        return None, 0.0


def disease_model_status():

    try:

        load_disease_model()

        return True, "MobileNetV2 model loaded successfully."

    except Exception as e:

        return False, str(e)


def clean_label(name):

    return (
        name
        .replace("___", " — ")
        .replace("_", " ")
    )


def disease_confidence(conf):

    if conf >= 80:
        return "High-confidence prediction"

    if conf >= 60:
        return "Moderate-confidence prediction"

    if conf >= 40:
        return "Low-confidence — verify visually"

    return "Very low confidence — treat as uncertain"


def disease_direction(label, conf):

    text = label.lower()

    if "healthy" in text:

        return (
            "🌿 Direction: The leaf appears consistent "
            "with the healthy class. Continue regular monitoring."
        )

    if "bacterial" in text:

        return (
            "🦠 Direction: Possible bacterial disease pattern. "
            "Inspect spots and affected areas and confirm before treatment."
        )

    if "early blight" in text:

        return (
            "🦠 Direction: Possible early blight pattern. "
            "Inspect lower/older leaves and confirm visually."
        )

    if "late blight" in text:

        return (
            "🦠 Direction: Possible late blight pattern. "
            "Inspect rapidly spreading lesions and confirm urgently."
        )

    if "leaf mold" in text:

        return (
            "🦠 Direction: Possible leaf-mold pattern. "
            "Check humidity and leaf-surface symptoms."
        )

    if "septoria" in text:

        return (
            "🦠 Direction: Possible Septoria leaf-spot pattern. "
            "Inspect for small dark lesions and confirm visually."
        )

    if "target spot" in text:

        return (
            "🦠 Direction: Possible target-spot pattern. "
            "Inspect circular lesions and confirm before treatment."
        )

    if "yellow leaf curl" in text:

        return (
            "🦠 Direction: Possible Tomato Yellow Leaf Curl Virus pattern. "
            "Check for curling/yellowing and inspect for whiteflies."
        )

    if "mosaic" in text:

        return (
            "🦠 Direction: Possible mosaic-virus pattern. "
            "Inspect for mottling and confirm through proper diagnosis."
        )

    if "spider" in text:

        return (
            "🐛 Direction: Possible spider-mite-related damage. "
            "Inspect leaf undersides and webbing."
        )

    if "powdery mildew" in text:

        return (
            "🦠 Direction: Possible powdery-mildew pattern. "
            "Inspect for white powder-like growth."
        )

    return (
        "🔎 Direction: The model identified a visual disease/class pattern. "
        "Verify the result through field inspection before treatment."
    )


# ============================================================
# GREEN GRAM FALLBACK
# ============================================================

def green_gram_health_screening(image):

    """
    This is NOT a disease classifier.

    It is only a visual Green Gram health/stress screening
    that runs when MobileNetV2 cannot return a prediction.
    """

    arr = np.asarray(
        image.convert("RGB").resize((224, 224))
    ).astype(np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green = (
        (g > r * 1.05)
        & (g > b * 1.03)
    )

    yellow = (
        (r > b * 1.18)
        & (g > b * 1.12)
        & (r > 70)
    )

    brown = (
        (r > g * 1.12)
        & (g > b * 1.05)
        & (r > 70)
        & (g < 170)
    )

    green_ratio = float(green.mean())
    yellow_ratio = float(yellow.mean())
    brown_ratio = float(brown.mean())

    stress = int(
        clamp(
            yellow_ratio * 120
            + brown_ratio * 100
            - green_ratio * 20
        )
    )

    if stress >= 65:

        status = "High visible stress"

        recommendation = (
            "Inspect Green Gram leaves closely for "
            "disease, pest damage, or nutrient stress."
        )

    elif stress >= 40:

        status = "Moderate visible stress"

        recommendation = (
            "Monitor Green Gram leaves closely and "
            "inspect yellow or brown areas."
        )

    elif stress >= 20:

        status = "Mild visible stress"

        recommendation = (
            "Green Gram appears mildly stressed; "
            "continue monitoring."
        )

    else:

        status = "Low visible stress"

        recommendation = (
            "Green Gram appears visually healthy; "
            "continue routine monitoring."
        )

    return (
        stress,
        status,
        recommendation
    )


# ============================================================
# PEST DETECTION
# ============================================================

def pest_predict(image):

    try:

        model = load_pest_model()

        results = model.predict(
            source=np.array(
                image.convert("RGB")
            ),
            conf=0.25,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                cid = int(
                    box.cls[0]
                )

                conf = float(
                    box.conf[0]
                ) * 100

                detections.append(
                    (
                        result.names[cid],
                        conf
                    )
                )

        detections.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return detections

    except Exception:

        return []


# ============================================================
# NUTRIENT STRESS
# ============================================================

def nutrient_assessment(image):

    arr = np.asarray(
        image.convert("RGB").resize((224, 224))
    ).astype(np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green = (
        (g > r * 1.05)
        & (g > b * 1.03)
    )

    yellow = (
        (r > b * 1.18)
        & (g > b * 1.12)
        & (r > 70)
    )

    brown = (
        (r > g * 1.12)
        & (g > b * 1.05)
        & (r > 70)
        & (g < 170)
    )

    green_ratio = float(green.mean())
    yellow_ratio = float(yellow.mean())
    brown_ratio = float(brown.mean())

    score = int(
        clamp(
            yellow_ratio * 120
            + brown_ratio * 100
            - green_ratio * 20
        )
    )

    if score >= 65:
        level = "High visible stress"

    elif score >= 40:
        level = "Moderate visible stress"

    elif score >= 20:
        level = "Mild visible stress"

    else:
        level = "Low visible stress"

    if yellow_ratio > .22 and green_ratio < .50:

        pattern = (
            "Possible nitrogen-related chlorosis"
        )

    elif yellow_ratio > .16 and green_ratio < .58:

        pattern = (
            "Possible magnesium/iron-related chlorosis"
        )

    elif brown_ratio > .18:

        pattern = (
            "Possible potassium-related / edge-burn stress"
        )

    elif score >= 20:

        pattern = (
            "General visible nutrient-stress pattern"
        )

    else:

        pattern = (
            "No obvious nutrient-stress pattern"
        )

    return (
        score,
        level,
        pattern
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🌱 Smart Plant Guardian</h1>

<p>
AI + IoT Smart Farming Assistant for real-time crop,
water and environmental intelligence
</p>

</div>
""", unsafe_allow_html=True)


st.markdown(
    f"""
    <div class="card">
        <b>System time:</b>
        {time.strftime("%A, %d %B %Y • %I:%M:%S %p")}
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LIVE FARM MONITORING
# ============================================================

st.markdown(
    '<div class="section-title">📡 Live Farm Monitoring</div>',
    unsafe_allow_html=True
)


@st.fragment(run_every=2)
def live_dashboard():

    data = get_blynk_data()

    pin_map = get_pin_map()

    temp = get_value(
        data,
        pin_map["temperature"]
    )

    humidity = get_value(
        data,
        pin_map["humidity"]
    )

    soil = get_value(
        data,
        pin_map["soil"]
    )

    tank = get_value(
        data,
        pin_map["tank"]
    )

    pump = get_value(
        data,
        pin_map["pump"]
    )

    alert = get_value(
        data,
        pin_map["alert"]
    )

    light = get_value(
        data,
        pin_map["light"]
    )

    waterlog = get_value(
        data,
        pin_map["waterlog"]
    )

    flow = get_value(
        data,
        pin_map["flow"]
    )

    priority = get_value(
        data,
        pin_map["priority"]
    )

    rain = get_value(
        data,
        pin_map["rain"]
    )

    safety_pump = get_value(
        data,
        pin_map["safety_pump"]
    )

    temp_s, temp_e = sensor_status_temperature(temp)

    hum_s, hum_e = sensor_status_humidity(humidity)

    soil_s, soil_e = soil_info(soil)

    tank_s, tank_e = tank_info(tank)

    light_s, light_e = light_info(light)

    rain_s, rain_e = rain_info(rain)

    wl_s, wl_e = waterlog_info(waterlog)

    pump_s, pump_e = pump_info(pump)

    safety_s, safety_e = safety_pump_info(
        safety_pump
    )

    flow_s, flow_e = flow_info(flow)

    drought = drought_score(
        soil,
        temp,
        humidity,
        tank,
        rain,
        waterlog
    )

    heat = heat_score(
        temp,
        humidity
    )

    wl_score = waterlog_score(
        waterlog
    )

    rows = [

        [

            (
                "Temperature",
                f"{temp:.1f} °C"
                if temp is not None
                else "N/A",
                temp_s,
                temp_e
            ),

            (
                "Humidity",
                f"{humidity:.1f} %"
                if humidity is not None
                else "N/A",
                hum_s,
                hum_e
            ),

            (
                "Soil Sensor",
                f"{soil:.0f}"
                if soil is not None
                else "N/A",
                soil_s,
                soil_e
            ),

            (
                "Tank Water",
                tank_s,
                tank_s,
                tank_e
            ),

            (
                "Irrigation Pump",
                pump_s,
                pump_s,
                pump_e
            ),

        ],

        [

            (
                "Light Sensor",
                f"{light:.0f}"
                if light is not None
                else "N/A",
                light_s,
                light_e
            ),

            (
                "Rain Sensor",
                rain_s,
                rain_s,
                rain_e
            ),

            (
                "Waterlogging",
                wl_s,
                wl_s,
                wl_e
            ),

            (
                "Water Flow",
                flow_s,
                (
                    "Normal"
                    if flow is not None and flow > .05
                    else "No flow"
                ),
                flow_e
            ),

            (
                "Safety Pump",
                safety_s,
                safety_s,
                safety_e
            ),

        ],

        [

            (
                "Irrigation Priority",
                f"{priority:.0f}/100"
                if priority is not None
                else "N/A",
                "Live",
                "Priority comes from the ESP32 irrigation decision."
            ),

            (
                "Drought Stress",
                f"{drought}/100",
                severity(drought),
                "Estimated from available sensor conditions."
            ),

            (
                "Heat Stress",
                f"{heat}/100",
                severity(heat),
                "Estimated from temperature and humidity."
            ),

            (
                "Waterlogging Stress",
                f"{wl_score}/100",
                severity(wl_score),
                "Estimated from the field waterlogging sensor."
            ),

            (
                "Agricultural Risk",
                f"{agricultural_risk(drought, heat, wl_score, tank)}/100",
                severity(
                    agricultural_risk(
                        drought,
                        heat,
                        wl_score,
                        tank
                    )
                ),
                "Combined current farm risk."
            ),

        ],
    ]

    for row in rows:

        cols = st.columns(5)

        for col, item in zip(cols, row):

            label, val, status, explain = item

            with col:

                st.markdown(
                    f"""
                    <div class="card">

                        <div class="label">
                            {label}
                        </div>

                        <div class="metric">
                            {val}
                        </div>

                        <div class="status">
                            {status}
                        </div>

                        <div class="explain">
                            {explain}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

    risk = agricultural_risk(
        drought,
        heat,
        wl_score,
        tank
    )

    st.markdown(
        f"""
        <div class="info-box">

        <b>Current farm interpretation:</b>

        Drought/water stress
        <b>{drought}/100</b>
        •
        Heat stress
        <b>{heat}/100</b>
        •
        Waterlogging stress
        <b>{wl_score}/100</b>
        •
        Current agricultural risk
        <b>{risk}/100</b>.

        </div>
        """,
        unsafe_allow_html=True
    )

    if not data:

        st.markdown(
            """
            <div class="warning-box">

            <b>Blynk data unavailable.</b>

            The dashboard is waiting for live device data.

            </div>
            """,
            unsafe_allow_html=True
        )

    return {

        "data": data,

        "temp": temp,

        "humidity": humidity,

        "soil": soil,

        "tank": tank,

        "pump": pump,

        "alert": alert,

        "light": light,

        "waterlog": waterlog,

        "flow": flow,

        "priority": priority,

        "rain": rain,

        "safety_pump": safety_pump,

        "drought": drought,

        "heat": heat,

        "waterlog_score": wl_score,

        "risk": risk,

    }


live = live_dashboard()


# ============================================================
# DECISION FLOW
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Smart Decision Flow</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="flow">

        <div class="flow-item">MONITOR</div>

        <div class="arrow">→</div>

        <div class="flow-item">ANALYZE</div>

        <div class="arrow">→</div>

        <div class="flow-item">DECIDE</div>

        <div class="arrow">→</div>

        <div class="flow-item">ACT</div>

        <div class="arrow">→</div>

        <div class="flow-item">ALERT</div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CROP AI ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">🔬 AI Crop-Health Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">

    <b>AI pipeline:</b>
    MobileNetV2 disease prediction →
    Pest detection →
    Visual nutrient-stress assessment.

    <br><br>

    <b>Fallback:</b>
    Green Gram visual screening is used
    <b>only when MobileNetV2 cannot return a prediction.</b>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL STATUS
# ============================================================

model_loaded, model_message = disease_model_status()

if model_loaded:

    st.success(
        "✅ MobileNetV2: Model loaded and ready."
    )

else:

    st.warning(
        "⚠️ MobileNetV2: Model is not available. "
        "Green Gram fallback will be used if an image is uploaded."
    )

    with st.expander("View model status details"):

        st.code(
            model_message
        )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload a clear crop / leaf image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# DEFAULT AI VARIABLES
# ============================================================

disease_label = None
disease_conf = 0.0

green_gram_screening_active = False

green_gram_score = 0

green_gram_status = "Not analyzed"

green_gram_recommendation = ""

pests = []

nutrient_score = 0

nutrient_level = "Not analyzed"

nutrient_pattern = "Not analyzed"

disease_top_conf = 0

pest_top_conf = 0


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded:

    try:

        image = Image.open(
            uploaded
        ).convert("RGB")

    except Exception:

        st.error(
            "Unable to read the uploaded image."
        )

        image = None


    if image is not None:

        c1, c2 = st.columns(
            [1, 1.4]
        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        with c1:

            st.image(
                image,
                caption="Uploaded crop image",
                use_container_width=True
            )


        # ----------------------------------------------------
        # DISEASE
        # ----------------------------------------------------

        with c2:

            st.markdown(
                "#### 🦠 Disease Detection — Single Best Prediction"
            )

            # IMPORTANT:
            # MobileNetV2 ALWAYS runs first.
            disease_label, disease_conf = disease_predict(
                image
            )

            # ------------------------------------------------
            # CASE 1:
            # MobileNetV2 returned a prediction
            # ------------------------------------------------

            if disease_label is not None:

                readable = clean_label(
                    disease_label
                )

                disease_top_conf = disease_conf

                st.success(
                    f"🌿 Prediction: {readable}"
                )

                st.metric(
                    "Confidence",
                    f"{disease_conf:.2f}%"
                )

                st.caption(
                    disease_confidence(
                        disease_conf
                    )
                )

                st.write(
                    disease_direction(
                        readable,
                        disease_conf
                    )
                )

            # ------------------------------------------------
            # CASE 2:
            # MobileNetV2 returned NOTHING
            #
            # THIS IS THE ONLY CONDITION THAT ACTIVATES
            # GREEN GRAM FALLBACK.
            # ------------------------------------------------

            else:

                green_gram_screening_active = True

                st.info(
                    "MobileNetV2 did not return a usable prediction. "
                    "Switching to Green Gram visual screening."
                )

                (
                    green_gram_score,
                    green_gram_status,
                    green_gram_recommendation
                ) = green_gram_health_screening(
                    image
                )

                st.markdown(
                    "##### 🌱 Green Gram Visual Screening"
                )

                a, b = st.columns(2)

                with a:

                    st.metric(
                        "Visible Stress Score",
                        f"{green_gram_score}/100"
                    )

                with b:

                    st.metric(
                        "Visual Status",
                        green_gram_status
                    )

                st.caption(
                    green_gram_recommendation
                )

                st.markdown(
                    """
                    <div class="warning-box">

                    <b>Important:</b>
                    This is a visual Green Gram
                    health screening, not a disease diagnosis.

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ------------------------------------------------
            # PEST DETECTION
            # ------------------------------------------------

            st.markdown(
                "#### 🐛 Pest Detection"
            )

            pests = pest_predict(
                image
            )

            if pests:

                for pest, conf in pests:

                    st.write(
                        f"**{pest} — {conf:.2f}%**"
                    )

                    st.caption(
                        (
                            "Low-confidence detection — "
                            "verify visually."
                            if conf < 50
                            else
                            "Model detection — "
                            "verify before treatment."
                        )
                    )

                pest_top_conf = pests[0][1]

            else:

                st.success(
                    "No pest detected at the current detection threshold."
                )


        # ====================================================
        # NUTRIENT STRESS
        # ====================================================

        (
            nutrient_score,
            nutrient_level,
            nutrient_pattern
        ) = nutrient_assessment(
            image
        )

        st.markdown(
            "#### 🌿 Visual Nutrient-Stress Assessment"
        )

        a, b, c = st.columns(3)

        with a:

            st.markdown(
                f"""
                <div class="card">

                    <div class="label">
                        Visual Stress Score
                    </div>

                    <div class="metric">
                        {nutrient_score}/100
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with b:

            st.markdown(
                f"""
                <div class="card">

                    <div class="label">
                        Severity
                    </div>

                    <div class="metric">
                        {nutrient_level}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c:

            st.markdown(
                f"""
                <div class="card">

                    <div class="label">
                        Possible Pattern
                    </div>

                    <div class="explain">
                        <b>{nutrient_pattern}</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown(
            """
            <div class="warning-box">

            <b>Preliminary visual assessment:</b>

            This does not measure soil or leaf NPK concentration.
            Confirm suspected deficiency with appropriate
            agricultural testing before fertilizer application.

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CROP STRESS & AGRICULTURAL RISK
# ============================================================

st.markdown(
    '<div class="section-title">📊 Crop Stress & Agricultural Risk</div>',
    unsafe_allow_html=True
)

drought = live["drought"]

heat = live["heat"]

waterlog_s = live["waterlog_score"]


# IMPORTANT:
# If Green Gram fallback is active, disease contribution = 0.
# We do NOT treat Green Gram visual screening as disease confidence.

disease_risk_value = (
    disease_top_conf
    if disease_label is not None
    else 0
)


risk = agricultural_risk(

    drought,

    heat,

    waterlog_s,

    live["tank"],

    disease_risk_value,

    pest_top_conf
    if pests
    else 0,

    nutrient_score
)


c1, c2, c3, c4 = st.columns(4)


for col, title, score in [

    (
        c1,
        "Drought / Water Stress",
        drought
    ),

    (
        c2,
        "Heat Stress",
        heat
    ),

    (
        c3,
        "Waterlogging Stress",
        waterlog_s
    ),

    (
        c4,
        "Agricultural Risk",
        risk
    ),

]:

    with col:

        st.markdown(
            f"""
            <div class="card">

                <div class="label">
                    {title}
                </div>

                <div class="metric">
                    {score}/100
                </div>

                <div class="status">
                    {severity(score)} risk level
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FARMER RECOMMENDATION
# ============================================================

st.markdown(
    '<div class="section-title">👨‍🌾 Farmer Recommendation</div>',
    unsafe_allow_html=True
)


recommendations = []


if live["waterlog"] is not None and live["waterlog"] >= 1000:

    recommendations.append(
        "🚫 Avoid irrigation because waterlogging is detected."
    )


elif live["tank"] is not None and live["tank"] < 1000:

    recommendations.append(
        "💧 Tank water is low/unavailable; avoid unnecessary pump operation."
    )


elif live["soil"] is not None and live["soil"] >= 3000:

    recommendations.append(
        "💧 Soil is dry; irrigation can be considered if water is available and rain/waterlogging protection is clear."
    )


else:

    recommendations.append(
        "🌱 Soil is not currently in the dry range; continue monitoring."
    )


if heat >= 60:

    recommendations.append(
        "🌡️ Heat stress is high; prioritize crop protection and moisture monitoring."
    )

elif heat >= 35:

    recommendations.append(
        "🌡️ Moderate heat stress detected; monitor temperature and soil moisture."
    )


if drought >= 60:

    recommendations.append(
        "☀️ High drought/water stress detected; monitor water availability closely."
    )


if green_gram_screening_active:

    recommendations.append(
        "🌱 MobileNetV2 was unavailable for this image, so Green Gram visual screening was used."
    )


if disease_label is not None:

    readable_disease = clean_label(
        disease_label
    )

    if "healthy" not in readable_disease.lower():

        recommendations.append(
            f"🦠 AI disease indication: {readable_disease}. "
            "Verify visually before treatment."
        )

    else:

        recommendations.append(
            "🌿 AI result is consistent with a healthy crop class; continue routine monitoring."
        )


if pests:

    recommendations.append(
        f"🐛 Pest indication detected: {pests[0][0]}. "
        "Verify the affected plant area before control action."
    )


if nutrient_score >= 40:

    recommendations.append(
        "🌿 Visible nutrient-stress pattern detected; confirm with soil/plant testing before fertilizer application."
    )


for recommendation in recommendations:

    st.markdown(
        f"""
        <div class="info-box">
            {recommendation}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FARMER HELP ASSISTANT
# ============================================================

st.markdown(
    '<div class="section-title">👨‍🌾 Farmer Help Assistant</div>',
    unsafe_allow_html=True
)


questions = [

    "Should I irrigate now?",

    "Is my soil too dry?",

    "Is there a water shortage?",

    "Is the crop under heat stress?",

    "Is waterlogging detected?",

    "What should I do if a disease is suspected?",

    "What should I do if pests are detected?",

    "What does the agricultural risk score mean?",

]


selected_question = st.selectbox(

    "Select a common question",

    ["Choose a question..."]
    + questions

)


typed_question = st.text_input(

    "Or type your question",

    placeholder="Example: Should I water my crop now?"

)


question = (

    typed_question.strip()

    if typed_question.strip()

    else (

        ""

        if selected_question
        == "Choose a question..."

        else selected_question

    )

)


if question:

    q = question.lower()


    # --------------------------------------------------------
    # IRRIGATION
    # --------------------------------------------------------

    if (
        "irrigat" in q
        or "water now" in q
        or (
            "water" in q
            and "soil" in q
        )
    ):

        if (
            live["waterlog"] is not None
            and live["waterlog"] >= 1000
        ):

            answer = (
                "Do not irrigate now. "
                "Waterlogging is detected."
            )

        elif (
            live["rain"] is not None
            and live["rain"] < 2500
        ):

            answer = (
                "Rain is currently detected. "
                "Avoid unnecessary irrigation."
            )

        elif (
            live["tank"] is not None
            and live["tank"] < 1000
        ):

            answer = (
                "Avoid pump operation because "
                "tank water is currently low/unavailable."
            )

        elif (
            live["soil"] is not None
            and live["soil"] >= 3000
        ):

            answer = (
                "The soil is dry. If water is available "
                "and rain/waterlogging protection is clear, "
                "irrigation can be prioritized."
            )

        else:

            answer = (
                "The soil is not currently in the dry range. "
                "Continue monitoring before irrigating."
            )


    # --------------------------------------------------------
    # SOIL
    # --------------------------------------------------------

    elif "soil" in q:

        answer = soil_info(
            live["soil"]
        )[1]


    # --------------------------------------------------------
    # WATER
    # --------------------------------------------------------

    elif (
        "shortage" in q
        or "tank" in q
        or "water availability" in q
    ):

        answer = tank_info(
            live["tank"]
        )[1]


    # --------------------------------------------------------
    # HEAT
    # --------------------------------------------------------

    elif (
        "heat" in q
        or "temperature" in q
    ):

        answer = (

            f"Current heat-stress score is "
            f"{heat}/100 ({severity(heat)}). "
            "Monitor crop moisture and temperature."

        )


    # --------------------------------------------------------
    # WATERLOGGING
    # --------------------------------------------------------

    elif (
        "waterlogging" in q
        or "standing" in q
    ):

        answer = waterlog_info(
            live["waterlog"]
        )[1]


    # --------------------------------------------------------
    # DISEASE
    # --------------------------------------------------------

    elif "disease" in q:

        if disease_label is not None:

            answer = (

                f"Current AI indication: "
                f"{clean_label(disease_label)} "
                f"with {disease_conf:.2f}% confidence. "

                "This is preliminary decision support. "
                "Inspect the plant and confirm the suspected "
                "disease before treatment."

            )

        elif green_gram_screening_active:

            answer = (

                "MobileNetV2 did not return a usable prediction "
                "for the uploaded image. Green Gram visual screening "
                f"reported {green_gram_score}/100 visible stress "
                f"({green_gram_status}). "

                "This is not a disease diagnosis; inspect the crop "
                "and confirm the cause."

            )

        else:

            answer = (

                "Upload a crop image to run "
                "the AI disease analysis."

            )


    # --------------------------------------------------------
    # PEST
    # --------------------------------------------------------

    elif "pest" in q:

        if pests:

            answer = (

                f"The AI detected "
                f"{pests[0][0]} at "
                f"{pests[0][1]:.2f}% confidence. "

                "Inspect the affected crop area and "
                "verify before taking control action."

            )

        else:

            answer = (

                "No pest was detected at the current "
                "detection threshold."

            )


    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    elif "risk" in q:

        answer = (

            f"The current Agricultural Risk Score is "
            f"{risk}/100 ({severity(risk)}). "

            "It combines available water/soil, heat, "
            "waterlogging and crop-health signals."

        )


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    else:

        answer = (

            "I can help with irrigation, soil moisture, "
            "water availability, heat stress, waterlogging, "
            "crop disease, pests, nutrient stress, "
            "and agricultural risk."

        )


    st.markdown(

        f"""
        <div class="assistant-card">

            <b>🌱 Assistant:</b>

            <br><br>

            {answer}

        </div>
        """,

        unsafe_allow_html=True

    )


# ============================================================
# SMART FARMING CAPABILITIES
# ============================================================

st.markdown(
    '<div class="section-title">🌾 Smart Farming Capabilities</div>',
    unsafe_allow_html=True
)


features = [

    (
        "📡 Multi-Sensor Monitoring",

        "Temperature, humidity, soil moisture, light, rain, tank water, waterlogging and water flow."
    ),

    (
        "💧 Intelligent Irrigation",

        "Uses soil condition, environment and water availability instead of simple dry-soil switching."
    ),

    (
        "🌧️ Rain & Waterlogging Protection",

        "Prevents unnecessary irrigation when rain or excess field water is detected."
    ),

    (
        "🚰 Flow-Based Pump Fault Detection",

        "Uses water-flow feedback to identify missing or abnormal irrigation flow."
    ),

    (
        "📱 Blynk IoT Monitoring",

        "Live farm values, pump status, alerts, flow and irrigation priority."
    ),

    (
        "🔬 AI Disease Detection",

        "MobileNetV2 single-best crop disease prediction with confidence interpretation."
    ),

    (
        "🌱 Green Gram Fallback",

        "Visual Green Gram health screening is activated only when the disease model cannot return a prediction."
    ),

    (
        "🐛 Pest Detection",

        "YOLO-based visual pest screening with confidence-aware interpretation."
    ),

    (
        "🌿 Nutrient-Stress Screening",

        "Visual leaf-color assessment for possible nutrient-stress patterns."
    ),

    (
        "📊 Risk Intelligence",

        "Drought/water stress, heat stress, waterlogging stress and overall Agricultural Risk Score."
    ),

    (
        "👨‍🌾 Farmer Help Assistant",

        "Predefined questions plus free-text farmer questions with practical responses."
    ),

]


cols = st.columns(4)


for i, (title, desc) in enumerate(features):

    with cols[i % 4]:

        st.markdown(

            f"""
            <div class="feature">

                <b>{title}</b>

                <br><br>

                <span class="explain">
                    {desc}
                </span>

            </div>
            """,

            unsafe_allow_html=True

        )


# ============================================================
# OVERALL SYSTEM
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Overall System</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="info-box">

    <b>ESP32 + Sensors</b>
    →
    real-time farm monitoring
    →

    <b>Blynk IoT</b>
    →
    live farm state
    →

    <b>Intelligent Decision Layer</b>
    →
    irrigation / protection decisions
    →

    <b>AI Crop Analysis</b>
    →
    disease + pest + visual nutrient-stress screening
    →

    <b>Risk Engine</b>
    →
    drought + heat + waterlogging + crop-health signals
    →

    <b>Farmer Assistant</b>
    →
    practical recommendations.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SAFETY NOTE
# ============================================================

st.markdown(
    """
    <div class="warning-box">

    <b>Safety & interpretation note</b>

    <br><br>

    AI results are preliminary decision-support outputs,
    not guaranteed diagnoses.

    <br>

    Visual nutrient assessment does not measure NPK concentration.

    <br>

    Green Gram fallback is a visual health screening,
    not a disease classifier.

    <br>

    Confirm disease, pest or nutrient problems with field
    inspection and appropriate agricultural testing before treatment.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(

    '<div class="footer">'
    'Smart Plant Guardian • AI + IoT Smart Farming Assistant • SIH Prototype'
    '</div>',

    unsafe_allow_html=True

)
