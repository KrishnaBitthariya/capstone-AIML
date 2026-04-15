"""
NIDS Flask Backend
Routes:
  GET  /                  → dashboard
  GET  /api/simulate      → generate & classify one fake packet
  POST /api/predict       → classify a JSON packet manually
  GET  /api/stats         → model stats (accuracy, confusion matrix)
  GET  /api/model-info    → feature importances
"""

from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np
import random
import time
import os

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "model.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

bundle = load_model()
model   = bundle['model']
scaler  = bundle['scaler']
le      = bundle['label_encoder']
FEATURES = bundle['features']

# ── Helpers ───────────────────────────────────────────────────────────────────
ATTACK_COLORS = {
    'Normal':     '#00ff88',
    'DoS':        '#ff4444',
    'PortScan':   '#ffaa00',
    'BruteForce': '#ff00aa',
}

ATTACK_DESCRIPTIONS = {
    'Normal':     'Legitimate network traffic — no threat detected.',
    'DoS':        'Denial of Service: Flood attack overwhelming the target server.',
    'PortScan':   'Port Scanning: Attacker mapping open ports to find vulnerabilities.',
    'BruteForce': 'Brute Force Login: Repeated failed authentication attempts detected.',
}

def make_fake_packet(attack_type=None):
    """Generate a realistic fake network packet for simulation."""
    np.random.seed(int(time.time() * 1000) % 2**31)

    if attack_type is None:
        # Weighted so attacks are visible but not too frequent
        attack_type = random.choices(
            ['Normal', 'DoS', 'PortScan', 'BruteForce'],
            weights=[55, 20, 15, 10]
        )[0]

    templates = {
        'Normal': {
            'duration': round(np.random.exponential(2), 2),
            'src_bytes': int(np.random.normal(5000, 2000)),
            'dst_bytes': int(np.random.normal(4000, 1500)),
            'land': 0, 'wrong_fragment': 0, 'urgent': 0,
            'hot': random.randint(0, 4),
            'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0,
            'count': random.randint(1, 50),
            'srv_count': random.randint(1, 50),
            'serror_rate': round(random.uniform(0, 0.1), 3),
            'rerror_rate': round(random.uniform(0, 0.1), 3),
            'same_srv_rate': round(random.uniform(0.8, 1.0), 3),
            'diff_srv_rate': round(random.uniform(0, 0.1), 3),
            'dst_host_count': random.randint(50, 255),
            'dst_host_srv_count': random.randint(50, 255),
        },
        'DoS': {
            # Moderate DoS range: count 150-450, serror 0.5-0.95
            # Rule needs count>400 AND serror>0.85 — many packets fall below this
            # AI trained on the full distribution still classifies these correctly
            'duration': 0,
            'src_bytes': int(np.random.normal(40000, 15000)),
            'dst_bytes': int(np.random.normal(100, 50)),
            'land': random.choice([0, 1]),
            'wrong_fragment': random.randint(0, 3),
            'urgent': 0,
            'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0,
            'count': random.randint(150, 450),
            'srv_count': random.randint(150, 450),
            'serror_rate': round(random.uniform(0.50, 0.98), 3),
            'rerror_rate': round(random.uniform(0, 0.1), 3),
            'same_srv_rate': round(random.uniform(0.85, 1.0), 3),
            'diff_srv_rate': round(random.uniform(0, 0.05), 3),
            'dst_host_count': random.randint(150, 255),
            'dst_host_srv_count': random.randint(150, 255),
        },
        'PortScan': {
            # Stealthy scan range: diff_srv 0.6-0.92, rerror 0.55-0.85
            # Rule needs diff_srv>0.85 AND rerror>0.80 — stealthy scans evade it
            'duration': 0,
            'src_bytes': random.randint(50, 200),
            'dst_bytes': 0,
            'land': 0, 'wrong_fragment': 0, 'urgent': 0,
            'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0,
            'count': random.randint(1, 10),
            'srv_count': random.randint(1, 5),
            'serror_rate': round(random.uniform(0, 0.1), 3),
            'rerror_rate': round(random.uniform(0.55, 0.92), 3),
            'same_srv_rate': round(random.uniform(0, 0.1), 3),
            'diff_srv_rate': round(random.uniform(0.60, 0.95), 3),
            'dst_host_count': random.randint(1, 20),
            'dst_host_srv_count': random.randint(200, 255),
        },
        'BruteForce': {
            # Early-stage BF: failed_logins 1-6, some below the >5 threshold
            # Rule also requires logged_in==0 — distributed BF may not trigger this
            'duration': round(np.random.exponential(5), 2),
            'src_bytes': int(np.random.normal(1000, 300)),
            'dst_bytes': int(np.random.normal(500, 200)),
            'land': 0, 'wrong_fragment': 0, 'urgent': 0,
            'hot': random.randint(3, 15),
            'num_failed_logins': random.randint(1, 6),
            'logged_in': random.choice([0, 0, 0, 1]),  # mostly 0 but occasionally 1
            'num_compromised': random.randint(0, 5),
            'count': random.randint(10, 80),
            'srv_count': random.randint(10, 80),
            'serror_rate': round(random.uniform(0, 0.3), 3),
            'rerror_rate': round(random.uniform(0.4, 0.85), 3),
            'same_srv_rate': round(random.uniform(0.85, 1.0), 3),
            'diff_srv_rate': round(random.uniform(0, 0.1), 3),
            'dst_host_count': random.randint(1, 30),
            'dst_host_srv_count': random.randint(1, 30),
        }
    }

    pkt = templates[attack_type]
    pkt['src_bytes'] = max(0, pkt['src_bytes'])
    pkt['dst_bytes'] = max(0, pkt['dst_bytes'])

    # Random IPs
    src_ip  = f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    dst_ip  = f"192.168.{random.randint(0,5)}.{random.randint(1,50)}"
    port    = random.choice([22, 23, 80, 443, 3306, 8080, 21, 25, 53, random.randint(1024, 65535)])
    proto   = random.choice(['TCP', 'UDP', 'ICMP'])

    return pkt, attack_type, src_ip, dst_ip, port, proto


def classify_packet(pkt_dict):
    """Run both AI and rule-based classifiers."""
    vec = np.array([[pkt_dict[f] for f in FEATURES]])
    vec_scaled = scaler.transform(vec)

    # AI prediction
    ai_pred_enc  = model.predict(vec_scaled)[0]
    ai_label     = le.inverse_transform([ai_pred_enc])[0]
    ai_proba     = model.predict_proba(vec_scaled)[0]
    ai_confidence = round(float(max(ai_proba)) * 100, 1)
    ai_proba_dict = {cls: round(float(p)*100, 1) for cls, p in zip(le.classes_, ai_proba)}

    # Rule-based prediction — same deliberately-limited rules as training
    # Only checks 3 features, high thresholds → misses moderate attacks
    count         = pkt_dict['count']
    serror        = pkt_dict['serror_rate']
    diff_srv      = pkt_dict['diff_srv_rate']
    rerror        = pkt_dict['rerror_rate']
    failed_logins = pkt_dict['num_failed_logins']
    logged_in     = pkt_dict['logged_in']

    if count > 400 and serror > 0.85:
        rule_label = 'DoS'
    elif diff_srv > 0.85 and rerror > 0.80:
        rule_label = 'PortScan'
    elif failed_logins > 5 and logged_in == 0:
        rule_label = 'BruteForce'
    else:
        rule_label = 'Normal'

    return ai_label, ai_confidence, ai_proba_dict, rule_label


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/simulate')
def simulate():
    """Generate and classify one live packet."""
    pkt, true_label, src_ip, dst_ip, port, proto = make_fake_packet()
    ai_label, ai_conf, ai_proba, rule_label = classify_packet(pkt)

    return jsonify({
        'timestamp': time.strftime('%H:%M:%S'),
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'port': port,
        'protocol': proto,
        'features': pkt,
        'true_label': true_label,
        'ai': {
            'label': ai_label,
            'confidence': ai_conf,
            'probabilities': ai_proba,
            'color': ATTACK_COLORS[ai_label],
            'description': ATTACK_DESCRIPTIONS[ai_label],
            'correct': ai_label == true_label
        },
        'rule': {
            'label': rule_label,
            'color': ATTACK_COLORS[rule_label],
            'description': ATTACK_DESCRIPTIONS[rule_label],
            'correct': rule_label == true_label
        }
    })


@app.route('/api/stats')
def stats():
    return jsonify({
        'ai_accuracy':        round(bundle['ai_accuracy'] * 100, 2),
        'rule_accuracy':      round(bundle['rule_accuracy'] * 100, 2),
        'confusion_matrix':   bundle['confusion_matrix'],
        'rule_confusion_matrix': bundle.get('rule_confusion_matrix', []),
        'class_names':        bundle['class_names'],
        'ai_per_class':       bundle.get('ai_per_class', {}),
        'rule_per_class':     bundle.get('rule_per_class', {}),
    })


@app.route('/api/model-info')
def model_info():
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(FEATURES, importances.tolist()),
        key=lambda x: x[1], reverse=True
    )[:8]
    return jsonify({'feature_importances': feat_imp})


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/learn')
def learn():
    return render_template('learn.html')


@app.route('/api/trigger', methods=['POST'])
def trigger_attack():
    """Force a specific attack type and return classification result."""
    data = request.get_json() or {}
    attack_type = data.get('attack_type', 'DoS')
    if attack_type not in ['Normal', 'DoS', 'PortScan', 'BruteForce']:
        return jsonify({'error': 'Invalid attack type'}), 400

    pkt, true_label, src_ip, dst_ip, port, proto = make_fake_packet(attack_type=attack_type)
    ai_label, ai_conf, ai_proba, rule_label = classify_packet(pkt)

    return jsonify({
        'timestamp': time.strftime('%H:%M:%S'),
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'port': port,
        'protocol': proto,
        'features': pkt,
        'true_label': true_label,
        'ai': {
            'label': ai_label,
            'confidence': ai_conf,
            'probabilities': ai_proba,
            'correct': ai_label == true_label
        },
        'rule': {
            'label': rule_label,
            'correct': rule_label == true_label
        }
    })


@app.route('/api/dataset-sample')
def dataset_sample():
    """Return EDA stats + sample rows for the about page."""
    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'dataset.csv'))

    stats = {}
    for label in df['label'].unique():
        sub = df[df['label'] == label]
        stats[label] = {
            'count': int(len(sub)),
            'avg_src_bytes': round(float(sub['src_bytes'].mean()), 1),
            'avg_count': round(float(sub['count'].mean()), 1),
            'avg_serror_rate': round(float(sub['serror_rate'].mean()), 3),
            'avg_rerror_rate': round(float(sub['rerror_rate'].mean()), 3),
            'avg_diff_srv_rate': round(float(sub['diff_srv_rate'].mean()), 3),
            'avg_failed_logins': round(float(sub['num_failed_logins'].mean()), 2),
        }

    # 5 sample rows per class — use concat to preserve label column
    sample = pd.concat([
        df[df['label']==lbl].sample(5, random_state=42)
        for lbl in df['label'].unique()
    ]).reset_index(drop=True)
    sample_rows = sample[['label','duration','src_bytes','dst_bytes','count',
                           'serror_rate','rerror_rate','diff_srv_rate',
                           'num_failed_logins','logged_in']].to_dict(orient='records')

    feat_desc = {
        'duration':          'Length of the connection in seconds',
        'src_bytes':         'Bytes sent from source to destination',
        'dst_bytes':         'Bytes sent from destination to source',
        'land':              '1 if src/dst host & port are equal',
        'wrong_fragment':    'Number of wrong fragments in packet',
        'urgent':            'Number of urgent packets',
        'hot':               'Number of hot indicators (privileged ops)',
        'num_failed_logins': 'Number of failed login attempts',
        'logged_in':         '1 if logged in successfully, 0 otherwise',
        'num_compromised':   'Number of compromised conditions',
        'count':             'Connections to same host in past 2 seconds',
        'srv_count':         'Connections to same service in past 2 seconds',
        'serror_rate':       'Rate of SYN errors (% of connections)',
        'rerror_rate':       'Rate of REJ errors (% of connections)',
        'same_srv_rate':     'Rate of connections to same service',
        'diff_srv_rate':     'Rate of connections to different services',
        'dst_host_count':    'Connections to same destination host',
        'dst_host_srv_count':'Connections to same destination service',
    }

    return jsonify({
        'total_rows': int(len(df)),
        'features': len(FEATURES),
        'class_stats': stats,
        'sample_rows': sample_rows,
        'feature_descriptions': feat_desc,
    })


if __name__ == '__main__':
    print("🚀 NIDS Dashboard → http://127.0.0.1:5000")
    app.run(debug=True, port=5000)