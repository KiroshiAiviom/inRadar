import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap
# Optional deps
try:
    import osmnx as ox
    import networkx as nx
except Exception:
    ox = None
    nx = None

def _choose_art_dir():
    for p in [Path("data/processed"), Path("data"), Path("artifacts")]:
        if p.exists():
            return p
    return Path("artifacts")

ART_DIR = _choose_art_dir()
CFG_PATH = ART_DIR / "config.json"
SRI_EDGE = ART_DIR / "sri_edge.parquet"
EVENTS_EDGE = ART_DIR / "events_edge.parquet"
MATCHED = ART_DIR / "matched_points.parquet"
EDGE_FEAT = ART_DIR / "features_edge.parquet"
SAFE_PU = ART_DIR / "safe_pickups.geojson"
GEOFENCES = ART_DIR / "safety_geofences.geojson"
GRAPH = ART_DIR / "osm_graph.graphml"

# -----------------------------
# Utils
# -----------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def load_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return None

# Cached loaders and builders (Streamlit)
@st.cache_data(show_spinner=False)
def read_parquet_cached(path: str, columns=None):
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        # Fallback if engine doesn't support 'columns'
        return pd.read_parquet(path)

@st.cache_data(show_spinner=False)
def read_json_cached(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=True)
def build_mmj_cached(matched_path: str, sri_path: str, events_edge_path: str|None):
    # Load minimal columns to reduce memory/time
    matched = read_parquet_cached(matched_path, columns=["u","v","key","lat","lng"])  # type: ignore[arg-type]
    sri = read_parquet_cached(sri_path, columns=["u","v","key","SRI"])  # type: ignore[arg-type]
    mmj = matched.merge(sri, on=["u","v","key"], how="inner")
    if events_edge_path and os.path.exists(events_edge_path):
        ev = read_parquet_cached(events_edge_path, columns=["u","v","key","n_obs","n_ids","WWO","ILS","UUT","HBR","HCT","GLD"])  # type: ignore[arg-type]
        mmj = mmj.merge(ev, on=["u","v","key"], how="left")
    else:
        mmj["n_obs"] = np.nan
        mmj["n_ids"] = np.nan
        for r in ["WWO","ILS","UUT","HBR","HCT","GLD"]:
            mmj[r] = np.nan
    return mmj

@st.cache_resource(show_spinner=False)
def get_graph_with_costs(graph_path: str, sri_edge_path: str, edge_feat_path: str):
    # Import inside to avoid import cost when routing is unused
    import osmnx as ox  # type: ignore
    import pandas as pd  # local alias
    G = ox.load_graphml(graph_path)
    # Load minimal columns
    sri_df = read_parquet_cached(sri_edge_path, columns=["u","v","key","SRI"])  # type: ignore[arg-type]
    try:
        ef = read_parquet_cached(edge_feat_path, columns=["u","v","key","p50_spd"])  # type: ignore[arg-type]
    except Exception:
        ef = read_parquet_cached(edge_feat_path)  # type: ignore[arg-type]
    # Build maps
    sri_map = {(r.u, r.v, r.key): float(r.SRI) for r in sri_df.itertuples()}
    # Pair-level fallback (ignore key)
    try:
        sri_pair = sri_df.groupby(["u", "v"], as_index=False)["SRI"].max()
        sri_pair_map = {(int(r.u), int(r.v)): float(r.SRI) for r in sri_pair.itertuples()}
    except Exception:
        sri_pair_map = {}

    spd_map = {}
    spd_pair_map = {}
    if "p50_spd" in ef.columns:
        for r in ef.itertuples():
            spd_map[(r.u, r.v, r.key)] = float(max(1.0, getattr(r, "p50_spd", 10.0)))
            spd_pair_map[(r.u, r.v)] = float(max(1.0, getattr(r, "p50_spd", 10.0)))

    # Precompute per-edge costs; track coverage
    assigned_sri = 0
    total_edges = 0
    for u, v, k, data in G.edges(keys=True, data=True):
        total_edges += 1
        length = float(data.get("length", 20.0))
        vexp = float(
            spd_map.get((u, v, k),
                        spd_map.get((v, u, k),
                                    spd_pair_map.get((u, v),
                                                     spd_pair_map.get((v, u), 10.0))))
        )
        sri = float(
            sri_map.get((u, v, k),
                        sri_map.get((v, u, k),
                                    sri_pair_map.get((u, v),
                                                     sri_pair_map.get((v, u), 0.0))))
        )
        if sri > 0:
            assigned_sri += 1
        data["eta_sec"] = length / max(1.0, vexp)
        # Store sri for routing rules; base risk_cost on eta seconds for comparable units
        data["sri"] = sri
        data["risk_cost"] = (sri / 100.0) * data["eta_sec"]
    G.graph["sri_assigned_pct"] = (100.0 * assigned_sri / max(1, total_edges))
    return G

# -----------------------------
# Load data
# -----------------------------
st.set_page_config(layout="wide", page_title="inRadar — Радар безопасности")
st.title("inRadar — Радар безопасности")
st.caption("Аналитика безопасности дорожной сети: горячие точки, безопасные зоны посадки и безопасная навигация.")

# Session state для маршрута и метрик
if "routes" not in st.session_state:
    st.session_state["routes"] = {"safe": None}

cfg = load_json(CFG_PATH, default={"K_ANON": 10, "SRI_WEIGHTS": {}})
w = cfg.get("SRI_WEIGHTS", {})
K_DEFAULT = int(cfg.get("K_ANON", 10))


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Слои и фильтры")
layer_type = st.sidebar.radio("Тепловая карта", ["SRI (индекс риска)", "Плотность (кол-во)"])
sri_thr = st.sidebar.slider("Порог SRI (горячо при ≥)", min_value=0, max_value=100, value=85, step=1)
k_anon = st.sidebar.slider("K-анонимность (мин. уникальных ID)", min_value=0, max_value=50, value=K_DEFAULT, step=1)
show_markers = st.sidebar.checkbox("Показывать маркеры риска (выборка)", value=True)
sample_markers = st.sidebar.slider("Макс. маркеров", 100, 2000, 1000, step=100)
max_heat_pts = st.sidebar.slider("Макс. точек теплокарты", 10000, 200000, 60000, step=10000)

st.sidebar.header("Безопасные точки посадки (покрытие)")
cov_radius = st.sidebar.slider("Радиус покрытия, м", 50, 200, 120, step=10)
recompute_pu = st.sidebar.checkbox("Пересчитать точки из SRI", value=True)
safe_sri_max = st.sidebar.slider("Макс. SRI для безопасной точки", 0, 100, 40, step=5)
num_safe_spots = st.sidebar.slider("Количество безопасных точек", 1, 50, 10, step=1)

st.sidebar.header("Безопасный маршрут (ETA ⊕ Риск)")
beta = st.sidebar.slider("Вес риска β", 0.0, 5.0, 0.6, step=0.1)
risk_scale = st.sidebar.slider("Масштаб риска γ", 0.1, 10.0, 2.0, step=0.1)
avoid_hot = st.sidebar.checkbox("Избегать горячих ребер", value=False)
avoid_thr = st.sidebar.slider("Порог SRI для избегания", 0, 100, sri_thr, step=1)
st.sidebar.caption("Стоимость = ETA + β × (γ × Risk). Опция избегания добавляет крупный штраф ребрам с SRI ≥ порога.")

# -----------------------------
# Precompute joins & filters
# -----------------------------
if not (SRI_EDGE.exists() and MATCHED.exists()):
    st.error("Не найдены необходимые артефакты (sri_edge.parquet и matched_points.parquet). Запустите S06 и S02.")
    st.stop()

# Build joined matched points with SRI and optional events (cached)
mmj = build_mmj_cached(str(MATCHED), str(SRI_EDGE), str(EVENTS_EDGE) if EVENTS_EDGE.exists() else None)

# Apply K-anonymity filter
if "n_ids" in mmj.columns:
    mmj = mmj[(mmj["n_ids"].fillna(9999) >= k_anon)]
# Apply SRI threshold for "hot" selection (used in coverage and markers)
hot = mmj[mmj["SRI"] >= sri_thr]

# Compute reason contributions per edge (weighted by w_i)
def reason_scores(row):
    n = row.get("n_obs", np.nan)
    if not np.isfinite(n) or n <= 0:
        return {}
    out = {}
    for r in ["WWO","ILS","UUT","HBR","HCT","GLD"]:
        k = row.get(r, 0.0)
        out[r] = w.get(r, 1.0) * (k / n)
    return out

# For markers we need top reasons
if "n_obs" in mmj.columns:
    # Reduce duplicate edges by sampling points
    mark_df = hot.sample(min(len(hot), sample_markers), random_state=42).copy()
    top_reasons = []
    for _, rw in mark_df.iterrows():
        sc = reason_scores(rw)
        if sc:
            top3 = sorted(sc.items(), key=lambda x: x[1], reverse=True)[:3]
            top_reasons.append(", ".join([f"{a}:{b:.2f}" for a,b in top3]))
        else:
            top_reasons.append("—")
    mark_df["reasons"] = top_reasons
else:
    mark_df = hot.sample(min(len(hot), sample_markers), random_state=42).copy()
    mark_df["reasons"] = "—"

# -----------------------------
# Coverage@R for safe pick-ups
# -----------------------------
coverage_pct = None
hot_pts = []
if GEOFENCES.exists():
    gj_hot = read_json_cached(str(GEOFENCES))
    hot_pts = [(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]) for f in gj_hot.get("features", [])]
else:
    # Fallback to current hot selection from data
    if len(hot) > 0:
        hot_pts = list(zip(hot["lat"].tolist(), hot["lng"].tolist()))

# Compute or load safe pick-ups
pu_dynamic = []  # list of dicts with lat,lng,safety
pu_pts = []      # list of (lat,lng) for coverage
if recompute_pu and hot_pts:
    # Candidates: low-SRI points from matched join
    cand = mmj[["lat","lng","SRI"]].dropna().copy()
    cand = cand[cand["SRI"] <= float(safe_sri_max)]
    cand = cand.reset_index(drop=True)
    # Build neighbors (hot points within radius) without sklearn
    if len(cand) > 0:
        EARTH_R = 6_371_000.0
        cand_xy = cand[["lat","lng"]].to_numpy()
        hot_xy = np.array(hot_pts, dtype=float)
        lat_tol = cov_radius / 111_000.0
        order = np.argsort(hot_xy[:, 0])
        hot_sorted = hot_xy[order]
        neighbors = []
        for lat, lng in cand_xy:
            lo, hi = lat - lat_tol, lat + lat_tol
            i0 = np.searchsorted(hot_sorted[:, 0], lo)
            i1 = np.searchsorted(hot_sorted[:, 0], hi)
            if i1 <= i0:
                neighbors.append(np.array([], dtype=int))
                continue
            clat, clng = np.radians([lat, lng])
            block = np.radians(hot_sorted[i0:i1])
            dlat = block[:, 0] - clat
            dlng = block[:, 1] - clng
            a = np.sin(dlat/2)**2 + np.cos(clat)*np.cos(block[:, 0])*np.sin(dlng/2)**2
            dist = 2*EARTH_R*np.arcsin(np.sqrt(a))
            idx = order[i0:i1][dist <= cov_radius]
            neighbors.append(idx)
        covered = np.zeros(len(hot_xy), dtype=bool)
        chosen = []
        picks = min(int(num_safe_spots), len(cand))
        for _ in range(picks):
            gains = np.fromiter(((~covered[idxs]).sum() for idxs in neighbors), dtype=int, count=len(neighbors))
            best = int(gains.argmax()) if len(gains) > 0 else -1
            if best < 0 or gains[best] <= 0:
                break
            row = cand.loc[best]
            safety = 100.0 - float(row["SRI"])
            pu_dynamic.append({"lat": float(row["lat"]), "lng": float(row["lng"]), "safety": max(0.0, min(100.0, safety))})
            covered[neighbors[best]] = True
            neighbors[best] = np.array([], dtype=int)
        pu_pts = [(d["lat"], d["lng"]) for d in pu_dynamic]
elif SAFE_PU.exists():
    gj_pu = read_json_cached(str(SAFE_PU))
    pu_pts = [(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]) for f in gj_pu.get("features", [])]

if hot_pts and pu_pts:
    covered = 0
    for lt, lg in hot_pts:
        dmin = min(haversine_m(lt, lg, p[0], p[1]) for p in pu_pts)
        if dmin <= cov_radius:
            covered += 1
    coverage_pct = 100.0 * covered / len(hot_pts)

# -----------------------------
# Map rendering
# -----------------------------
m = folium.Map(location=[mmj["lat"].median(), mmj["lng"].median()], zoom_start=13, control_scale=True)

# Heat Layer (sampled for performance)
mmj_heat = mmj
if len(mmj_heat) > max_heat_pts:
    mmj_heat = mmj_heat.sample(max_heat_pts, random_state=1)
if layer_type == "SRI (индекс риска)":
    heat = mmj_heat[["lat","lng","SRI"]].values.tolist()
    HeatMap(heat, radius=7, blur=9, min_opacity=0.35, max_zoom=18).add_to(m)
else:
    heat = mmj_heat[["lat","lng"]].values.tolist()
    HeatMap(heat, radius=7, blur=9, min_opacity=0.35, max_zoom=18).add_to(m)

# Risk markers (sampled)
if show_markers and len(mark_df) > 0:
    fg = folium.FeatureGroup(name="Маркеры риска (выборка)")
    for _, r in mark_df.iterrows():
        tooltip = f"SRI: {float(r['SRI']):.1f} | наблюдений: {int(r.get('n_obs',0))} | пользователей: {int(r.get('n_ids',0))}"
        popup = folium.Popup(f"<b>Причины:</b> {r['reasons']}", max_width=280)
        folium.CircleMarker(location=[r["lat"], r["lng"]], radius=3, tooltip=tooltip, popup=popup,
                            color=None, fill=True, fill_opacity=0.9).add_to(fg)
    fg.add_to(m)

# Safe pick-ups & geofences
if recompute_pu and pu_dynamic:
    fg2 = folium.FeatureGroup(name="Безопасные посадки (пересчет)")
    for d in pu_dynamic:
        lat, lng = d["lat"], d["lng"]
        safe_val = float(d.get("safety", 0.0))
        folium.Marker(
            location=[lat, lng],
            tooltip=f"Оценка безопасности: {safe_val:.1f}",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(fg2)
    fg2.add_to(m)
elif SAFE_PU.exists():
    gj = read_json_cached(str(SAFE_PU))
    fg2 = folium.FeatureGroup(name="Безопасные посадки")
    for f in gj.get("features", []):
        lng, lat = f["geometry"]["coordinates"]
        props = f.get("properties", {})
        # Prefer explicit 'safety' if present, otherwise derive from 'score'
        if "safety" in props:
            safe_val = float(props.get("safety", 0.0))
        else:
            raw_score = float(props.get("score", 0.0))
            # Backward-compat: earlier S07 exported score = -0.8 * SRI (<= 0)
            # Convert to intuitive safety in [0,100]: safety = 100 - SRI = 100 + score/0.8
            if raw_score <= 0:
                safe_val = 100.0 + (raw_score / 0.8)
            else:
                # If already positive, treat as safety directly
                safe_val = raw_score
        # Clamp to [0, 100] for display
        safe_val = max(0.0, min(100.0, safe_val))
        folium.Marker(
            location=[lat, lng],
            tooltip=f"Оценка безопасности: {safe_val:.1f}",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(fg2)
    fg2.add_to(m)

if GEOFENCES.exists():
    gj = read_json_cached(str(GEOFENCES))
    fg3 = folium.FeatureGroup(name="Горячие точки")
    for f in gj.get("features", [])[:3000]:
        lng, lat = f["geometry"]["coordinates"]
        val = f["properties"].get("SRI", 0.0)
        folium.CircleMarker(location=[lat,lng], radius=2, tooltip=f"SRI: {val:.1f}", color="red",
                            fill=True, fill_opacity=0.4).add_to(fg3)
    fg3.add_to(m)

st.subheader("Карта")
# If we have previously built routes, overlay them on the map (toggleable)
routes_state = st.session_state.get("routes", {"safe": None})
if routes_state.get("safe") and isinstance(routes_state["safe"].get("coords", None), list):
    routes_fg = folium.FeatureGroup(name="Маршрут (безопасный)", overlay=True, show=True)
    folium.PolyLine(routes_state["safe"]["coords"], tooltip="безопасный", weight=6, color="green").add_to(routes_fg)
    routes_fg.add_to(m)
    # Маркеры начала/конца
    if "origin" in routes_state and "dest" in routes_state:
        try:
            folium.Marker(routes_state["origin"], tooltip="Старт", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
            folium.Marker(routes_state["dest"], tooltip="Финиш", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        except Exception:
            pass

# Ensure layer control reflects all layers, including routes
folium.LayerControl(collapsed=False).add_to(m)
st.components.v1.html(m._repr_html_(), height=680, scrolling=False)

# -----------------------------
# Metrics section
# -----------------------------
colA, colB, colC = st.columns(3)
colA.metric("Точек на карте", f"{len(mmj):,}")
if coverage_pct is not None:
    colB.metric(f"Покрытие@{cov_radius}м", f"{coverage_pct:.1f}%")
colC.metric("Порог горячих (SRI)", f"{sri_thr}")

st.caption("Применены фильтры: K-анонимность ≥ %d, SRI ≥ %d (для hot/markers)." % (k_anon, sri_thr))

# Показатели маршрута (если построен)
routes_state = st.session_state.get("routes", {"safe": None})
if routes_state.get("safe"):
    s = routes_state["safe"]
    c1, c2, c3 = st.columns(3)
    c1.metric("ETA безопасного", f"{s['eta']/60.0:.1f} мин")
    c2.metric("Длина", f"{s['len']/1000.0:.2f} км")
    if isinstance(s.get("hot_share"), (int, float)):
        c3.metric("Доля горячих ребер", f"{s['hot_share']:.1f}%")

# Подсказки / легенда
with st.expander("Легенда и подсказки"):
    st.markdown(
        "- Зеленая линия: построенный безопасный маршрут.\n"
        "- Красные точки: горячие места по порогу SRI.\n"
        "- Маркеры риска: точки с высоким вкладом детекторов.\n"
        "- Безопасные посадки: рекомендуемые точки с низким SRI; показатель — чем выше, тем безопаснее.\n"
        "- Настройте вес риска β и масштаб γ, чтобы сильнее учитывать риск; опция ‘Избегать горячих ребер’ принудительно штрафует опасные сегменты."
    )

# -----------------------------
# Routing demo (ETA ⊕ Risk)
# -----------------------------
rheader = "Безопасная навигация — безопасный маршрут (ETA ⊕ β·Риск)"
st.header(rheader)
routing_ready = (ox is not None) and GRAPH.exists() and EDGE_FEAT.exists() and SRI_EDGE.exists()
if not routing_ready:
    st.info("Для построения маршрута нужен OSM-граф и фичи: запустите S02 (граф), S03 (edge features), S06 (SRI).")
else:
    st.markdown("Введите координаты в формате **Широта, Долгота** (или используйте значения по умолчанию).")
    c1, c2 = st.columns(2)
    default_lat = float(mmj["lat"].median())
    default_lng = float(mmj["lng"].median())
    with c1:
        lat_o = st.number_input("Широта отправления", value=default_lat, format="%.6f")
        lng_o = st.number_input("Долгота отправления", value=default_lng-0.01, format="%.6f")
    with c2:
        lat_d = st.number_input("Широта назначения", value=default_lat, format="%.6f")
        lng_d = st.number_input("Долгота назначения", value=default_lng+0.01, format="%.6f")

    build_col, clear_col = st.columns([1,1])
    build_clicked = build_col.button("Построить безопасный маршрут")
    clear_clicked = clear_col.button("Очистить маршрут")
    if clear_clicked:
        st.session_state["routes"] = {"safe": None}
        st.rerun()

    if build_clicked:
        with st.spinner("Строим маршрут…"):
            # Load graph and edge costs once per session
            G = get_graph_with_costs(str(GRAPH), str(SRI_EDGE), str(EDGE_FEAT))
            cov = float(G.graph.get("sri_assigned_pct", 0.0))
            if cov < 5.0:
                st.warning(f"Низкое покрытие SRI в графе: {cov:.1f}% ребер. Безопасный маршрут может совпадать с обычным.")
        try:
            o_node = ox.distance.nearest_nodes(G, X=[lng_o], Y=[lat_o])[0]
            d_node = ox.distance.nearest_nodes(G, X=[lng_d], Y=[lat_d])[0]
        except Exception as e:
            st.error(f"Не удалось найти ближайшие узлы: {e}")
            st.stop()

        # NetworkX weight function must accept (u, v, data)
        def path_cost(beta_val, gamma_val, avoid_flag, avoid_threshold):
            def w(u, v, data):
                base = data.get("eta_sec", 2.0) + beta_val * gamma_val * data.get("risk_cost", 0.0)
                if avoid_flag and data.get("sri", 0.0) >= float(avoid_threshold):
                    return base + 300.0  # large penalty in seconds
                return base
            return w

        # Safe route (beta>0)
        try:
            path_safe = nx.shortest_path(G, o_node, d_node, weight=path_cost(beta, risk_scale, avoid_hot, avoid_thr))
        except Exception as e:
            st.error(f"Ошибка поиска безопасного маршрута: {e}")
            path_safe = None

        def summarize_path(path, beta_val, gamma_val, avoid_flag, avoid_threshold):
            if path is None:
                return None
            eta, rcost, length = 0.0, 0.0, 0.0
            coords = []
            hot_edges = 0
            total_edges = 0
            for u,v in zip(path[:-1], path[1:]):
                # choose the edge key that matches the route's cost (min eta + beta*risk)
                best_k, best_c = None, float("inf")
                for k, data_k in G[u][v].items():
                    c = data_k.get("eta_sec", 0.0) + beta_val * gamma_val * data_k.get("risk_cost", 0.0)
                    if avoid_flag and data_k.get("sri", 0.0) >= float(avoid_threshold):
                        c += 300.0
                    if c < best_c:
                        best_c = c
                        best_k = k
                data = G[u][v][best_k]
                eta += data.get("eta_sec", 0.0)
                rcost += data.get("risk_cost", 0.0) * gamma_val
                length += data.get("length", 0.0)
                total_edges += 1
                if data.get("sri", 0.0) >= float(avoid_threshold):
                    hot_edges += 1
                if "geometry" in data:
                    xs, ys = data["geometry"].xy
                    coords += list(zip(ys, xs))
                else:
                    coords += [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
            hot_share = (100.0 * hot_edges / max(1, total_edges))
            return {"eta": eta, "risk": rcost, "len": length, "coords": coords, "hot_share": hot_share}

        s_safe = summarize_path(path_safe, beta, risk_scale, avoid_hot, avoid_thr)

        # Save route to session state and rerun to draw on the main map above
        st.session_state["routes"] = {"safe": s_safe, "origin": (float(lat_o), float(lng_o)), "dest": (float(lat_d), float(lng_d))}
        try:
            st.rerun()
        except Exception:
            pass
