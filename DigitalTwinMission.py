import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. MODEL MATEMATYCZNY I FIZYCZNY (DIGITAL TWIN CORE)
# ==========================================

# Stałe planetarne
G = 6.67430e-11  # Stała grawitacyjna
M_EARTH = 5.972e24  # Masa Ziemi (kg)
R_EARTH = 6371000  # Promień Ziemi (m)


def get_atmosphere_density(altitude):
    """Model barometryczny atmosfery Ziemi."""
    if altitude > 80000:  # Powyżej 80km brak atmosfery
        return 0.0
    rho0 = 1.225  # kg/m^3 przy powierzchni
    H = 8500  # Wysokość skali (m)
    return rho0 * np.exp(-altitude / H)


def rocket_equations(t, y, params):
    """Równania ruchu rakiety 2D."""
    x, y_pos, vx, vy, mf1, mf2 = y

    r = np.sqrt(x ** 2 + y_pos ** 2)
    altitude = r - R_EARTH

    if altitude < 0:
        # Jeśli uderzy w ziemię, zatrzymujemy ruch
        return [0, 0, 0, 0, 0, 0]

    # Siła grawitacji
    g_force = G * M_EARTH / r ** 2
    ax_g = -g_force * (x / r)
    ay_g = -g_force * (y_pos / r)

    # Profil pochylenia (Gravity Turn)
    pitch_angle = 0.0
    if t > params['pitch_start_time']:
        fraction = (t - params['pitch_start_time']) / params['pitch_duration']
        pitch_angle = min(fraction * (np.pi / 2 - params['target_pitch_end']), np.pi / 2 - params['target_pitch_end'])

    local_vertical = np.arctan2(y_pos, x)
    thrust_angle = local_vertical - pitch_angle

    thrust = 0.0
    dm1 = 0.0
    dm2 = 0.0
    stage = 1

    # Wyznaczenie czasów wypalenia na podstawie masy paliwa, ciągu i Isp
    s1_mdot = params['s1_thrust'] / (params['s1_isp'] * 9.81)
    s2_mdot = params['s2_thrust'] / (params['s2_isp'] * 9.81)

    s1_burn_time = params['s1_fuel_mass'] / s1_mdot

    if mf1 > 0:
        thrust = params['s1_thrust']
        dm1 = -s1_mdot
        m_total = params['payload_mass'] + params['s2_dry_mass'] + mf2 + params['s1_dry_mass'] + mf1
        stage = 1
    elif t < s1_burn_time + params['separation_delay']:
        thrust = 0.0
        m_total = params['payload_mass'] + params['s2_dry_mass'] + mf2
        stage = 1.5
    elif mf2 > 0:
        thrust = params['s2_thrust']
        dm2 = -s2_mdot
        m_total = params['payload_mass'] + params['s2_dry_mass'] + mf2
        stage = 2
    else:
        thrust = 0.0
        m_total = params['payload_mass'] + params['s2_dry_mass']
        stage = 3

    m_total += params['rocket_mass']

    # Aerodynamika
    rho = 0
    v_speed = np.sqrt(vx ** 2 + vy ** 2)
    if altitude < 100000:
        rho = get_atmosphere_density(altitude)

    cd = params['cd_s1'] if stage <= 1.5 else params['cd_s2']
    drag = 0.5 * rho * v_speed ** 2 * cd * params['cross_section_area']

    if v_speed > 0:
        ax_d = -drag * (vx / v_speed) / m_total
        ay_d = -drag * (vy / v_speed) / m_total
    else:
        ax_d, ay_d = 0, 0

    ax_t = thrust * np.cos(thrust_angle) / m_total
    ay_t = thrust * np.sin(thrust_angle) / m_total

    d_x = vx
    d_y = vy
    d_vx = ax_g + ax_d + ax_t
    d_vy = ay_g + ay_d + ay_t

    return [d_x, d_y, d_vx, d_vy, dm1, dm2]


def run_simulation(params):
    """Uruchamia integrację numeryczną lotu."""

    def hit_ground_event(t, y, params):
        """Zwraca odległość od powierzchni Ziemi. Gdy osiągnie 0, symulacja się kończy."""
        x, y_pos, _, _, _, _ = y
        r = np.sqrt(x ** 2 + y_pos ** 2)
        return r - R_EARTH

    hit_ground_event.terminal = True
    hit_ground_event.direction = -1

    y0 = [0.0, R_EARTH, 0.0, 0.0, params['s1_fuel_mass'], params['s2_fuel_mass']]
    t_span = (0, params['max_sim_time'])

    sol = solve_ivp(rocket_equations, t_span, y0, args=(params,), events=hit_ground_event, method='RK45', max_step=1.0)

    df = pd.DataFrame({
        'time': sol.t,
        'x': sol.y[0],
        'y': sol.y[1],
        'vx': sol.y[2],
        'vy': sol.y[3],
        'mf1': sol.y[4],
        'mf2': sol.y[5]
    })

    df['altitude'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2) - R_EARTH
    df['velocity'] = np.sqrt(df['vx'] ** 2 + df['vy'] ** 2)

    # Obliczanie Stabilności Lotu (Angle of Attack - Kąt Natarcia)
    df['v_angle'] = np.arctan2(df['vy'], df['vx'])
    rocket_angles = []
    for t in df['time']:
        p_ang = 0.0
        if t > params['pitch_start_time']:
            frac = (t - params['pitch_start_time']) / params['pitch_duration']
            p_ang = min(frac * (np.pi / 2 - params['target_pitch_end']), np.pi / 2 - params['target_pitch_end'])
        local_vert = np.pi / 2  # Start z bieguna pionowo w górę
        rocket_angles.append(local_vert - p_ang)

    df['rocket_angle'] = rocket_angles
    df['aoa'] = np.degrees(np.abs(df['rocket_angle'] - df['v_angle']))
    # Powyżej gęstej atmosfery brak oporu aerodynamicznego znosi pojęcie niestabilności AoA
    df.loc[df['altitude'] > 65000, 'aoa'] = 0.0

    s1_mdot = params['s1_thrust'] / (params['s1_isp'] * 9.81)
    s1_burn_time = params['s1_fuel_mass'] / s1_mdot

    stages = []
    for idx, r in df.iterrows():
        if r['mf1'] > 0:
            stages.append('1. Stopień')
        elif r['time'] < s1_burn_time + params['separation_delay']:
            stages.append('Separacja Stopni')
        elif r['mf2'] > 0:
            stages.append('2. Stopień')
        else:
            stages.append('Orbita / Dryf')
    df['stage'] = stages

    df['v_orbital_required'] = np.sqrt(G * M_EARTH / (R_EARTH + df['altitude']))

    orbit_achieved = False
    orbit_zone = df[df['stage'] == 'Orbita / Dryf']
    if not orbit_zone.empty:
        max_alt = orbit_zone['altitude'].max()
        max_v = orbit_zone['velocity'].max()
        req_v = orbit_zone['v_orbital_required'].iloc[-1]
        if max_alt >= 140000 and max_v >= req_v * 0.92:
            orbit_achieved = True

    return df, orbit_achieved


# ==========================================
# 2. OPTYMALIZACJA TRAJEKTORII (AI / HEURYSTYKA)
# ==========================================

def optimize_pitch(params):
    """Szuka najlepszego czasu rozpoczęcia manewru oraz profilu pochylenia."""

    def optimize_pitch(params):
        """Szuka parametrów Gravity Turn (czasu i kąta), które pozwolą okrążyć Ziemię w jak największym stopniu."""

        def objective(x):
            test_params = params.copy()
            # Wyciągamy oba parametry z wektora x
            test_params['pitch_start_time'] = float(x[0])
            test_params['target_pitch_end'] = float(x[1])

            try:
                df, success = run_simulation(test_params)

                if df is None or len(df) < 2:
                    return 1e12

                # 1. Kąt pokonany wokół Ziemi
                angles = np.unwrap(np.arctan2(df['y'], df['x']))
                total_travelled_angle = np.abs(np.degrees(angles[-1] - angles[0]))

                # 2. Dodatkowa nagroda za prędkość składowej poziomej (klucz do okrążenia Ziemi)
                # Im mniejsze pochylenie końcowe (bliżej 0 rad / poziomu), tym większa prędkość orbitalna
                max_velocity = df['velocity'].max()

                # Łączymy pokonany kąt z uzyskaną prędkością, by algorytm widział sens w pochylaniu rakiety
                score = total_travelled_angle * 10 + (max_velocity / 100)

                # Kary za nieudany lot
                if df['altitude'].max() < 100000:
                    score -= 1000

                if success:
                    score += 50000  # Ogromny bonus za pełną orbitę

                return -float(score)

            except Exception as e:
                return 1e12

        try:
            # Zmiana metody na 'Powell' – znacznie lepiej radzi sobie z optymalizacją kątów i czasu jednocześnie
            # Ustawiamy punkt startowy na obecne wartości z suwaków, żeby AI szukało ulepszenia od Twoich ustawień
            x0 = [float(params['pitch_start_time']), float(params['target_pitch_end'])]

            res = minimize(
                objective,
                x0=x0,
                bounds=[(4.0, 50.0), (-0.2, 0.8)],
                method='Powell',
                options={'xtol': 1e-3, 'ftol': 1e-3}
            )

            if res.x is not None and len(res.x) >= 2:
                return float(res.x[0]), float(res.x[1])

        except Exception as e:
            print(f"Błąd optymalizatora: {e}")

        return float(params['pitch_start_time']), float(params['target_pitch_end'])


# ==========================================
# 3. INTERFEJS UŻYTKOWNIKA (DASHBOARD)
# ==========================================

default_params = {
    'rocket_mass': 30,
    'payload_mass': 0,
    's1_dry_mass': 30,
    's1_fuel_mass': 500,
    's1_thrust': 30000,
    's1_isp': 400,
    's2_dry_mass': 30,
    's2_fuel_mass': 185,
    's2_thrust': 10000,
    's2_isp': 400,
    'cross_section_area': 0.05,
    'cd_s1': 0.30,
    'cd_s2': 0.15,
    'pitch_start_time': 32.0,
    'pitch_duration': 7.0,
    'target_pitch_end': -0.064,
    'separation_delay': 4.0,
    'max_sim_time': 30000
}

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .rc-slider-mark-text, .dash-slider-mark {
                color: #aaaaaa !important; 
                font-weight: 500;
            }

            .rc-slider-mark-text-active, .dash-slider-mark-outside-selection {
                color: #444444 !important; 
            }

            .dash-input-container {
                min-width: 70px;
                color: black;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div(
    style={'backgroundColor': '#1e2130', 'color': '#e1e1e1', 'padding': '20px', 'fontFamily': 'Arial'}, children=[

        html.Div([
            html.H1("SATELLITE LAUNCH SYSTEM | DIGITAL TWIN 3D",
                    style={'textAlign': 'center', 'color': '#4A90E2', 'fontWeight': 'bold'}),
            html.H5("Cyfrowy Bliźniak Systemu Nośnego i Telemetrii Misji z Analizą Stabilności",
                    style={'textAlign': 'center', 'color': '#8B9BB4'})
        ], style={'borderBottom': '2px solid #2b2e4a', 'paddingBottom': '10px'}),

        html.Div(className='row', children=[

            # Panel Lewy - Sterowanie Twin Control
            html.Div(className='four columns',
                     style={'backgroundColor': '#24273c', 'padding': '20px', 'borderRadius': '10px',
                            'marginTop': '20px'}, children=[
                    html.H3("⚙️ Twin Parameters", style={'color': '#00FFFF', 'fontSize': '18px', 'marginTop': '0'}),

                    html.Label("Masa rakiety [kg]:"),
                    dcc.Slider(id='slider-rocket', min=0, max=200, step=1, value=default_params['rocket_mass'],
                               marks={i: f"{i}" for i in range(0, 201, 50)}),

                    html.Label("Masa ładunku [kg]:"),
                    dcc.Slider(id='slider-payload', min=0, max=50, step=1, value=default_params['payload_mass'],
                               marks={i: f"{i}" for i in range(0, 51, 15)}),

                    html.Label("Paliwo 1. Stopnia [kg]:", style={'marginTop': '15px'}),
                    dcc.Slider(id='slider-s1-fuel', min=0, max=1000, step=5, value=default_params['s1_fuel_mass'],
                               marks={i: f"{i}" for i in range(0, 1001, 200)}),

                    html.Label("Ciąg 1. Stopnia [kN]:", style={'marginTop': '15px'}),
                    dcc.Slider(id='slider-s1-thrust', min=500, max=50000, step=100, value=default_params['s1_thrust'],
                               marks={i: f"{i // 1000}" for i in range(500, 50001, 10000)}),

                    html.Label("Paliwo 2. Stopnia [kg]:", style={'marginTop': '15px'}),
                    dcc.Slider(id='slider-s2-fuel', min=0, max=1000, step=5, value=default_params['s2_fuel_mass'],
                               marks={i: f"{i}" for i in range(0, 1001, 200)}),

                    html.Label("Ciąg 2. Stopnia [kN]:", style={'marginTop': '15px'}),
                    dcc.Slider(id='slider-s2-thrust', min=500, max=50000, step=100, value=default_params['s2_thrust'],
                               marks={i: f"{i // 1000}" for i in range(500, 50001, 10000)}),

                    html.Label("Czas na separację [s]:", style={'marginTop': '15px'}),
                    dcc.Slider(id='slider-pitch-duration', min=0, max=30, step=1,
                               value=default_params['pitch_duration'],
                               marks={i: f"{i}" for i in range(0, 31, 6)}),

                    html.Label("Start manewru Gravity Turn [s]:", style={'marginTop': '15px'}),
                    dcc.Slider(id='slider-pitch-start', min=0, max=50, step=1, value=default_params['pitch_start_time'],
                               marks={i: f"{i}" for i in range(0, 51, 10)}),

                    html.Label("Końcowe pochylenie [rad]:", style={'marginTop': '15px'}),
                    dcc.Slider(id='slider-target-pitch', min=-1, max=1, step=0.001,
                               value=default_params['target_pitch_end'],
                               marks={i: f"{i / 2}" for i in range(-2, 2, 1)}),

                    html.Div(style={'marginTop': '30px', 'textAlign': 'center'}, children=[
                        html.Button('🚀 URUCHOM SYMULACJĘ', id='btn-simulate', n_clicks=0,
                                    style={'backgroundColor': '#00C851', 'color': 'white', 'fontWeight': 'bold',
                                           'fontSize': '20px',
                                           'width': '100%', 'border': 'none', 'padding': '12px', 'borderRadius': '5px',
                                           'cursor': 'pointer', 'paddingBottom': '47px'}),

                        html.Button('🤖 OPTYMALIZUJ GRAVITY TURN', id='btn-optimize', n_clicks=0,
                                    style={'backgroundColor': '#aa66cc', 'color': 'white', 'fontWeight': 'bold',
                                           'fontSize': '20px',
                                           'width': '100%', 'border': 'none', 'padding': '12px', 'borderRadius': '5px',
                                           'marginTop': '10px', 'cursor': 'pointer', 'paddingBottom': '47px'})
                    ]),

                    html.Div(id='orbit-status-box')
                ]),

            # Panel Prawy - Wykresy i Telemetria
            html.Div(className='eight columns', children=[

                html.Div(className='row', style={'marginTop': '20px'}, children=[
                    html.Div(className='four columns',
                             style={'backgroundColor': '#24273c', 'padding': '15px', 'borderRadius': '8px',
                                    'textAlign': 'center'}, children=[
                            html.H6("Apogeum (Max Wysokość)", style={'color': '#8B9BB4', 'margin': '0'}),
                            html.H3(id='txt-max-alt',
                                    style={'color': '#00E5FF', 'margin': '5px 0 0 0', 'fontWeight': 'bold'})
                        ]),
                    html.Div(className='four columns',
                             style={'backgroundColor': '#24273c', 'padding': '15px', 'borderRadius': '8px',
                                    'textAlign': 'center'}, children=[
                            html.H6("Prędkość Maksymalna", style={'color': '#8B9BB4', 'margin': '0'}),
                            html.H3(id='txt-max-vel',
                                    style={'color': '#00E5FF', 'margin': '5px 0 0 0', 'fontWeight': 'bold'})
                        ]),
                    html.Div(className='four columns',
                             style={'backgroundColor': '#24273c', 'padding': '15px', 'borderRadius': '8px',
                                    'textAlign': 'center'}, children=[
                            html.H6("Maks. Niestabilność (AoA)", style={'color': '#8B9BB4', 'margin': '0'}),
                            html.H3(id='txt-max-aoa',
                                    style={'color': '#ffbb33', 'margin': '5px 0 0 0', 'fontWeight': 'bold'})
                        ]),
                ]),

                # Główny Panel Graficzny 3D oraz Profil 2D
                html.Div(style={'marginTop': '20px'}, children=[
                    dcc.Graph(id='graph-3d-orbit'),
                    dcc.Graph(id='graph-trajectory'),
                ]),

                # Segment Rozbitych Wykresów Telemetrii (Kafle 2x2)
                html.Div(className='row', style={'marginTop': '20px'}, children=[
                    html.Div(className='six columns', children=[dcc.Graph(id='graph-altitude')]),
                    html.Div(className='six columns', children=[dcc.Graph(id='graph-velocity')])
                ]),
                html.Div(className='row', style={'marginTop': '20px'}, children=[
                    html.Div(className='six columns', children=[dcc.Graph(id='graph-fuel')]),
                    html.Div(className='six columns', children=[dcc.Graph(id='graph-stability')])
                ])
            ])
        ])
    ])


# ==========================================
# 4. KONTROLER (CALLBACKS)
# ==========================================

@app.callback(
    [Output('graph-3d-orbit', 'figure'),
     Output('graph-trajectory', 'figure'),
     Output('graph-altitude', 'figure'),
     Output('graph-velocity', 'figure'),
     Output('graph-fuel', 'figure'),
     Output('graph-stability', 'figure'),
     Output('txt-max-alt', 'children'),
     Output('txt-max-vel', 'children'),
     Output('txt-max-aoa', 'children'),
     Output('orbit-status-box', 'children'),
     Output('slider-pitch-start', 'value'),
     Output('slider-target-pitch', 'value')],
    [Input('btn-simulate', 'n_clicks'),
     Input('btn-optimize', 'n_clicks')],
    [State('slider-rocket', 'value'),
     State('slider-payload', 'value'),
     State('slider-s1-fuel', 'value'),
     State('slider-s1-thrust', 'value'),
     State('slider-s2-fuel', 'value'),
     State('slider-s2-thrust', 'value'),
     State('slider-pitch-duration', 'value'),
     State('slider-pitch-start', 'value'),
     State('slider-target-pitch', 'value')]
)
def update_mission(sim_clicks, opt_clicks, rocket_mass, payload, s1_fuel, s1_thrust, s2_fuel, s2_thrust, pitch_duration,
                   pitch_start, target_pitch):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'btn-simulate'

    # Tworzenie dynamicznego zestawu parametrów na bazie suwaków
    current_params = default_params.copy()
    current_params['rocket_mass'] = float(rocket_mass)
    current_params['payload_mass'] = float(payload)
    current_params['s1_fuel_mass'] = float(s1_fuel)
    current_params['s1_thrust'] = float(s1_thrust)
    current_params['s2_fuel_mass'] = float(s2_fuel)
    current_params['s2_thrust'] = float(s2_thrust)
    current_params['pitch_duration'] = float(pitch_duration)
    current_params['pitch_start_time'] = float(pitch_start)
    current_params['target_pitch_end'] = float(target_pitch)

    # Jeśli wybrano optymalizację przez AI
    if triggered_id == 'btn-optimize':
        try:
            result = optimize_pitch(current_params)
            if result is not None and isinstance(result, (tuple, list)) and len(result) == 2:
                optimized_time, optimized_angle = result
            else:
                # Jeśli funkcja jakimś cudem zwróciła None lub coś innego
                optimized_time = float(pitch_start)
                optimized_angle = float(target_pitch)
        except Exception as e:
            print(f"Awaryjne przechwycenie błędu rozpakowania: {e}")
            optimized_time = float(pitch_start)
            optimized_angle = float(target_pitch)

        current_params['pitch_start_time'] = optimized_time
        current_params['target_pitch_end'] = optimized_angle
        pitch_start = round(optimized_time, 1)
        target_pitch = round(optimized_angle, 3)

    # Wywołanie bezpiecznego silnika fizycznego
    df, success = run_simulation(current_params)

    # ----------------------------------------------------
    # GENEROWANIE WIZUALIZACJI TRÓJWYMIAROWEJ (3D GRAPH)
    # ----------------------------------------------------
    fig_3d = go.Figure()

    # Tworzenie kuli ziemskiej w przestrzeni trójwymiarowej
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_earth = R_EARTH * np.outer(np.cos(u), np.sin(v)) / 1000
    y_earth = R_EARTH * np.outer(np.sin(u), np.sin(v)) / 1000
    z_earth = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v)) / 1000

    fig_3d.add_trace(
        go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', opacity=0.15, showscale=False, name='Ziemia'))

    # Mapowanie płaskiej trajektorii orbitalnej (x,y) na osie przestrzenne 3D (X, Y=0, Z)
    for stage_name, color in zip(['1. Stopień', 'Separacja Stopni', '2. Stopień', 'Orbita / Dryf'],
                                 ['#ef553b', '#636efa', '#00cc96', '#ab63fa']):
        stage_df = df[df['stage'] == stage_name]
        if not stage_df.empty:
            fig_3d.add_trace(go.Scatter3d(
                x=stage_df['x'] / 1000, y=np.zeros_like(stage_df['x']), z=stage_df['y'] / 1000,
                mode='lines', line=dict(color=color, width=5), name=stage_name
            ))

    fig_3d.update_layout(
        title='Cyfrowy Bliźniak: Przestrzenna Wizualizacja Trajektorii Orbitalnej 3D',
        template='plotly_dark', paper_bgcolor='#24273c',
        scene=dict(
            xaxis_title='Oś X (km)', yaxis_title='Oś Y (km)', zaxis_title='Wysokość osiowa Z (km)',
            aspectmode='data'
        )
    )

    df['x_km'] = df['x'] / 1000
    df['y_km'] = df['y'] / 1000
    R_EARTH_KM = R_EARTH / 1000

    # Wykres 1: Profil Trajektorii we współrzędnych orbitalnych 2D
    fig_traj = px.line(df, x='x_km', y='y_km', color='stage',
                       title='Profil Trajektorii Lotu Rakiety (Współrzędne Środka Ziemi)',
                       labels={'x': 'Dystans orbitalny X (km)', 'y_km': 'Dystans orbitalny Y (km)', 'stage': 'Faza Lotu'},
                       color_discrete_map={'1. Stopień': '#ef553b', 'Separacja Stopni': '#636efa',
                                           '2. Stopień': '#00cc96', 'Orbita / Dryf': '#ab63fa'})

    # Dodanie wizualnej powłoki Ziemi
    earth_angles = np.linspace(-np.pi, np.pi, 100)
    fig_traj.add_trace(go.Scatter(x=R_EARTH * np.cos(earth_angles) / 1000, y=R_EARTH * np.sin(earth_angles) / 1000,
                                  mode='lines', name='Krzywizna Ziemi', line=dict(dash='dash', color='#4A90E2')))

    fig_traj.update_layout(template='plotly_dark', paper_bgcolor='#24273c', plot_bgcolor='#24273c')

    # ----------------------------------------------------
    # SEKCJA NOWYCH ROZBITYCH WYKRESÓW TELEMETRYCZNYCH
    # ----------------------------------------------------

    # Wykres A: Profil Wysokości
    df['altitude_km'] = df['altitude'] / 1000
    fig_alt = px.line(df, x='time', y='altitude_km', title='Profil Wysokości w funkcji czasu',
                      labels={'altitude_km': 'Wysokość (km)', 'time': 'Czas (s)'})
    fig_alt.update_traces(line_color='#00E5FF').update_layout(template='plotly_dark', paper_bgcolor='#24273c',
                                                              plot_bgcolor='#24273c')

    # Wykres B: Profil Prędkości
    fig_vel = px.line(df, x='time', y='velocity', title='Profil Prędkości w funkcji czasu',
                      labels={'velocity': 'Prędkość (m/s)', 'time': 'Czas (s)'})
    fig_vel.update_traces(line_color='#00C851').update_layout(template='plotly_dark', paper_bgcolor='#24273c',
                                                              plot_bgcolor='#24273c')

    # Wykres C: Zużycie paliwa (oba zbiorniki na jednym wykresie)
    fig_fuel = go.Figure()
    fig_fuel.add_trace(go.Scatter(x=df['time'], y=df['mf1'], mode='lines', name='Zbiornik Stopnia 1',
                                  line=dict(color='#ef553b', width=2.5)))
    fig_fuel.add_trace(go.Scatter(x=df['time'], y=df['mf2'], mode='lines', name='Zbiornik Stopnia 2',
                                  line=dict(color='#00cc96', width=2.5)))
    fig_fuel.update_layout(title='Zużycie Paliwa w czasie lotu', xaxis_title='Czas (s)', yaxis_title='Masa paliwa (kg)',
                           template='plotly_dark', paper_bgcolor='#24273c', plot_bgcolor='#24273c')

    # Wykres D: Stabilność lotu (AoA - Angle of Attack)
    fig_stab = px.line(df, x='time', y='aoa', title='Stabilność Aerodynamiczna (Kąt Natarcia AoA)',
                       labels={'aoa': 'Odchylenie AoA (stopnie)', 'time': 'Czas (s)'})
    fig_stab.update_traces(line_color='#ffbb33').update_layout(template='plotly_dark', paper_bgcolor='#24273c',
                                                               plot_bgcolor='#24273c')
    # Dodanie czerwonej linii granicznej dla krytycznych obciążeń strukturalnych w gęstej atmosferze
    fig_stab.add_hline(y=10.0, line_dash="dash", line_color="red", annotation_text="Limit stabilności konstrukcji")

    # Obliczenie statystyk końcowych
    max_alt_km = df['altitude'].max() / 1000
    max_vel_ms = df['velocity'].max()
    max_aoa = df.loc[df['altitude'] < 50000, 'aoa'].max() if not df[df['altitude'] < 50000].empty else 0.0

    # Element UI informujący o statusie misji (Osiągnięcie orbity)
    if success:
        status_element = html.Div("🟢 SUKCES: Orbita LEO osiągnięta poprawnie!",
                                  style={'backgroundColor': '#00C851', 'color': 'white', 'marginTop': '25px',
                                         'padding': '15px', 'borderRadius': '5px', 'textAlign': 'center',
                                         'fontWeight': 'bold'})
    else:
        status_element = html.Div("🔴 NIEPOWODZENIE: Za niska prędkość lub wysokość",
                                  style={'backgroundColor': '#ffbb33', 'color': 'black', 'marginTop': '25px',
                                         'padding': '15px', 'borderRadius': '5px', 'textAlign': 'center',
                                         'fontWeight': 'bold'})

    return fig_3d, fig_traj, fig_alt, fig_vel, fig_fuel, fig_stab, f"{max_alt_km:.2f} km", f"{max_vel_ms:.1f} m/s", f"{max_aoa:.2f}°", status_element, pitch_start, target_pitch


if __name__ == '__main__':
    app.run(debug=True)