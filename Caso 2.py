import numpy as np
import matplotlib.pyplot as plt
from solvers import step_ssprk33

# PARAMETROS
MI_AGUA = 1.0e-3        # Viscosidade da água [Pa·s]
MI_GAS = 2.0e-5         # Viscosidade do gás [Pa·s]
PHI = 0.25              # Porosidade [-]
S_AGUA_RES = 0.2        # Saturação residual de água (Swc) [-]
S_GAS_RES = 0.0         # Saturação residual de gás (Sgr) [-]
A_ESPUMA = 400.0        # Parâmetro da textura de equilíbrio [-]
S_AGUA_CRIT = 0.37      # Saturação crítica de água (Sw*) [-]
K_C = 1.0e-6            # Constante de tempo [1/s]
N_MAX = 8.0e13          # Textura máxima de espuma [m⁻³]
U_X = 3.0e-6            # Velocidade em x [m/s]
U_Y = 1.0e-6               # Velocidade em y [m/s]
COMP_X = 1.0            # Comprimento [m]
COMP_Y = 1.0            # Altura [m]
NUM_X = 100             # Número de células em x
NUM_Y = 30              # Número de células em y
DX = COMP_X / NUM_X
DY = COMP_Y / NUM_Y

# CONDIÇÕES INICIAIS E DE CONTORNO
S_AGUA_INIC = 0.9       # Saturação inicial de água [-]
N_D_INIC = 0.0          # Textura inicial de espuma [-]
S_AGUA_INJ = 0.372      # Saturação de água injetada [-]
N_D_INJ = 0.0           # Textura de espuma injetada [-]
TEMPO_FINAL = 10000.0   # Tempo final [s]
DT_INIC = 1.0           # Passo de tempo inicial [s]

# =============================================================================
def minmod(a, b):
    return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)

def reconstruir_com_minmod(u):
    delta_u = u[:, 1:] - u[:, :-1]
    delta_u_esq = np.zeros_like(u)
    delta_u_esq[:, 1:-1] = minmod(delta_u[:, :-1], delta_u[:, 1:])
    delta_u_esq[:, 0] = minmod(0, delta_u[:, 0])
    delta_u_esq[:, -1] = minmod(delta_u[:, -2], 0)

    u_L = u[:, :-1] + 0.5 * delta_u_esq[:, :-1]
    u_R = u[:, 1:] - 0.5 * delta_u_esq[:, 1:]

    return u_L, u_R

def perm_rel_agua(S_agua):
    S_ef = (S_agua - S_AGUA_RES) / (1.0 - S_AGUA_RES - S_GAS_RES)
    return S_ef ** 4

def perm_rel_gas_sem_espuma(S_agua):
    S_gas = 1.0 - S_agua
    S_ef = (S_gas - S_GAS_RES) / (1.0 - S_AGUA_RES - S_GAS_RES)
    return S_ef ** 2

def perm_rel_gas(S_agua, n_D):
    krg0 = perm_rel_gas_sem_espuma(S_agua)
    return krg0 / (18500.0 * n_D + 1.0)

def mobilidade_agua(S_agua):
    return perm_rel_agua(S_agua) / MI_AGUA

def mobilidade_gas(S_agua, n_D):
    return perm_rel_gas(S_agua, n_D) / MI_GAS

def fluxo_frac_agua(S_agua, n_D):
    lambda_agua = mobilidade_agua(S_agua)
    lambda_gas = mobilidade_gas(S_agua, n_D)
    lambda_total = lambda_agua + lambda_gas
    return (lambda_agua / lambda_total)

def fluxo_frac_gas(S_agua, n_D):
    return 1.0 - fluxo_frac_agua(S_agua, n_D)

def n_equilibrio(S_agua):
    return np.where(S_agua > S_AGUA_CRIT,
                    np.tanh(A_ESPUMA * (S_agua - S_AGUA_CRIT)),
                    0.0)

def taxa_geracao_espuma(S_agua, n_D):
    return K_C * N_MAX * (n_equilibrio(S_agua) - n_D)

# =============================================================================
class SolverTransporteEspuma:
    def __init__(self):
        self.S_agua = np.full((NUM_Y, NUM_X), S_AGUA_INIC)
        self.n_D = np.full((NUM_Y, NUM_X), N_D_INIC)
        self.tempo = 0.0
        self.num_celulas = NUM_Y * NUM_X

    def calcular_lado_direito(self, t, u):
        S_agua = u[:self.num_celulas].reshape(NUM_Y, NUM_X)
        n_D = u[self.num_celulas:].reshape(NUM_Y, NUM_X)

        # Reconstrução com minmod
        S_agua_L, S_agua_R = reconstruir_com_minmod(S_agua)
        n_D_L, n_D_R = reconstruir_com_minmod(n_D)

        # Cálculo dos fluxos nas interfaces
        f_agua_L = fluxo_frac_agua(S_agua_L, n_D_L)
        f_agua_R = fluxo_frac_agua(S_agua_R, n_D_R)
        f_gas_L = fluxo_frac_gas(S_agua_L, n_D_L)
        f_gas_R = fluxo_frac_gas(S_agua_R, n_D_R)

        # Velocidade local (a_{i+1/2})
        a_agua = np.maximum(np.abs(f_agua_L), np.abs(f_agua_R))
        a_gas = np.maximum(np.abs(f_gas_L), np.abs(f_gas_R))

        # Fluxo numérico KNP
        fluxo_agua_x = 0.5 * (f_agua_L + f_agua_R) * U_X - 0.5 * a_agua * (S_agua_R - S_agua_L)
        fluxo_espuma_x = 0.5 * (f_gas_L * n_D_L + f_gas_R * n_D_R) * U_X - 0.5 * a_gas * (n_D_R - n_D_L)

        # Tratamento de contorno
        if U_X > 0:
            fluxo_agua_x[:, 0] = fluxo_frac_agua(S_AGUA_INJ, N_D_INJ) * U_X
            fluxo_espuma_x[:, 0] = fluxo_frac_gas(S_AGUA_INJ, N_D_INJ) * U_X * N_D_INJ
        else:
            fluxo_agua_x[:, -1] = 0.0
            fluxo_espuma_x[:, -1] = 0.0

        # Divergência dos fluxos
        div_fluxo_agua = np.zeros_like(S_agua)
        div_fluxo_agua[:, 1:-1] = (fluxo_agua_x[:, :-1] - fluxo_agua_x[:, 1:]) / DX
        div_fluxo_agua[:, 0] = (fluxo_agua_x[:, 0] - fluxo_agua_x[:, 1]) / DX
        div_fluxo_agua[:, -1] = fluxo_agua_x[:, -1] / DX

        div_fluxo_espuma = np.zeros_like(S_agua)
        div_fluxo_espuma[:, 1:-1] = (fluxo_espuma_x[:, :-1] - fluxo_espuma_x[:, 1:]) / DX
        div_fluxo_espuma[:, 0] = (fluxo_espuma_x[:, 0] - fluxo_espuma_x[:, 1]) / DX
        div_fluxo_espuma[:, -1] = fluxo_espuma_x[:, -1] / DX

        # Fonte de espuma
        fonte_espuma = PHI * (1.0 - S_agua) * taxa_geracao_espuma(S_agua, n_D) / N_MAX

        # Derivadas temporais
        dS_agua_dt = div_fluxo_agua / PHI
        S_gas = 1.0 - S_agua
        d_SgnD_dt = (div_fluxo_espuma + fonte_espuma) / PHI
        dn_D_dt = (d_SgnD_dt + n_D * dS_agua_dt) / S_gas

        return np.concatenate([dS_agua_dt.ravel(), dn_D_dt.ravel()])

    def avancar_passo(self, dt):
        u_n = np.concatenate([self.S_agua.ravel(), self.n_D.ravel()])
        u_novo = step_ssprk33(self.calcular_lado_direito, u_n, self.tempo, dt)
        self.S_agua = u_novo[:self.num_celulas].reshape(NUM_Y, NUM_X)
        self.n_D = u_novo[self.num_celulas:].reshape(NUM_Y, NUM_X)
        self.tempo += dt

    def executar(self, tempo_final, dt, num_snapshots=10):
        snapshots = []
        tempos_snapshot = np.linspace(0, tempo_final, num_snapshots)
        idx_snapshot = 0
        snapshots.append({
            'S_agua': self.S_agua.copy(),
            'n_D': self.n_D.copy(),
            'tempo': self.tempo
        })
        idx_snapshot += 1

        while self.tempo < tempo_final:
            dt_cfl = 0.5 * DX / U_X
            dt_atual = min(dt, dt_cfl, tempo_final - self.tempo)

            self.avancar_passo(dt_atual)

            if idx_snapshot < len(tempos_snapshot) and self.tempo >= tempos_snapshot[idx_snapshot]:
                snapshots.append({
                    'S_agua': self.S_agua.copy(),
                    'n_D': self.n_D.copy(),
                    'tempo': self.tempo
                })

                idx_snapshot += 1

        return snapshots

# =============================================================================

def plotar_resultados(snapshots):
    coord_x = np.linspace(0, COMP_X, NUM_X)
    coord_y = np.linspace(0, COMP_Y, NUM_Y)
    y_meio = NUM_Y // 2
    x_meio = NUM_X // 2

    num_tempos = len(snapshots)
    cores = plt.cm.viridis(np.linspace(0, 1, num_tempos))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, snap in enumerate(snapshots):
        rotulo = f"t = {snap['tempo']:.0f} s"

        axes[0, 0].plot(coord_x, snap['S_agua'][y_meio, :], color=cores[i], label=rotulo)
        axes[0, 1].plot(coord_y, snap['S_agua'][:, x_meio], color=cores[i], label=rotulo)
        axes[1, 0].plot(coord_x, snap['n_D'][y_meio, :], color=cores[i], label=rotulo)
        axes[1, 1].plot(coord_y, snap['n_D'][:, x_meio], color=cores[i], label=rotulo)

    axes[0, 0].set_xlabel('x [m]')
    axes[0, 0].set_ylabel('$S_w$')
    axes[0, 0].set_title('Saturação de água ao longo de x')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0.15, 1.05])

    axes[0, 1].set_xlabel('y [m]')
    axes[0, 1].set_ylabel('$S_w$')
    axes[0, 1].set_title('Saturação de água ao longo de y')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('x [m]')
    axes[1, 0].set_ylabel('$n_D$')
    axes[1, 0].set_title('Textura de espuma ao longo de x')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([-0.05, 1.05])

    axes[1, 1].set_xlabel('y [m]')
    axes[1, 1].set_ylabel('$n_D$')
    axes[1, 1].set_title('Textura de espuma ao longo de y')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('resultados.png', dpi=150)
    print("\nFigura salva: resultados.png")

# =============================================================================

if __name__ == "__main__":

    solver = SolverTransporteEspuma()
    snapshots = solver.executar(TEMPO_FINAL, DT_INIC, num_snapshots=10)

    plotar_resultados(snapshots)