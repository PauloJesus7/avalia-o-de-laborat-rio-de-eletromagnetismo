import numpy as np
import matplotlib.pyplot as plt

class AnalisadorCavidadeFDTD:
    def __init__(self, largura=19.05, profundidade=9.53):
        # 1. Configuração do Guia Comercial (WR-75) [cite: 19]
        self.a = largura / 1000
        self.d = profundidade / 1000
        
        # Constantes Físicas para interior em ar [cite: 20]
        self.c = 299792458
        self.mu0 = 4 * np.pi * 1e-7
        self.eps0 = 8.854187e-12
        
        # 2. Malha Numérica (FDTD) [cite: 13]
        self.Nx, self.Ny = 60, 50
        self.dx, self.dy = self.a / self.Nx, self.d / self.Ny
        # Condição de estabilidade de Courant [cite: 13]
        self.dt = 0.99 / (self.c * np.sqrt((1/self.dx)**2 + (1/self.dy)**2))
        
        # 3. Inicialização dos Campos (Modo TM para Cavidade 2D) [cite: 21]
        self.Ez = np.zeros((self.Nx + 1, self.Ny + 1))
        self.Hx = np.zeros((self.Nx + 1, self.Ny))
        self.Hy = np.zeros((self.Nx, self.Ny + 1))
        
        self.historico_campo = []

    def simular(self, passos=8000):
        # Parâmetros da fonte de curta duração [cite: 28]
        t0 = 50 * self.dt
        sigma = 15 * self.dt
        
        print(f"Executando simulação numérica ({passos} passos)...")

        for n in range(passos):
            t = n * self.dt
            
            # Atualização do Campo Magnético (Lei de Faraday) [cite: 13]
            self.Hx[:, :] -= (self.dt / (self.mu0 * self.dy)) * (self.Ez[:, 1:] - self.Ez[:, :-1])
            self.Hy[:, :] += (self.dt / (self.mu0 * self.dx)) * (self.Ez[1:, :] - self.Ez[:-1, :])
            
            # Atualização do Campo Elétrico (Lei de Ampère) [cite: 13]
            # Condição PEC: Ez permanece zero em todas as paredes [cite: 27]
            self.Ez[1:-1, 1:-1] += (self.dt / self.eps0) * (
                (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) / self.dx - 
                (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1]) / self.dy
            )
            
            # Injeção do Pulso Gaussiano de banda larga [cite: 28]
            if n < 400:
                pulso = np.exp(-((t - t0)**2) / (2 * sigma**2))
                self.Ez[self.Nx//2, self.Ny//2] += pulso
            
            # Coleta de dados para identificação de modos 
            # Ponto estratégico fora do centro para evitar nós de modos superiores
            self.historico_campo.append(self.Ez[self.Nx//3, self.Ny//4])

        print("Simulação concluída. Gerando análise espectral...")
        self.gerar_graficos_finais()

    def gerar_graficos_finais(self):
        # --- ANÁLISE ESPECTRAL (FFT) --- 
        sinal = np.array(self.historico_campo)
        espectro = np.abs(np.fft.fft(sinal))
        freqs = np.fft.fftfreq(len(sinal), d=self.dt)
        
        # Filtro para visualização na faixa de operação de micro-ondas
        mask = (freqs > 0) & (freqs < 30e9)
        f_ghz, mag = freqs[mask]/1e9, espectro[mask]

        # JANELA 1: Espectro de Ressonância
        plt.figure("Espectro de Ressonância", figsize=(8, 4))
        plt.plot(f_ghz, mag, color='blue', linewidth=1.5)
        plt.title("Frequências de Ressonância")
        plt.xlabel("Frequência (GHz)")
        plt.ylabel("Magnitude (unid. abs.)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Identificar o pico principal automaticamente
        idx_pico = np.argmax(mag)
        plt.annotate(f'Modo Fundamental: {f_ghz[idx_pico]:.2f} GHz', 
                     xy=(f_ghz[idx_pico], mag[idx_pico]), 
                     xytext=(f_ghz[idx_pico]+2, mag[idx_pico]),
                     arrowprops=dict(facecolor='black', shrink=0.05))

        # JANELA 2: Padrão Estacionário [cite: 29, 30]
        plt.figure("Campo Estacionário", figsize=(7, 5))
        # Visualização da densidade de energia (módulo do campo)
        padrao_visual = np.abs(self.Ez) 
        plt.imshow(padrao_visual.T, cmap='magma', origin='lower',
                   extent=[0, self.a*1000, 0, self.d*1000])
        plt.title(" Campo Estacionário")
        plt.xlabel("Dimensão x (mm)")
        plt.ylabel("Dimensão y (mm)")
        plt.colorbar(label="|Ez|")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Configuração final para o guia WR-75 [cite: 19]
    analisador = AnalisadorCavidadeFDTD(largura=19.05, profundidade=9.53)
    analisador.simular()