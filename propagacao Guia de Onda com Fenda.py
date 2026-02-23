import numpy as np
import matplotlib.pyplot as plt
import cmath

# Força o uso de janela externa para garantir que a animação abra
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

class SimularPropagacaoFendas:
    def __init__(self, largura=19.05, altura=9.53, frequencia=12e9):
        # Parâmetros Físicos (WR-75)
        self.a, self.b = largura / 1000, altura / 1000
        self.freq = frequencia
        self.pi = np.pi
        self.mu0, self.eps0 = 4 * self.pi * 1e-7, 8.854e-12
        self.c = 299792458
        self.omega = 2 * self.pi * self.freq

        # Cálculos de Propagação (Modo Dominante TE10)
        self.kc = np.sqrt((self.pi / self.a)**2)
        self.k = self.omega / self.c
        self.beta = cmath.sqrt(self.k**2 - self.kc**2).real
        self.lambda_g = (2 * self.pi / self.beta)
        
        print(f"Iniciando Simulação: {self.freq/1e9:.1f} GHz")
        print(f"Lambda Guiado: {self.lambda_g*1000:.2f} mm")

    def executar_visualizacao(self, comprimento_z=0.25, passos_tempo=1500):
        # Malha FDTD
        dx = self.a / 30
        dz = dx
        Nx_guia = int(self.a / dx)
        Nx_total = Nx_guia + 60 
        Nz = int(comprimento_z / dz)
        dt = 0.95 / (self.c * np.sqrt((1/dx)**2 + (1/dz)**2))

        Ey = np.zeros((Nx_total, Nz))
        Hx = np.zeros((Nx_total, Nz))
        Hz = np.zeros((Nx_total, Nz))

        # Configuração das Fendas (Interrupção PEC)
        x_parede = Nx_guia
        passo_fendas = int(self.lambda_g / dz)
        largura_fenda = int(0.004 / dz)
        
        parede_metalica = np.ones(Nz, dtype=bool)
        for f in range(1, 6):
            inicio = f * passo_fendas
            fim = inicio + largura_fenda
            if fim < Nz:
                parede_metalica[inicio:fim] = False

        # Preparação do Gráfico 2D
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(Ey.T, cmap='seismic', vmin=-0.8, vmax=0.8, origin='lower',
                        extent=[0, Nx_total*dx*1000, 0, comprimento_z*1000])
        
        ax.set_title("Simulação FDTD: Propagação TE10 e Radiação por Fendas")
        ax.set_xlabel("X (mm) - Largura + Espaço de Radiação")
        ax.set_ylabel("Z (mm) - Comprimento do Guia")
        plt.colorbar(im, label="Amplitude Ey")
        
        # Linha pontilhada indicando a parede com fendas
        ax.axvline(x=self.a*1000, color='k', linestyle='--', alpha=0.5)

        # Loop de Propagação
        for step in range(passos_tempo):
            t = step * dt
            
            # Atualização de Maxwell
            Hx[:, :-1] += (dt / (self.mu0 * dz)) * (Ey[:, 1:] - Ey[:, :-1])
            Hz[:-1, :] -= (dt / (self.mu0 * dx)) * (Ey[1:, :] - Ey[:-1, :])
            Ey[1:-1, 1:-1] += (dt / self.eps0) * (
                (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dz - (Hz[1:-1, 1:-1] - Hz[:-2, 1:-1]) / dx
            )
            
            # Paredes PEC
            Ey[0, :] = 0
            for z_idx in range(Nz):
                if parede_metalica[z_idx]: 
                    Ey[x_parede, z_idx] = 0
            
            # Excitação Senoidal Permanente (TE10)
            perfil = np.sin(self.pi * np.linspace(0, self.a, Nx_guia) / self.a)
            Ey[0:Nx_guia, 1] = perfil * np.sin(self.omega * t)

            # Atualização da Animação
            if step % 20 == 0:
                im.set_data(Ey.T)
                plt.pause(0.001)

        plt.ioff()
        plt.show()

if __name__ == "__main__":
    sim = SimularPropagacaoFendas()
    sim.executar_visualizacao()