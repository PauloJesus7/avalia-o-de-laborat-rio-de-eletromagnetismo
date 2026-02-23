import numpy as np
import matplotlib.pyplot as plt
import cmath
import matplotlib

# Força o uso de uma janela externa para a animação funcionar no VS Code
try:
    matplotlib.use('TkAgg')
except:
    pass

class AntenaFendasVisual:
    def __init__(self, largura=19.05, altura=9.53, frequencia=12e9):
        # Parâmetros Físicos (Padrão WR-75)
        self.a, self.b = largura / 1000, altura / 1000
        self.freq = frequencia
        self.pi = np.pi
        self.mu0, self.eps0 = 4 * self.pi * 1e-7, 8.854e-12
        self.c = 299792458
        self.omega = 2 * self.pi * self.freq

        # Cálculos de Propagação (Modo TE10)
        self.kc = np.sqrt((self.pi / self.a)**2)
        self.k = self.omega / self.c
        self.beta = cmath.sqrt(self.k**2 - self.kc**2).real
        self.lambda_g = (2 * self.pi / self.beta)

    def simular_animacao(self, comprimento_z=0.25, passos_tempo=2000):
        # 1. Configuração da Malha FDTD
        dx = self.a / 30
        dz = dx
        Nx_guia = int(self.a / dx)
        Nx_total = Nx_guia + 80  # Espaço para ver a radiação externa
        Nz = int(comprimento_z / dz)
        dt = 0.95 / (self.c * np.sqrt((1/dx)**2 + (1/dz)**2))

        # Inicialização dos Campos
        Ey = np.zeros((Nx_total, Nz))
        Hx = np.zeros((Nx_total, Nz))
        Hz = np.zeros((Nx_total, Nz))

        # 2. Configuração das Fendas
        x_parede = Nx_guia
        passo_fendas = int(self.lambda_g / dz)
        largura_fenda = int(0.004 / dz)
        
        parede_metalica = np.ones(Nz, dtype=bool)
        for f in range(1, 6):
            inicio = f * passo_fendas
            fim = inicio + largura_fenda
            if fim < Nz:
                parede_metalica[inicio:fim] = False

        # --- PREPARAÇÃO DA JANELA GRÁFICA ---
        plt.ion() 
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Ajuste de escala (vmin/vmax) para realçar as ondas externas
        im = ax.imshow(Ey.T, cmap='seismic', vmin=-0.3, vmax=0.3, origin='lower',
                       extent=[0, Nx_total*dx*1000, 0, comprimento_z*1000])
        
        ax.set_title("Propagação Interna (TE10) e Irradiação por Fendas")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")
        ax.axvline(x=self.a*1000, color='black', linestyle='--', linewidth=2)
        
        plt.show(block=False)

        # 3. Loop de Tempo FDTD
        for step in range(passos_tempo):
            t = step * dt
            
            # --- ATUALIZAÇÃO DOS CAMPOS (CORREÇÃO DA SINTAXE) ---
            # Atualiza Campo Magnético (Hx e Hz)
            Hx[:, :-1] += (dt / (self.mu0 * dz)) * (Ey[:, 1:] - Ey[:, :-1])
            Hz[:-1, :] -= (dt / (self.mu0 * dx)) * (Ey[1:, :] - Ey[:-1, :])
            
            # Atualiza Campo Elétrico (Ey)
            Ey[1:-1, 1:-1] += (dt / self.eps0) * (
                (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dz - 
                (Hz[1:-1, 1:-1] - Hz[:-2, 1:-1]) / dx
            )
            
            # --- CONDIÇÕES DE CONTORNO PEC ---
            Ey[0, :] = 0  # Parede esquerda (metal)
            for z_idx in range(Nz):
                if parede_metalica[z_idx]:
                    Ey[x_parede, z_idx] = 0  # Parede direita (metal onde não há fenda)
            
            # --- EXCITAÇÃO MODO TE10 ---
            perfil = np.sin(self.pi * np.linspace(0, self.a, Nx_guia) / self.a)
            Ey[0:Nx_guia, 1] = perfil * np.sin(self.omega * t)

            # Absorção nas bordas (evita reflexões)
            Ey[-1, :] *= 0.95
            Ey[:, -1] *= 0.95

            # --- ATUALIZAÇÃO DO QUADRO ---
            if step % 15 == 0:
                im.set_data(Ey.T)
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

        plt.ioff()
        plt.show()

if __name__ == "__main__":
    antena = AntenaFendasVisual()
    antena.simular_animacao()