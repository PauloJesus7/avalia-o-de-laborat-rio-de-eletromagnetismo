import numpy as np
import matplotlib.pyplot as plt
import cmath
import matplotlib

# Força o uso de uma janela externa para a animação funcionar no VS Code
try:
    matplotlib.use('TkAgg')
except:
    pass

class AntenaFendasCompleta:
    def __init__(self, largura=19.05, altura=9.53, frequencia=12e9):
        # Parâmetros Físicos (Padrão WR-75)
        self.a, self.b = largura / 1000, altura / 1000
        self.freq = frequencia
        self.pi = np.pi
        self.mu0, self.eps0 = 4 * self.pi * 1e-7, 8.854e-12
        self.c = 299792458
        self.omega = 2 * self.pi * self.freq

        # Cálculos de Propagação
        self.k = self.omega / self.c
        self.kc = np.sqrt((self.pi / self.a)**2)
        self.beta = cmath.sqrt(self.k**2 - self.kc**2).real
        self.lambda_g = (2 * self.pi / self.beta)

    def simular(self, comprimento_z=0.25, passos_tempo=2200):
        # 1. Configuração da Malha FDTD
        dx = self.a / 30
        dz = dx
        Nx_guia = int(self.a / dx)
        Nx_total = Nx_guia + 80 
        Nz = int(comprimento_z / dz)
        dt = 0.95 / (self.c * np.sqrt((1/dx)**2 + (1/dz)**2))

        Ey = np.zeros((Nx_total, Nz))
        Hx = np.zeros((Nx_total, Nz))
        Hz = np.zeros((Nx_total, Nz))

        # 2. Configuração das Fendas e Sensores de Campo Próximo
        x_parede = Nx_guia
        passo_fendas = int(self.lambda_g / dz)
        largura_fenda = int(0.004 / dz)
        
        indices_fendas_z = []
        parede_metalica = np.ones(Nz, dtype=bool)
        for f in range(1, 6):
            inicio = f * passo_fendas
            fim = inicio + largura_fenda
            if fim < Nz:
                parede_metalica[inicio:fim] = False
                indices_fendas_z.append((inicio + fim) // 2)

        # Monitoramento para Far-Field (Complexo para fase)
        captura_fendas = {idx: [] for idx in indices_fendas_z}

        # --- PREPARAÇÃO DA JANELA DE ANIMAÇÃO ---
        plt.ion() 
        fig, ax = plt.subplots(figsize=(7, 8))
        im = ax.imshow(Ey.T, cmap='seismic', vmin=-0.3, vmax=0.3, origin='lower',
                       extent=[0, Nx_total*dx*1000, 0, comprimento_z*1000])
        ax.set_title("FDTD: Campos Próximos e Propagação")
        ax.axvline(x=self.a*1000, color='black', linestyle='--')
        plt.show(block=False)

        # 3. Loop de Tempo
        for step in range(passos_tempo):
            t = step * dt
            
            # Atualização Maxwell
            Hx[:, :-1] += (dt / (self.mu0 * dz)) * (Ey[:, 1:] - Ey[:, :-1])
            Hz[:-1, :] -= (dt / (self.mu0 * dx)) * (Ey[1:, :] - Ey[:-1, :])
            Ey[1:-1, 1:-1] += (dt / self.eps0) * (
                (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dz - 
                (Hz[1:-1, 1:-1] - Hz[:-2, 1:-1]) / dx
            )
            
            # Condições PEC
            Ey[0, :] = 0 
            for z_idx in range(Nz):
                if parede_metalica[z_idx]: Ey[x_parede, z_idx] = 0 
            
            # Excitação
            perfil = np.sin(self.pi * np.linspace(0, self.a, Nx_guia) / self.a)
            Ey[0:Nx_guia, 1] = perfil * np.sin(self.omega * t)

            # Absorção
            Ey[-1, :] *= 0.96; Ey[:, -1] *= 0.96

            # Captura para Far-Field (Apenas no final da simulação para estabilidade)
            if step > passos_tempo - 400:
                for idx in indices_fendas_z:
                    captura_fendas[idx].append(Ey[x_parede + 2, idx])

            if step % 20 == 0:
                im.set_data(Ey.T)
                fig.canvas.draw()
                fig.canvas.flush_events()

        plt.ioff()
        plt.close()
        self.calcular_far_field(captura_fendas, indices_fendas_z, dz)

    def calcular_far_field(self, dados, indices_z, dz):
        # Converte sinais temporais para fasores (amplitude e fase)
        fasores = []
        for idx in indices_z:
            sinal = np.array(dados[idx])
            # FFT simplificada para pegar a componente na frequência de operação
            fft_val = np.fft.fft(sinal)
            fasores.append(fft_val[np.argmax(np.abs(fft_val))])

        # Ângulos de observação (-90 a 90 graus)
        theta = np.linspace(-np.pi/2, np.pi/2, 360)
        pattern = np.zeros(len(theta), dtype=complex)

        # Posições relativas das fendas em Z
        z_pos = np.array(indices_z) * dz

        # Transformação Far-Field: E(theta) = sum( Ai * exp(j * k * zi * sin(theta)) )
        for i, angle in enumerate(theta):
            for j, fasor in enumerate(fasores):
                # O termo exp(j*k*z*sin(theta)) representa o atraso de fase no campo distante
                pattern[i] += fasores[j] * np.exp(1j * self.k * z_pos[j] * np.sin(angle))

        # Normalização
        mag_pattern = np.abs(pattern)
        mag_pattern /= np.max(mag_pattern)

        # Plotagem dos Diagramas
        plt.figure(figsize=(12, 5))

        # 1. Diagrama Polar (Padrão de Irradiação)
        ax1 = plt.subplot(1, 2, 1, projection='polar')
        ax1.plot(theta, mag_pattern, color='red', linewidth=2)
        ax1.set_theta_zero_location("N") # Broadside para o topo
        ax1.set_title("Diagrama de Irradiação 2D (Polar)", pad=20)

        # 2. Diagrama Cartesiano (Ganho Relativo)
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(np.degrees(theta), 20 * np.log10(mag_pattern + 1e-6))
        ax2.set_title("Padrão de Irradiação (dB)")
        ax2.set_xlabel("Ângulo (graus)")
        ax2.set_ylabel("Ganho Relativo (dB)")
        ax2.set_ylim([-40, 0])
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    antena = AntenaFendasCompleta()
    antena.simular()