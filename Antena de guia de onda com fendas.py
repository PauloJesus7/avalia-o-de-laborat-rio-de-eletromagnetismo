import numpy as np
import matplotlib.pyplot as plt
import cmath

class AntenaGuiaFendasAnalitica:
    def __init__(self, largura=19.05, altura=9.53, frequencia=12e9):
        self.a, self.b = largura / 1000, altura / 1000
        self.freq = frequencia
        self.pi = np.pi
        self.mu0, self.eps0 = 4 * self.pi * 1e-7, 8.854e-12
        self.c = 1 / np.sqrt(self.mu0 * self.eps0)
        self.omega = 2 * self.pi * self.freq

        # Cálculos de Propagação
        self.kc = np.sqrt((self.pi / self.a)**2)
        self.k = self.omega / self.c
        self.beta = cmath.sqrt(self.k**2 - self.kc**2).real
        self.lambda_g = (2 * self.pi / self.beta)
        
        # Armazenamento para análise quantitativa
        self.dados_fendas = [] 

    def simular(self, comprimento_z=0.25, passos_tempo=1500):
        dx = self.a / 30
        dz = dx
        Nx_guia = int(self.a / dx)
        Nx_total = Nx_guia + 60 
        Nz = int(comprimento_z / dz)
        dt = 0.95 / (self.c * np.sqrt((1/dx)**2 + (1/dz)**2))

        Ey = np.zeros((Nx_total, Nz))
        Hx = np.zeros((Nx_total, Nz))
        Hz = np.zeros((Nx_total, Nz))

        # Configuração das Fendas e Sensores
        x_parede = Nx_guia
        passo_fendas = int(self.lambda_g / dz)
        largura_fenda = int(0.004 / dz)
        indices_fendas = []
        
        parede_metalica = np.ones(Nz, dtype=bool)
        for f in range(1, 6):
            inicio = f * passo_fendas
            fim = inicio + largura_fenda
            if fim < Nz:
                parede_metalica[inicio:fim] = False
                indices_fendas.append((inicio + fim) // 2) # Centro da fenda para monitorar

        # Matrizes para capturar o sinal temporal em cada fenda
        sinais_fendas = {idx: [] for idx in indices_fendas}

        # Loop FDTD
        for step in range(passos_tempo):
            t = step * dt
            
            # Maxwell
            Hx[:, :-1] += (dt / (self.mu0 * dz)) * (Ey[:, 1:] - Ey[:, :-1])
            Hz[:-1, :] -= (dt / (self.mu0 * dx)) * (Ey[1:, :] - Ey[:-1, :])
            Ey[1:-1, 1:-1] += (dt / self.eps0) * (
                (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dz - (Hz[1:-1, 1:-1] - Hz[:-2, 1:-1]) / dx
            )
            
            # Condições de Contorno e Excitação
            Ey[0, :] = 0
            for z_idx in range(Nz):
                if parede_metalica[z_idx]: Ey[x_parede, z_idx] = 0
            
            # Absorção simples nas bordas externas
            Ey[-1, :] *= 0.98; Ey[:, -1] *= 0.98
            
            # Injeção de sinal senoidal estável para análise de fase
            perfil = np.sin(self.pi * np.linspace(0, self.a, Nx_guia) / self.a)
            Ey[0:Nx_guia, 1] = perfil * np.sin(self.omega * t)

            # Quantificação: Captura o campo saindo de cada fenda
            for idx in indices_fendas:
                sinais_fendas[idx].append(Ey[x_parede + 2, idx])

        self.analisar_resultados(sinais_fendas, dt, indices_fendas)

    def analisar_resultados(self, sinais, dt, indices):
        plt.figure(figsize=(12, 5))
        
        # 1. Gráfico de Sinais Temporais (Comparação de Fase)
        plt.subplot(1, 2, 1)
        for i, idx in enumerate(indices):
            sinal = sinais[idx][-400:] # Pega os últimos ciclos
            plt.plot(sinal, label=f'Fenda {i+1}')
        
        plt.title("Interferência Construtiva")
        plt.xlabel("Passos de tempo (últimos ciclos)")
        plt.ylabel("Amplitude Ey (V/m)")
        plt.legend()
        plt.grid(True)

        # 2. Gráfico de Vazamento de Energia (Quantificação)
        plt.subplot(1, 2, 2)
        energias = [np.max(np.abs(sinais[idx])) for idx in indices]
        plt.bar([f'Fenda {i+1}' for i in range(len(indices))], energias, color='darkred')
        plt.title("Vazamento de Energia")
        plt.ylabel("Amplitude Máxima Capturada")
        
        plt.tight_layout()
        plt.show()
        
        # Verificação Teórica de Interferência
        print("\n--- RELATÓRIO TÉCNICO DE INTERFERÊNCIA ---")
        print(f"Espaçamento entre fendas: {self.lambda_g*1000:.2f} mm (λg)")
        print("Observação: Se os picos das fendas no gráfico 1 estão alinhados,")
        print("temos INTERFERÊNCIA CONSTRUTIVA na direção broadside (90°).")

if __name__ == "__main__":
    antena = AntenaGuiaFendasAnalitica()
    antena.simular()