import matplotlib
# FORÇA A ABERTURA DE JANELA EXTERNA - Deve ser a primeira linha
matplotlib.use('TkAgg') 

import numpy as np
import matplotlib.pyplot as plt

class AnimacaoFDTD:
    def __init__(self, largura=19.05, profundidade=9.53):
        # Dimensões WR-75 em metros
        self.a, self.d = largura / 1000, profundidade / 1000
        self.c = 299792458
        self.mu0, self.eps0 = 4*np.pi*1e-7, 8.854e-12
        
        # Malha FDTD
        self.Nx, self.Ny = 60, 50
        self.dx, self.dy = self.a / self.Nx, self.d / self.Ny
        self.dt = 0.95 / (self.c * np.sqrt((1/self.dx)**2 + (1/self.dy)**2))
        
        # Campos Iniciais
        self.Ez = np.zeros((self.Nx + 1, self.Ny + 1))
        self.Hx = np.zeros((self.Nx + 1, self.Ny))
        self.Hy = np.zeros((self.Nx, self.Ny + 1))

    def executar(self, passos=15000):
        t0, sigma = 50 * self.dt, 15 * self.dt
        
        # Criar a figura e garantir que ela apareça
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Configuração da imagem
        im = ax.imshow(self.Ez.T, cmap='RdBu', origin='lower',
                        extent=[0, self.a*1000, 0, self.d*1000],
                        vmin=-0.1, vmax=0.1)
        
        ax.set_title(" Animação 2D: Ressonância na Cavidade")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, label="Amplitude Ez")
        
        # Forçar a janela a saltar para a frente
        fig.show()
        plt.show(block=False)

        print("Simulando... A janela deve estar aberta agora.")

        for n in range(passos):
            # Equações de Maxwell (FDTD)
            self.Hx[:, :] -= (self.dt / (self.mu0 * self.dy)) * (self.Ez[:, 1:] - self.Ez[:, :-1])
            self.Hy[:, :] += (self.dt / (self.mu0 * self.dx)) * (self.Ez[1:, :] - self.Ez[:-1, :])
            
            # Ez com PEC nas bordas
            self.Ez[1:-1, 1:-1] += (self.dt / self.eps0) * (
                (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) / self.dx - 
                (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1]) / self.dy
            )
            
            # Fonte Gaussiana (Pulso Curto)
            if n < 400:
                t = n * self.dt
                pulso = np.exp(-((t - t0)**2) / (2 * sigma**2))
                self.Ez[self.Nx//2, self.Ny//2] += pulso
            
            # Atualização visual a cada 40 passos (mais rápido/fluido)
            if n % 40 == 0:
                im.set_data(self.Ez.T)
                
                # Ajuste automático de escala para manter o padrão modal visível
                v_max = np.max(np.abs(self.Ez))
                if v_max > 1e-4:
                    im.set_clim(-v_max, v_max)
                
                # Comandos críticos para o Windows não travar a janela
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

        print("Simulação concluída.")
        plt.ioff()
        plt.show() # Mantém a janela aberta no final

if __name__ == "__main__":
    app = AnimacaoFDTD()
    app.executar()